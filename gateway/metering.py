import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx


class Metering:
    def __init__(self, config: dict):
        settings = config.get("metering", {}) or {}
        self.enabled = bool(settings.get("enabled", True))
        self.history_size = int(settings.get("history_size", 500))
        self.path = Path(settings.get("persist_path", "/app/data/metering.json"))
        pricing = settings.get("pricing", {}) or {}
        self.pricing_url = pricing.get("url", "https://models.dev/api.json")
        self.pricing_refresh = int(pricing.get("refresh_interval_seconds", 86400))
        self.pricing_data: dict[str, Any] = {}
        self.pricing_expires = 0.0
        self.data = self._load()

    def _empty(self) -> dict:
        return {
            "requests": 0,
            "compressed_requests": 0,
            "input_chars": 0,
            "output_chars": 0,
            "records": [],
        }

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, dict):
                base = self._empty()
                base.update(data)
                return base
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        return self._empty()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self.path.parent, prefix="metering-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, separators=(",", ":"))
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    async def _pricing(self) -> dict:
        now = time.time()
        if self.pricing_data and now < self.pricing_expires:
            return self.pricing_data
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(self.pricing_url)
                response.raise_for_status()
                self.pricing_data = response.json()
                self.pricing_expires = now + self.pricing_refresh
        except (httpx.HTTPError, ValueError):
            self.pricing_expires = now + 300
        return self.pricing_data

    async def model_info(self, provider: str, model: str) -> dict:
        data = await self._pricing()
        providers = data.get("providers", data) if isinstance(data, dict) else {}
        provider_data = providers.get(provider, {}) if isinstance(providers, dict) else {}
        models = provider_data.get("models", {}) if isinstance(provider_data, dict) else {}
        return models.get(model, {}) if isinstance(models, dict) else {}

    @staticmethod
    def _cost(tokens: int, price: float | None) -> float:
        return tokens * float(price or 0) / 1_000_000

    async def record(
        self,
        *,
        session: str,
        parent_session: str | None,
        upstream: str,
        model: str,
        original_chars: int,
        compressed_chars: int,
        compressed: bool,
        usage: dict | None,
        original_messages: list[dict] | None = None,
        compressed_messages: list[dict] | None = None,
        error_status: int | None = None,
    ) -> dict | None:
        if not self.enabled:
            return None
        usage = usage or {}
        details = usage.get("prompt_tokens_details", {}) or {}
        cache_read = int(usage.get("cache_read_input_tokens", details.get("cached_tokens", 0)) or 0)
        cache_write = int(usage.get("cache_write_input_tokens", 0) or 0)
        prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
        original_tokens = max(1, round(original_chars / 4)) if original_chars else 0
        info = await self.model_info("opencode-go" if upstream == "opencode-go" else upstream, model)
        costs = info.get("cost", {}) or {}
        limits = info.get("limit", {}) or {}
        input_price = costs.get("input", 0)
        output_price = costs.get("output", 0)
        reasoning_price = costs.get("reasoning", output_price)
        cache_read_price = costs.get("cache_read", 0)
        cache_write_price = costs.get("cache_write", 0)
        original_prompt_cost = self._cost(original_tokens, input_price)
        billed_prompt_cost = self._cost(prompt_tokens, input_price)
        billed_output_cost = self._cost(completion_tokens, output_price)
        billed_reasoning_cost = self._cost(reasoning_tokens, reasoning_price)
        billed_cache_read_cost = self._cost(cache_read, cache_read_price)
        billed_cache_write_cost = self._cost(cache_write, cache_write_price)
        billed_cost = (
            billed_prompt_cost
            + billed_output_cost
            + billed_reasoning_cost
            + billed_cache_read_cost
            + billed_cache_write_cost
        )
        record = {
            "timestamp": int(time.time()),
            "session": session or "unattributed",
            "parent_session": parent_session,
            "upstream": upstream,
            "model": model,
            "compressed": compressed,
            "original_chars": original_chars,
            "compressed_chars": compressed_chars,
            "original_tokens_estimate": original_tokens,
            "billed_prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "context_limit": limits.get("context"),
            "input_price_per_million": input_price,
            "output_price_per_million": output_price,
            "original_prompt_cost_usd": original_prompt_cost,
            "billed_prompt_cost_usd": billed_prompt_cost,
            "billed_output_cost_usd": billed_output_cost,
            "billed_reasoning_cost_usd": billed_reasoning_cost,
            "billed_cache_read_cost_usd": billed_cache_read_cost,
            "billed_cache_write_cost_usd": billed_cache_write_cost,
            "billed_cost_usd": billed_cost,
            "saved_prompt_cost_usd": max(0, original_prompt_cost - billed_prompt_cost),
        }
        if error_status is not None:
            record["error_status"] = error_status
        if original_messages is not None:
            record["original_messages"] = original_messages
        if compressed_messages is not None:
            record["compressed_messages"] = compressed_messages
        self.data["requests"] += 1
        self.data["compressed_requests"] += int(compressed)
        self.data["input_chars"] += original_chars
        self.data["output_chars"] += compressed_chars
        self.data["records"].append(record)
        self.data["records"] = self.data["records"][-self.history_size:]
        self._save()
        return record

    def _records_for(self, session: str | None) -> list[dict]:
        records = self.data.get("records", [])
        if session:
            return [r for r in records if r.get("session") == session]
        return records

    @staticmethod
    def _summary(records: list[dict]) -> dict:
        if not records:
            return {"requests": 0}
        def total(key):
            return sum(float(r.get(key, 0) or 0) for r in records)
        original_tokens = total("original_tokens_estimate")
        billed_tokens = total("billed_prompt_tokens")
        original_cost = total("original_prompt_cost_usd")
        billed_cost = total("billed_cost_usd")
        return {
            "session": records[-1].get("session"),
            "requests": len(records),
            "compressed_requests": sum(bool(r.get("compressed")) for r in records),
            "original_chars": int(total("original_chars")),
            "compressed_chars": int(total("compressed_chars")),
            "original_tokens_estimate": int(original_tokens),
            "billed_prompt_tokens": int(billed_tokens),
            "completion_tokens": int(total("completion_tokens")),
            "reasoning_tokens": int(total("reasoning_tokens")),
            "cache_read_tokens": int(total("cache_read_tokens")),
            "cache_write_tokens": int(total("cache_write_tokens")),
            "original_prompt_cost_usd": original_cost,
            "billed_cost_usd": billed_cost,
            "saved_prompt_cost_usd": max(0, original_cost - total("billed_prompt_cost_usd")),
            "savings_percent": round(max(0, original_tokens - billed_tokens) / original_tokens * 100, 2) if original_tokens else 0,
            "context_limit": records[-1].get("context_limit"),
        }

    def stats(self) -> dict:
        return self._summary(self.data.get("records", [])) | {
            "requests": self.data.get("requests", 0),
            "compressed_requests": self.data.get("compressed_requests", 0),
            "input_chars": self.data.get("input_chars", 0),
            "output_chars": self.data.get("output_chars", 0),
        }

    def sessions(self, limit: int = 20) -> list[dict]:
        sessions = []
        seen = set()
        for record in reversed(self.data.get("records", [])):
            session = record.get("session", "unattributed")
            if session in seen:
                continue
            seen.add(session)
            sessions.append(self._summary(self._records_for(session)))
            if len(sessions) >= limit:
                break
        return sessions

    def report(self, session: str | None = None) -> dict:
        records = self._records_for(session)
        if not records:
            return {"requests": 0, "session": session}
        public = []
        for record in records[-20:]:
            item = dict(record)
            item.pop("original_messages", None)
            item.pop("compressed_messages", None)
            public.append(item)
        return self._summary(records) | {"records": public}

    def latest(self) -> dict | None:
        records = self.data.get("records", [])
        return records[-1] if records else None
