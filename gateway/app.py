import os
import json
import logging
import re
import time
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from mcp_server import create_mcp
from metering import Metering

CONFIG_FILE = os.getenv("CONFIG_FILE", "/app/config.yml")
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")
LINGUA_URL = os.getenv("LINGUA_URL", "http://lingua:8001").rstrip("/")

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f) or {}

app = FastAPI(title="Lingua OpenCode Gateway", version="0.1.0")
metering = Metering(CONFIG)
logger = logging.getLogger("uvicorn.error")

_model_cache: dict[str, dict[str, Any]] = {}


def auth_ok(authorization: str | None, x_api_key: str | None = None) -> bool:
    if not GATEWAY_API_KEY:
        return True
    return (
        authorization == f"Bearer {GATEWAY_API_KEY}"
        or x_api_key == GATEWAY_API_KEY
    )


def require_auth(authorization: str | None, x_api_key: str | None = None):
    if not auth_ok(authorization, x_api_key):
        presented = authorization or (f"(x-api-key) {x_api_key}" if x_api_key else None)
        raise HTTPException(status_code=401, detail=f"Invalid gateway API key {presented}")


def upstream_config(upstream_id: str) -> dict:
    upstreams = CONFIG.get("upstreams", {}) or {}
    upstream = upstreams.get(upstream_id)
    if not isinstance(upstream, dict) or not upstream.get("base_url"):
        raise HTTPException(status_code=400, detail=f"Unknown upstream: {upstream_id}")
    return upstream


def select_upstream(requested: str | None) -> tuple[str, dict]:
    upstream_id = requested or CONFIG.get("routing", {}).get("default_upstream", "zen")
    return upstream_id, upstream_config(upstream_id)


def forwarded_headers(request: Request, upstream: dict) -> dict[str, str]:
    names = upstream.get(
        "forward_headers",
        CONFIG.get("routing", {}).get("forward_headers", []),
    ) or []
    allowed = {str(name).lower() for name in names}
    return {
        name: value
        for name, value in request.headers.items()
        if name.lower() in allowed
    }


def strip_thinking_block_binding(body: dict) -> dict:
    """Drop OpenCode's thinking block-binding control from Anthropic requests.

    OpenCode v2.3.2 adds ``thinking.block_binding`` (with
    ``prefix_mismatch_behavior``) to Claude requests. Anthropic-compatible
    upstreams such as OpenCode Go reject that optional control hint with
    ``thinking.adaptive.block_binding: Extra inputs are not permitted``.
    Only the ``thinking`` object is touched: messages, tools, and reasoning
    content are preserved.
    """

    def strip(value):
        if isinstance(value, dict):
            return {
                key: strip(item)
                for key, item in value.items()
                if key != "block_binding"
            }
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    thinking = body.get("thinking")
    if not isinstance(thinking, dict):
        return body
    cleaned = strip(thinking)
    if cleaned == thinking:
        return body
    result = dict(body)
    result["thinking"] = cleaned
    return result


def strip_thinking_binding_beta(headers: dict[str, str]) -> dict[str, str]:
    """Remove the thinking-binding-controls token from anthropic-beta."""
    for name, value in list(headers.items()):
        if name.lower() != "anthropic-beta":
            continue
        tokens = [
            token.strip()
            for token in value.split(",")
            if not token.strip().startswith("thinking-binding-controls-")
        ]
        if tokens:
            headers[name] = ", ".join(tokens)
        else:
            headers.pop(name, None)
    return headers


def upstream_headers(
    upstream: dict,
    forwarded: dict[str, str] | None = None,
    stream: bool = False,
) -> dict[str, str]:
    headers = dict(upstream.get("headers", {}) or {})
    headers.update(forwarded or {})
    api_key_env = upstream.get("api_key_env")
    api_key = os.getenv(api_key_env, "") if api_key_env else ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.setdefault("Accept", "text/event-stream" if stream else "application/json")
    return headers


async def upstream_request(
    upstream: dict,
    method: str,
    path: str,
    forwarded: dict[str, str] | None = None,
    stream: bool = False,
    **kwargs,
):
    headers = upstream_headers(upstream, forwarded, stream)
    headers.update(kwargs.pop("headers", {}))
    async with httpx.AsyncClient(timeout=300) as client:
        return await client.request(
            method,
            f"{upstream['base_url'].rstrip('/')}{path}",
            headers=headers,
            **kwargs,
        )


async def get_models(upstream_id: str, force: bool = False) -> dict:
    upstream = upstream_config(upstream_id)
    now = time.time()
    refresh = int(upstream.get(
        "refresh_interval_seconds",
        CONFIG.get("routing", {}).get("refresh_interval_seconds", 900),
    ))
    cached = _model_cache.get(upstream_id)
    if not force and cached and now < cached["expires"]:
        return cached["data"]

    models_path = upstream.get("models_path")
    if not models_path:
        return {"object": "list", "data": []}
    response = await upstream_request(upstream, "GET", models_path)
    response.raise_for_status()
    data = response.json()
    _model_cache[upstream_id] = {"expires": now + refresh, "data": data}
    return data


def model_ids(models: dict) -> set[str]:
    return {x.get("id") for x in models.get("data", []) if x.get("id")}


def protocol_for(upstream_id: str, model: str) -> str:
    upstream = upstream_config(upstream_id)
    for rule in upstream.get("protocol_rules", []) or []:
        if re.search(rule.get("pattern", ""), model):
            return rule.get("protocol", upstream.get("default_protocol", "chat"))
    return upstream.get("default_protocol", "chat")


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def message_text(message: dict) -> str:
    return extract_text(message.get("content"))


TOOL_ITEM_TYPES = {
    "tool-call",
    "tool-result",
    "tool_use",
    "tool_result",
    "function_call",
    "function_result",
    "tool_use_result",
}


def is_tool_structural(message: dict) -> bool:
    if message.get("role") == "tool":
        return True
    if message.get("tool_calls") or message.get("tool_call_id"):
        return True
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(item, dict) and item.get("type") in TOOL_ITEM_TYPES
        for item in content
    )


def structural_messages(messages: list[dict]) -> list[dict]:
    return [message for message in messages if is_tool_structural(message)]


def session_headers(request: Request) -> tuple[str, str | None]:
    session = (
        request.headers.get("x-opencode-session-id")
        or request.headers.get("x-opencode-session")
        or request.headers.get("x-session-affinity")
        or request.headers.get("x-session-id")
        or "unattributed"
    )
    parent = request.headers.get("x-opencode-parent-session-id") or request.headers.get(
        "x-parent-session-id"
    )
    return session, parent


async def compress_messages(messages: list[dict]) -> tuple[list[dict], dict]:
    compression = CONFIG.get("compression", {}) or {}
    if not compression.get("enabled", True):
        return messages, {"compressed": False, "reason": "disabled"}

    preserve_system = compression.get("preserve_system", True)
    preserve_latest = compression.get("preserve_latest_user", True)
    min_chars = int(compression.get("min_input_chars", 24000))
    rate = float(compression.get("rate", 0.5))

    candidates = []
    candidate_indexes = []

    latest_user_index = None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            latest_user_index = i

    for i, m in enumerate(messages):
        role = m.get("role")
        if preserve_system and role == "system":
            continue
        if preserve_latest and i == latest_user_index:
            continue
        if role not in ("user", "assistant", "tool", "developer"):
            continue
        if is_tool_structural(m):
            continue
        text = message_text(m)
        if text:
            candidates.append(text)
            candidate_indexes.append(i)

    combined = "\n\n".join(candidates)
    if len(combined) < min_chars:
        return messages, {
            "compressed": False,
            "reason": "below_threshold",
            "input_chars": len(combined),
        }

    payload = {"text": combined, "rate": rate}
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{LINGUA_URL}/compress", json=payload)
    r.raise_for_status()
    result = r.json()
    compressed = result.get("compressed_prompt")
    if not compressed:
        return messages, {"compressed": False, "reason": "lingua_empty_response"}

    # Preserve message roles but replace candidate context with one compressed
    # user/context message. This is intentionally conservative.
    out = [dict(m) for m in messages]
    first = candidate_indexes[0]
    for idx in sorted(candidate_indexes[1:], reverse=True):
        out.pop(idx)
    out[first] = {
        "role": "user",
        "content": (
            "[Compressed prior context generated by LLMLingua-2]\n\n"
            + compressed
        ),
    }

    if structural_messages(messages) != structural_messages(out):
        return messages, {
            "compressed": False,
            "reason": "unsafe_tool_structure",
            "input_chars": len(combined),
        }

    return out, {
        "compressed": True,
        "input_chars": len(combined),
        "output_chars": len(compressed),
        "rate": rate,
        "lingua": result,
    }


def to_responses_request(body: dict) -> dict:
    messages = body.pop("messages", [])
    system = []
    input_items = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system" or role == "developer":
            system.append(extract_text(content))
        else:
            input_items.append({"role": role, "content": content})
    out = dict(body)
    out["input"] = input_items
    if system:
        out["instructions"] = "\n\n".join(system)
    out["model"] = body.get("model")
    return out


def responses_to_chat(data: dict) -> dict:
    output_text = data.get("output_text")
    if output_text is None:
        chunks = []
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text"):
                    chunks.append(c.get("text", ""))
        output_text = "".join(chunks)

    return {
        "id": data.get("id", "response"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": output_text or ""},
            "finish_reason": "stop",
        }],
        "usage": data.get("usage", {}),
    }


def to_anthropic_request(body: dict) -> dict:
    out = dict(body)
    messages = out.pop("messages", [])
    system_parts = []
    converted = []

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("system", "developer"):
            system_parts.append(extract_text(content))
            continue
        converted.append({
            "role": "assistant" if role == "assistant" else "user",
            "content": content,
        })

    out["messages"] = converted
    if system_parts:
        out["system"] = "\n\n".join(system_parts)

    # Anthropic requires max_tokens for Messages API.
    out.setdefault("max_tokens", 8192)
    return out


def anthropic_to_chat(data: dict) -> dict:
    parts = []
    for item in data.get("content", []) or []:
        if item.get("type") == "text":
            parts.append(item.get("text", ""))

    return {
        "id": data.get("id", "message"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "".join(parts)},
            "finish_reason": "stop" if data.get("stop_reason") else None,
        }],
        "usage": data.get("usage", {}),
    }


def sse_data(event: str, payload: dict) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


async def iter_sse_events(response):
    event = "message"
    data = []
    async for line in response.aiter_lines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and data:
            raw = "\n".join(data)
            if raw != "[DONE]":
                try:
                    yield event, json.loads(raw)
                except json.JSONDecodeError:
                    pass
            event = "message"
            data = []
    if data and "\n".join(data) != "[DONE]":
        try:
            yield event, json.loads("\n".join(data))
        except json.JSONDecodeError:
            pass


def chat_chunk(
    response_id: str,
    model: str,
    delta: dict,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> bytes:
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    if usage is not None:
        payload["usage"] = usage
    return sse_data("", payload)


async def translated_stream(response, protocol: str, model: str, on_usage=None):
    response_id = f"chatcmpl-{int(time.time())}"
    started = False
    tool_indexes: dict[int, int] = {}
    next_tool_index = 0

    async for event, data in iter_sse_events(response):
        if protocol == "responses":
            event_type = data.get("type", event)
            if event_type == "response.created":
                response_id = data.get("response", {}).get("id", response_id)
            elif event_type == "response.output_text.delta":
                delta = {"content": data.get("delta", "")}
                if not started:
                    delta["role"] = "assistant"
                    started = True
                yield chat_chunk(response_id, model, delta)
            elif event_type == "response.output_item.added":
                item = data.get("item", {})
                if item.get("type") == "function_call":
                    output_index = data.get("output_index", next_tool_index)
                    tool_index = next_tool_index
                    next_tool_index += 1
                    tool_indexes[output_index] = tool_index
                    delta = {
                        "tool_calls": [{
                            "index": tool_index,
                            "id": item.get("call_id"),
                            "type": "function",
                            "function": {"name": item.get("name", ""), "arguments": ""},
                        }],
                    }
                    if not started:
                        delta["role"] = "assistant"
                    yield chat_chunk(response_id, model, delta)
                    started = True
            elif event_type == "response.function_call_arguments.delta":
                output_index = data.get("output_index", 0)
                yield chat_chunk(response_id, model, {
                    "tool_calls": [{
                        "index": tool_indexes.get(output_index, 0),
                        "function": {"arguments": data.get("delta", "")},
                    }],
                })
            elif event_type == "response.completed":
                result = data.get("response", {})
                if on_usage:
                    await on_usage(result.get("usage", {}))
                yield chat_chunk(response_id, model, {}, "stop", result.get("usage"))
        else:
            event_type = data.get("type", event)
            if event_type == "message_start":
                message = data.get("message", {})
                response_id = message.get("id", response_id)
            elif event_type == "content_block_start":
                block = data.get("content_block", {})
                if block.get("type") == "tool_use":
                    tool_index = data.get("index", next_tool_index)
                    tool_indexes[data.get("index", tool_index)] = tool_index
                    next_tool_index += 1
                    delta = {
                        "tool_calls": [{
                            "index": tool_index,
                            "id": block.get("id"),
                            "type": "function",
                            "function": {"name": block.get("name", ""), "arguments": ""},
                        }],
                    }
                    if not started:
                        delta["role"] = "assistant"
                    yield chat_chunk(response_id, model, delta)
                    started = True
            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    content = {"content": delta.get("text", "")}
                    if not started:
                        content["role"] = "assistant"
                        started = True
                    yield chat_chunk(response_id, model, content)
                elif delta.get("type") == "input_json_delta":
                    yield chat_chunk(response_id, model, {
                        "tool_calls": [{
                            "index": tool_indexes.get(data.get("index", 0), 0),
                            "function": {"arguments": delta.get("partial_json", "")},
                        }],
                    })
            elif event_type == "message_delta":
                stop_reason = data.get("delta", {}).get("stop_reason")
                finish = "tool_calls" if stop_reason == "tool_use" else "stop"
                if on_usage:
                    await on_usage(data.get("usage", {}))
                yield chat_chunk(response_id, model, {}, finish, data.get("usage"))
            elif event_type == "message_stop":
                yield b"data: [DONE]\n\n"
                return

    yield b"data: [DONE]\n\n"


async def streaming_upstream_chat(
    body: dict,
    upstream_id: str,
    protocol: str,
    forwarded: dict[str, str],
    on_usage=None,
):
    upstream = upstream_config(upstream_id)
    headers = {"Content-Type": "application/json"}

    if protocol == "chat":
        path = "/chat/completions"
        upstream_body = dict(body)
        upstream_body["stream"] = True
    elif protocol == "responses":
        path = "/responses"
        upstream_body = to_responses_request(dict(body))
        upstream_body["stream"] = True
    elif protocol == "messages":
        path = "/messages"
        upstream_body = to_anthropic_request(dict(body))
        upstream_body["stream"] = True
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Model uses protocol '{protocol}' and is not supported by /v1/chat/completions",
        )

    client = httpx.AsyncClient(timeout=httpx.Timeout(600, read=None))
    try:
        request = client.build_request(
            "POST",
            f"{upstream['base_url'].rstrip('/')}{path}",
            headers=upstream_headers(upstream, forwarded, True) | headers,
            json=upstream_body,
        )
        r = await client.send(request, stream=True)
    except Exception:
        await client.aclose()
        raise

    if r.status_code >= 400:
        content = await r.aread()
        await r.aclose()
        await client.aclose()
        return Response(
            content=content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    async def iterator():
        try:
            if protocol == "chat":
                async for line in r.aiter_lines():
                    yield (line + "\n").encode()
                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        if raw != "[DONE]":
                            try:
                                data = json.loads(raw)
                                if data.get("usage") and on_usage:
                                    await on_usage(data["usage"])
                            except json.JSONDecodeError:
                                pass
            else:
                async for chunk in translated_stream(r, protocol, body["model"], on_usage):
                    yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    return StreamingResponse(iterator(), media_type="text/event-stream")


def stream_usage(event: str, payload: dict) -> dict | None:
    """Pull usage out of Anthropic (and Responses) streaming events."""
    event_type = payload.get("type", event)
    if event_type == "message_start":
        return payload.get("message", {}).get("usage")
    if event_type == "message_delta":
        return payload.get("usage")
    if event_type == "response.completed":
        return payload.get("response", {}).get("usage")
    return None


async def streaming_upstream_request(
    upstream: dict,
    path: str,
    body: dict,
    forwarded: dict[str, str],
    on_usage=None,
    on_complete=None,
):
    client = httpx.AsyncClient(timeout=httpx.Timeout(600, read=None))
    try:
        request = client.build_request(
            "POST",
            f"{upstream['base_url'].rstrip('/')}{path}",
            headers=upstream_headers(upstream, forwarded, True) | {
                "Content-Type": "application/json",
            },
            json=body,
        )
        response = await client.send(request, stream=True)
    except Exception:
        await client.aclose()
        raise

    if response.status_code >= 400:
        content = await response.aread()
        await response.aclose()
        await client.aclose()
        return Response(
            content=content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    async def iterator():
        event = "message"
        data: list[str] = []
        try:
            async for line in response.aiter_lines():
                yield (line + "\n").encode()
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    raw = "\n".join(data)
                    if raw != "[DONE]":
                        try:
                            usage = stream_usage(event, json.loads(raw))
                            if usage and on_usage:
                                await on_usage(usage)
                        except json.JSONDecodeError:
                            pass
                    event = "message"
                    data = []
            if data and "\n".join(data) != "[DONE]":
                try:
                    usage = stream_usage(event, json.loads("\n".join(data)))
                    if usage and on_usage:
                        await on_usage(usage)
                except json.JSONDecodeError:
                    pass
        finally:
            if on_complete:
                await on_complete()
            await response.aclose()
            await client.aclose()

    return StreamingResponse(iterator(), media_type="text/event-stream")


async def upstream_chat(
    body: dict,
    upstream_id: str,
    protocol: str,
    forwarded: dict[str, str] | None = None,
    on_usage=None,
):
    if body.get("stream"):
        return await streaming_upstream_chat(
            body, upstream_id, protocol, forwarded or {}, on_usage
        )

    upstream = upstream_config(upstream_id)
    headers = {"Content-Type": "application/json"}

    if protocol == "chat":
        path = "/chat/completions"
        upstream_body = dict(body)
        upstream_body["stream"] = False
        converter = lambda x: x
    elif protocol == "responses":
        path = "/responses"
        upstream_body = to_responses_request(dict(body))
        upstream_body.pop("stream", None)
        upstream_body["stream"] = False
        converter = responses_to_chat
    elif protocol == "messages":
        path = "/messages"
        upstream_body = to_anthropic_request(dict(body))
        upstream_body.pop("stream", None)
        upstream_body["stream"] = False
        converter = anthropic_to_chat
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported protocol: {protocol}")

    r = await upstream_request(
        upstream,
        "POST",
        path,
        forwarded=forwarded,
        headers=headers,
        json=upstream_body,
    )

    if r.status_code >= 400:
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    return JSONResponse(converter(r.json()))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "upstreams": list((CONFIG.get("upstreams", {}) or {}).keys()),
        "lingua_url": LINGUA_URL,
        "compression_enabled": CONFIG.get("compression", {}).get("enabled", True),
    }


async def verify_exact(session: str | None = None, text: str | None = None) -> dict:
    record = metering.latest() if not session else (metering._records_for(session)[-1:] or [None])[0]
    if record is None:
        return {"error": "No metered request found"}
    messages = record.get("original_messages") if text is None else [{"role": "user", "content": text}]
    if not messages:
        return {"error": "Original messages are not available; provide text explicitly"}
    upstream_id = record["upstream"]
    protocol = protocol_for(upstream_id, record["model"])
    session_id = record.get("session", "exact-verification")
    forwarded = {
        "x-opencode-session-id": session_id,
        "x-opencode-session": session_id,
        "x-opencode-client": "opencode",
        "user-agent": "opencode/1.18.35",
    }
    async def measure(payload_messages: list[dict]) -> int:
        result = await upstream_chat(
            {
                "model": record["model"],
                "messages": payload_messages,
                "stream": False,
                "max_tokens": 1,
            },
            upstream_id,
            protocol,
            forwarded=forwarded,
        )
        if not isinstance(result, JSONResponse):
            return 0
        data = json.loads(result.body)
        usage = data.get("usage", {})
        return int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)

    try:
        exact_tokens = await measure(messages)
        compressed_messages = record.get("compressed_messages")
        exact_compressed_tokens = await measure(compressed_messages) if compressed_messages else None
    except httpx.HTTPError as exc:
        return {
            "error": "The stored session cannot be replayed exactly; provide text explicitly.",
            "detail": str(exc),
        }
    info = await metering.model_info("opencode-go" if upstream_id == "opencode-go" else upstream_id, record["model"])
    input_price = (info.get("cost", {}) or {}).get("input", 0)
    exact_cost = metering._cost(exact_tokens, input_price)
    billed = record.get("billed_prompt_tokens", 0)
    comparable_billed = exact_compressed_tokens if exact_compressed_tokens is not None else billed
    billed_cost = metering._cost(comparable_billed, input_price)
    return {
        "session": record.get("session"),
        "model": record["model"],
        "exact_original_prompt_tokens": exact_tokens,
        "billed_prompt_tokens": billed,
        "exact_compressed_prompt_tokens": exact_compressed_tokens,
        "saved_prompt_tokens": max(0, exact_tokens - comparable_billed),
        "exact_original_prompt_cost_usd": exact_cost,
        "billed_prompt_cost_usd": billed_cost,
        "saved_prompt_cost_usd": max(0, exact_cost - billed_cost),
        "context_limit": (info.get("limit", {}) or {}).get("context"),
    }


@app.get("/v1/metering/stats")
async def metering_stats(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    require_auth(authorization, x_api_key)
    return metering.stats()


@app.get("/v1/metering/sessions")
async def metering_sessions(
    limit: int = 20,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    require_auth(authorization, x_api_key)
    return {"data": metering.sessions(limit)}


@app.get("/v1/metering/sessions/{session}")
async def metering_session(
    session: str,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    require_auth(authorization, x_api_key)
    return metering.report(session)


@app.get("/v1/models")
async def models(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    x_gateway_upstream: str | None = Header(default=None),
):
    require_auth(authorization, x_api_key)
    upstream_id, _ = select_upstream(x_gateway_upstream)
    return await get_models(upstream_id)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    x_gateway_upstream: str | None = Header(default=None),
):
    require_auth(authorization, x_api_key)
    body = await request.json()
    upstream_id, upstream = select_upstream(x_gateway_upstream)

    if not body.get("model"):
        raise HTTPException(status_code=400, detail="model is required")

    model = body["model"]

    if upstream.get("validate_models", False):
        data = await get_models(upstream_id)
        if model not in model_ids(data):
            raise HTTPException(status_code=404, detail=f"Unknown {upstream_id} model: {model}")

    protocol = protocol_for(upstream_id, model)

    if protocol == "systemone" and upstream.get("reject_systemone_from_chat", True):
        raise HTTPException(
            status_code=400,
            detail="Jev/System One is not a chat-completions model. Use Zen /v1/systemone directly.",
        )

    messages = body.get("messages", [])
    session, parent_session = session_headers(request)
    original_chars = sum(len(message_text(message)) for message in messages)
    original_messages = [dict(message) for message in messages]
    compressed_messages, stats = await compress_messages(messages)
    body["messages"] = compressed_messages
    body["model"] = model

    compressed_chars = int(stats.get("output_chars", original_chars)) if stats.get("compressed") else original_chars
    usage_captured = False
    if stats.get("compressed") and CONFIG.get("compression", {}).get("log", True):
        logger.info(
            "compression session=%s upstream=%s model=%s original_chars=%s compressed_chars=%s saved_chars=%s",
            session,
            upstream_id,
            model,
            original_chars,
            compressed_chars,
            max(0, original_chars - compressed_chars),
        )

    async def capture_usage(usage: dict):
        nonlocal usage_captured
        if usage_captured or not usage:
            return
        usage_captured = True
        await metering.record(
            session=session,
            parent_session=parent_session,
            upstream=upstream_id,
            model=model,
            original_chars=original_chars,
            compressed_chars=compressed_chars,
            compressed=bool(stats.get("compressed")),
            usage=usage,
            original_messages=original_messages,
            compressed_messages=[dict(message) for message in compressed_messages],
        )

    result = await upstream_chat(
        body,
        upstream_id,
        protocol,
        forwarded_headers(request, upstream),
        capture_usage,
    )

    if isinstance(result, JSONResponse):
        try:
            payload = json.loads(result.body)
            await capture_usage(payload.get("usage", {}))
        except (TypeError, json.JSONDecodeError):
            pass
    elif isinstance(result, Response) and result.status_code >= 400:
        await metering.record(
            session=session,
            parent_session=parent_session,
            upstream=upstream_id,
            model=model,
            original_chars=original_chars,
            compressed_chars=compressed_chars,
            compressed=bool(stats.get("compressed")),
            usage={},
            original_messages=original_messages,
            compressed_messages=[dict(message) for message in compressed_messages],
            error_status=result.status_code,
        )

    # Add lightweight headers where possible.
    if isinstance(result, JSONResponse):
        result.headers["X-Lingua-Compression"] = str(stats.get("compressed", False)).lower()
        if stats.get("compressed"):
            result.headers["X-Lingua-Input-Chars"] = str(stats.get("input_chars"))
            result.headers["X-Lingua-Output-Chars"] = str(stats.get("output_chars"))
    elif isinstance(result, StreamingResponse):
        result.headers["X-Lingua-Compression"] = str(stats.get("compressed", False)).lower()
        if stats.get("compressed"):
            result.headers["X-Lingua-Input-Chars"] = str(stats.get("input_chars"))
            result.headers["X-Lingua-Output-Chars"] = str(stats.get("output_chars"))
    return result


@app.post("/v1/responses")
async def responses(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    x_gateway_upstream: str | None = Header(default=None),
):
    require_auth(authorization, x_api_key)
    body = await request.json()
    upstream_id, upstream = select_upstream(x_gateway_upstream)
    model = body.get("model", "")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    # This endpoint is passed through for clients that explicitly use
    # OpenAI Responses. Compression of Responses input is intentionally
    # conservative; use /v1/chat/completions for automatic compression.
    body["model"] = model

    if upstream.get("validate_models", False):
        data = await get_models(upstream_id)
        if model not in model_ids(data):
            raise HTTPException(status_code=404, detail=f"Unknown {upstream_id} model: {model}")

    if body.get("stream"):
        return await streaming_upstream_request(
            upstream,
            "/responses",
            body,
            forwarded_headers(request, upstream),
        )

    r = await upstream_request(
        upstream,
        "POST",
        "/responses",
        forwarded=forwarded_headers(request, upstream),
        stream=bool(body.get("stream")),
        headers={"Content-Type": "application/json"},
        json=body,
    )

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


@app.post("/v1/messages")
async def messages_api(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    x_gateway_upstream: str | None = Header(default=None),
):
    require_auth(authorization, x_api_key)
    body = strip_thinking_block_binding(await request.json())
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    upstream_id, upstream = select_upstream(x_gateway_upstream)
    if upstream.get("validate_models", False):
        data = await get_models(upstream_id)
        if model not in model_ids(data):
            raise HTTPException(status_code=404, detail=f"Unknown {upstream_id} model: {model}")

    session, parent_session = session_headers(request)
    messages = body.get("messages", []) or []
    original_chars = sum(
        len(extract_text(m.get("content")))
        for m in messages
        if isinstance(m, dict)
    )

    forwarded = strip_thinking_binding_beta(forwarded_headers(request, upstream))
    anthropic_version = request.headers.get("anthropic-version")
    if anthropic_version:
        forwarded["anthropic-version"] = anthropic_version
    # Anthropic-style upstreams read the credential from x-api-key rather
    # than the Authorization header that upstream_headers() injects.
    api_key_env = upstream.get("api_key_env")
    upstream_api_key = os.getenv(api_key_env, "") if api_key_env else ""
    if upstream_api_key:
        forwarded["x-api-key"] = upstream_api_key

    usage_acc: dict = {}
    metered = False

    async def record_metered(usage: dict | None, error_status: int | None = None):
        nonlocal metered
        if metered:
            return
        metered = True
        await metering.record(
            session=session,
            parent_session=parent_session,
            upstream=upstream_id,
            model=model,
            original_chars=original_chars,
            compressed_chars=original_chars,
            compressed=False,
            usage=usage,
            error_status=error_status,
        )

    async def capture_usage(usage: dict):
        usage_acc.update(usage)

    async def record_stream():
        await record_metered(usage_acc or None)

    # Anthropic Messages is passed through unchanged. No compression is
    # applied here: rewriting tool_use/tool_result blocks risks losing
    # the tool structure that /v1/chat/completions deliberately protects.
    if body.get("stream"):
        return await streaming_upstream_request(
            upstream,
            "/messages",
            body,
            forwarded,
            on_usage=capture_usage,
            on_complete=record_stream,
        )

    r = await upstream_request(
        upstream,
        "POST",
        "/messages",
        forwarded=forwarded,
        headers={"Content-Type": "application/json"},
        json=body,
    )
    if r.status_code >= 400:
        await record_metered(None, r.status_code)
    else:
        try:
            await record_metered(r.json().get("usage", {}))
        except ValueError:
            await record_metered(None)

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


class MCPGatewayAuth:
    def __init__(self, downstream):
        self.downstream = downstream

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers", []))
            expected = f"Bearer {GATEWAY_API_KEY}".encode()
            authorization = headers.get(b"authorization")
            x_api_key = headers.get(b"x-api-key", b"").decode("utf-8", "replace")
            authorized = authorization == expected or (
                bool(x_api_key) and x_api_key == GATEWAY_API_KEY
            )
            if GATEWAY_API_KEY and not authorized:
                if authorization:
                    received = authorization.decode("utf-8", "replace")
                elif x_api_key:
                    received = f"(x-api-key) {x_api_key}"
                else:
                    received = "None"
                body = json.dumps({"detail": f"Invalid gateway API key {received}"}).encode()
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                })
                await send({
                    "type": "http.response.body",
                    "body": body,
                })
                return
        await self.downstream(scope, receive, send)


metering_mcp = create_mcp(metering, verify_exact)
try:
    mcp_http_app = metering_mcp.streamable_http_app()
except AttributeError:
    mcp_http_app = metering_mcp.sse_app()
app.router.lifespan_context = mcp_http_app.router.lifespan_context
app.mount("", MCPGatewayAuth(mcp_http_app))
