# Agent Instructions

## Runtime

- This is a Docker Compose application with two services: `gateway` (FastAPI, port 4000) and `lingua` (LLMLingua-2 server, internal port 8001).
- Start the stack with `cp .env.example .env` followed by `docker compose up -d --build`; the first Lingua start downloads the Hugging Face model into the `hf-cache` volume.
- The six upstream API keys are required for useful requests. Keep credentials in `.env`; do not commit them.
- The gateway reads `config.yml` at startup through the container-mounted `/app/config.yml`; restart or recreate `gateway` after changing configuration.

## Code And Configuration

- `gateway/app.py` is the gateway entrypoint and owns authentication, header-based upstream selection, per-upstream model discovery/caching, compression, and protocol translation.
- `config.yml` is the source of truth for compression thresholds and the six upstream definitions. OpenCode's built-in catalogs provide model lists.
- `/v1/chat/completions` is the automatic-compression path. Zen GPT models translate to `/responses`, Zen Claude models to `/messages`, and other upstreams use `/chat/completions`.
- Chat requests preserve streaming. Chat-compatible upstreams are passed through as SSE; Zen Responses/Messages streams are translated to OpenAI-compatible SSE.
- Compression must never remove or rewrite assistant tool calls or their tool results; skip compression when tool structure cannot be preserved.
- `jev-*`/System One models are discoverable but intentionally rejected by `/v1/chat/completions`. The pass-through `/v1/responses` endpoint does not perform automatic compression.
- Metering persists per-session token/cost records in the `gateway-data` volume and exposes remote MCP tools at `/mcp`; use `lingua_session_report` when asked about current-session cost, omitting the session id to select the most recent session.
- The Dockerfiles are independent build contexts (`gateway/` and `lingua/`); dependency changes belong in the relevant context rather than at repository root.

## Verification

- There is no automated test, lint, formatter, or typecheck configuration in this repository.
- Validate Compose syntax with `docker compose config` before starting services.
- After startup, smoke-test the gateway with `curl http://localhost:4000/health` and authenticated model discovery with `curl http://localhost:4000/v1/models -H "Authorization: Bearer $GATEWAY_API_KEY"`.
- Inspect failures with `docker compose logs -f gateway` and `docker compose logs -f lingua`; test container-to-container Lingua connectivity with `docker compose exec gateway python -c "import urllib.request; print(urllib.request.urlopen('http://lingua:8001/health').read().decode())"`.
