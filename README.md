# Lingua + OpenCode Zen Gateway

This project provides an OpenAI-compatible local gateway in front of six upstream providers and a standalone LLMLingua-2 server.

Architecture:

    OpenCode
       |
       v
    localhost:4000
       |
       +--> route by X-Gateway-Upstream
       |
       +--> compress large chat context with Lingua
       |
       v
    Provider upstreams
       |
       +--> /chat/completions
       +--> /messages
       +--> /responses

## Important limitations

This is a reference gateway, not an official OpenCode or Lingua product.

- `/v1/models` is selected by the `X-Gateway-Upstream` header.
- OpenCode's built-in catalogs provide model lists, so new catalog models need no config update.
- Compression skips tool-call and tool-result messages; if a tool structure cannot be preserved safely, that request is sent uncompressed.
- `/v1/chat/completions` automatically compresses sufficiently large message history.
- `/v1/messages` passes Anthropic Messages requests through to the selected upstream (including streaming) and accepts `x-api-key` as well as `Authorization: Bearer`. No compression is applied on this path. OpenCode's `thinking.block_binding` control (and the matching `anthropic-beta` token) is stripped because Anthropic-compatible upstreams such as OpenCode Go reject it; messages, tools, and reasoning content are untouched.
- GPT and Claude models on the Zen upstream are translated to Zen's Responses and Messages APIs.
- Other upstreams use their OpenAI-compatible Chat Completions APIs.
- Jev/System One is intentionally not exposed through `/v1/chat/completions`.
- OpenAI-compatible upstream streams are passed through. Zen Responses/Messages streams are translated to OpenAI-compatible SSE, including text and basic tool-call deltas.

OpenCode documents the Zen endpoint families and the `/v1/models` catalog:
https://opencode.ai/docs/zen/

The official Lingua deployment exposes:
http://localhost:8001/compress

## 1. Configure

Copy:

    cp .env.example .env

Then edit `.env`:

    OPENCODE_API_KEY=...
    OPENCODE_GO_API_KEY=...
    OPENROUTER_API_KEY=...
    DEEPSEEK_API_KEY=...
    GOOGLE_API_KEY=...
    ORCAROUTER_API_KEY=...
    GATEWAY_API_KEY=...

## 2. Start

    docker compose up -d --build

The first start can take a while because the Lingua container downloads the Hugging Face compressor model.

Check:

    curl http://localhost:4000/health

Check the default upstream's models:

    curl http://localhost:4000/v1/models \
      -H "Authorization: Bearer sk-local-lingua"

## 3. Use with OpenCode

The repository includes an `opencode.jsonc` file. OpenCode automatically loads
this file when you run OpenCode from this repository:

    opencode

If you want to use the gateway from another project, copy `opencode.jsonc` into
that project's root directory (or into `.opencode/opencode.jsonc`). The file
configures OpenCode to send requests to:

    http://localhost:4000/v1

The local gateway key in `opencode.jsonc` must match `GATEWAY_API_KEY` in
`.env`. The default value for both is:

    sk-local-lingua

Do not put upstream provider keys in `opencode.jsonc`. They stay in `.env` and
are used by the gateway:

| OpenCode provider | Gateway upstream | Environment variable |
| --- | --- | --- |
| `opencode` | `opencode-zen` | `OPENCODE_API_KEY` |
| `opencode-go` | `opencode-go` | `OPENCODE_GO_API_KEY` |
| `openrouter` | `openrouter` | `OPENROUTER_API_KEY` |
| `deepseek` | `deepseek` | `DEEPSEEK_API_KEY` |
| `google` | `google` | `GOOGLE_API_KEY` |
| `orcarouter` | `orcarouter` | `ORCAROUTER_API_KEY` |

After the gateway is running:

1. Start OpenCode from the project directory:

       opencode

2. Open the model picker with `/models`.
3. Select a model from one of the enabled providers.
4. Use OpenCode normally. Large chat histories are compressed automatically
   before being sent upstream.

Each provider sends the `X-Gateway-Upstream` header so the gateway knows which
upstream should receive the request. The gateway then adds the corresponding
provider API key from `.env`.

### Important upstream-name correction

The gateway uses the exact upstream IDs from `config.yml`. The Zen upstream is
named `opencode-zen`, not `zen`. Therefore, the Zen provider in
`opencode.jsonc` must use:

    "X-Gateway-Upstream": "opencode-zen"

If your local file currently says `"zen"`, change it to
`"opencode-zen"` before using the Zen provider. The other upstream IDs are
`opencode-go`, `openrouter`, `deepseek`, `google`, and `orcarouter`.

### Metering and MCP

The same `opencode.jsonc` also connects OpenCode to the gateway's remote MCP
server:

    http://localhost:4000/mcp

The gateway persists per-session token and cost records in the `gateway-data`
volume. OpenCode exposes these MCP tools:

- `lingua_session_report`: current/most recent session tokens and cost.
- `lingua_context_report`: latest request's original estimate versus billed usage.
- `lingua_compression_stats`: cumulative compression totals.
- `lingua_list_sessions`: recent session summaries.
- `lingua_verify_exact`: explicitly re-measures a supplied prompt; this can incur a full upstream request cost.

Example prompt:

    What is the cost of this current session? Use the lingua session_report tool.

The normal reports use character-based original-token estimates and exact upstream billed usage. Exact replay of a session containing tool-call messages may not be possible; use `verify_exact` with explicit text in that case.

## 4. Compression

The gateway only compresses when the candidate context is larger than:

    compression:
      min_input_chars: 24000

It preserves system messages and the latest user message by default.

The compression request sent to Lingua is:

    POST http://lingua:8001/compress

with:

    {
      "text": "...",
      "rate": 0.5
    }

LLMLingua-2 defines the compression rate as compressed size / raw size. See:
https://github.com/microsoft/LLMLingua

## 5. CPU / GPU

The included compose defaults to CPU for portability.

For the official Open Edge Platform deployment, PyTorch supports CPU/CUDA/XPU and the OpenVINO backend supports CPU/GPU. Their current deployment documentation is:
https://github.com/open-edge-platform/edge-ai-libraries/blob/main/libraries/adaptive-token-compressor/docs/user-guide/lingua-deployment.md

For a local CPU test:

    LINGUA_DEVICE=cpu docker compose up -d --build

For Intel GPU/XPU, use the official Lingua deployment recipe because it includes the required device mappings and backend-specific Dockerfile.

## 6. Troubleshooting

See logs:

    docker compose logs -f lingua
    docker compose logs -f gateway

Test Lingua directly:

    curl http://localhost:4000/health

Then:

    docker compose exec gateway python -c \
      "import urllib.request; print(urllib.request.urlopen('http://lingua:8001/health').read().decode())"

If a model is new and uses an endpoint family that is not covered by the routing rules, add a regex rule in `config.yml`.
