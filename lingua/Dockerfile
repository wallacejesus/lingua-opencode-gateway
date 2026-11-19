FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# The adaptive-token-compressor package is maintained in the upstream
# repository but is not published on PyPI. The official deployment uses the
# standalone server file, so install only its runtime dependencies here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" \
    && pip install --no-cache-dir \
        "llmlingua>=0.2" \
        "transformers>=5.5.0" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "huggingface-hub>=0.20"

# Pin the server source to the upstream commit used during this fix instead
# of cloning the entire monorepo during every image build.
RUN curl -fsSL \
    https://raw.githubusercontent.com/open-edge-platform/edge-ai-libraries/680f93635f81fb6d1b1ba9b6306fc1e2cfcb290a/libraries/adaptive-token-compressor/src/adaptive_token_compressor/model_servers/lingua/app.py \
    -o /app/lingua_server.py

EXPOSE 8001

CMD python lingua_server.py --backend pytorch --device "${LINGUA_DEVICE:-cpu}" --port 8001 --host 0.0.0.0
