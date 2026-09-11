FROM python:3.11-slim

WORKDIR /app

# The graph schema is NOT baked into the image: SCHEMA_FILE points at
# /app/schema/graph_schema.yaml, a read-only bind mount provided by compose.
ENV SCHEMA_FILE=/app/schema/graph_schema.yaml \
    TRACE_DIR=/app/eval/traces

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Supply-chain pinning (OWASP LLM Supply Chain / A05): pre-bake the FastEmbed
# model into the image at build time and run with HF_HUB_OFFLINE=1, so the
# runtime never pulls from the Hugging Face hub on first query. Changing
# EMBEDDING_MODEL therefore requires a rebuild:
#   docker compose build --build-arg EMBEDDING_MODEL=<new-model> agent
# (or set HF_HUB_OFFLINE=0 in the environment to allow an unpinned fetch).
ARG EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large-v1
# Download during build (online), then lock the runtime offline.
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN HF_HUB_OFFLINE=0 python -c "import os; from fastembed import TextEmbedding; next(TextEmbedding(model_name=os.environ['EMBEDDING_MODEL']).embed(['warmup']))"
ENV HF_HUB_OFFLINE=1

COPY app ./app

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]