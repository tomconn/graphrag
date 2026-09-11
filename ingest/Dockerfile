# Ingest container — one-off GraphRAG ingestion job.
# data/ and schema/ are bind-mounted at /app/data and /app/schema by
# docker-compose.yml (read-only) — they are NOT copied into the image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Supply-chain pinning: pre-bake the FastEmbed model at build time and run
# offline (HF_HUB_OFFLINE=1) — must match the agent image's baked model or
# chunks and queries land in different vector spaces.
ARG EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large-v1
# Download during build (online), then lock the runtime offline.
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN HF_HUB_OFFLINE=0 python -c "import os; from fastembed import TextEmbedding; next(TextEmbedding(model_name=os.environ['EMBEDDING_MODEL']).embed(['warmup']))"
ENV HF_HUB_OFFLINE=1

COPY pipeline /app/pipeline

CMD ["python", "-m", "pipeline.main"]