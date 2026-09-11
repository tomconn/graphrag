# Ingest container — one-off GraphRAG ingestion job.
# data/ and schema/ are bind-mounted at /app/data and /app/schema by
# docker-compose.yml (read-only) — they are NOT copied into the image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY pipeline /app/pipeline

CMD ["python", "-m", "pipeline.main"]