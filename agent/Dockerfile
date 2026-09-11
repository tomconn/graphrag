FROM python:3.11-slim

WORKDIR /app

# The graph schema is NOT baked into the image: SCHEMA_FILE points at
# /app/schema/graph_schema.yaml, a read-only bind mount provided by compose.
ENV SCHEMA_FILE=/app/schema/graph_schema.yaml \
    TRACE_DIR=/app/eval/traces

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]