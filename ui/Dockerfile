# GraphRAG UI container: FastAPI backend (uvicorn :8000) + Vite dev server (:3000)
FROM node:20-bookworm-slim

# bookworm ships python3.11; python3-venv avoids the externally-managed-pip guard
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY frontend /app/frontend

# no package-lock.json in the repo — install resolves from package.json
RUN cd /app/frontend && npm install --no-audit --no-fund

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app
EXPOSE 3000 8000
ENTRYPOINT ["/app/entrypoint.sh"]