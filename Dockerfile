FROM python:3.12-slim

# Build-time args (override in docker-compose or CLI)
ARG OLLAMA_MODEL=gemma3n:e2b
ARG OLLAMA_EMBED_MODEL=nomic-embed-text

# Environment
ENV OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_MODEL=${OLLAMA_MODEL} \
    OLLAMA_EMBED_MODEL=${OLLAMA_EMBED_MODEL} \
    OLLAMA_NO_CLOUD=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    DO_NOT_TRACK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s CMD \
    python -c "import requests; requests.get('http://localhost:8501', timeout=3)"

CMD ["streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--browser.gatherUsageStats", "false"]
