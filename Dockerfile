# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FAIRFACE_HOST=0.0.0.0 \
    FAIRFACE_PORT=8000 \
    FAIRFACE_DATA_DIR=/var/lib/fairface/data \
    FAIRFACE_MODELS_DIR=/var/lib/fairface/models

# libglib is the only shared library the headless OpenCV wheel needs on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY run.py ./
COPY backend ./backend
COPY frontend ./frontend

# Unprivileged runtime user; state lives in two volumes it owns.
RUN useradd --system --uid 10001 --home-dir /var/lib/fairface fairface \
    && mkdir -p /var/lib/fairface/data /var/lib/fairface/models \
    && chown -R fairface:fairface /var/lib/fairface
USER fairface
VOLUME ["/var/lib/fairface/data", "/var/lib/fairface/models"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import json,sys,urllib.request as u; sys.exit(0 if json.load(u.urlopen('http://127.0.0.1:8000/api/health', timeout=4))['engine_ready'] else 1)"

# Weights are fetched (and SHA-256 verified) into the models volume on first start.
CMD ["python", "run.py", "--no-install"]
