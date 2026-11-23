# Fairview on any Linux server, behind Caddy (see docker-compose.yml).
# Build context is the repository root; Dockerfile.dockerignore keeps the
# context to the app, plus the two RFW score files from FaVFA-main/ when that
# (git-ignored) folder is present locally.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FAIRFACE_HOST=0.0.0.0 \
    FAIRFACE_PORT=8000 \
    FAIRFACE_MODELS_DIR=/opt/fairface/models \
    FAIRFACE_DATA_DIR=/var/lib/fairface/data

# libglib is the only shared library the headless OpenCV wheel needs on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

# run.py, backend/, frontend/ and, if present, FaVFA-main/ (the Fairness
# report's optional RFW section reads it from /app/FaVFA-main).
COPY . ./

# Download and SHA-256-verify the ONNX models at build time.
RUN python -c "from backend.engine import model_manager; model_manager.ensure_models()" \
    && chmod -R a+rX /opt/fairface

RUN useradd --system --uid 10001 --home-dir /var/lib/fairface fairface \
    && mkdir -p /var/lib/fairface/data \
    && chown -R fairface:fairface /var/lib/fairface
USER fairface

EXPOSE 8000
CMD ["python", "run.py", "--no-install"]
