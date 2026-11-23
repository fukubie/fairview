# Fairview on a Hugging Face Space (Docker SDK).
#
# Built from the folder that build_space.py assembles:
#   app/              the application (run.py, backend, frontend)
#   app/FaVFA-main/   published RFW scores for the Fairness report, included
#                     only when a local (git-ignored) FaVFA-main/ exists
#
# Spaces serve the container over HTTPS (which browsers need for the camera)
# and run it as user ID 1000, so every path the app writes to is owned by it.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FAIRFACE_HOST=0.0.0.0 \
    FAIRFACE_PORT=7860 \
    FAIRFACE_MODELS_DIR=/opt/fairface/models \
    FAIRFACE_DATA_DIR=/home/user/data \
    FAIRFACE_FRAME_ANCESTORS=https://huggingface.co \
    FAIRFACE_OPEN_ACCESS=1

# libglib is the only shared library the headless OpenCV wheel needs on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 user

WORKDIR /app
COPY app/requirements.txt ./
RUN pip install -r requirements.txt

COPY app/ /app/

# Download and SHA-256-verify the ~115 MB of ONNX models at build time, so a
# cold start (Spaces sleep when idle) does not wait for the download.
RUN python -c "from backend.engine import model_manager; model_manager.ensure_models()" \
    && chmod -R a+rX /opt/fairface

USER user
EXPOSE 7860

# Room links and the allowed Host name come from the Space's own address.
# FAIRFACE_OPEN_ACCESS=1 above: no access key, for a short-lived demo. To
# require one instead, add a Space secret FAIRFACE_API_KEY.
CMD ["sh", "-c", "if [ -z \"$FAIRFACE_PUBLIC_URL\" ] && [ -n \"$SPACE_HOST\" ]; then export FAIRFACE_PUBLIC_URL=\"https://$SPACE_HOST\"; fi; exec python run.py --no-install"]
