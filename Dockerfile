# Render-friendly image for ocrloop.
#
# We bake Tesseract (with rus + eng language packs) into the image because
# Render's native Python runtime does not allow apt-get installs at deploy
# time. EasyOCR is intentionally NOT installed by default — it pulls
# PyTorch (~3 GB) which won't fit on Render's free tier. To enable EasyOCR
# instead, build with `--build-arg INSTALL_EASYOCR=1` and set
# `OCR_ENGINE=easyocr` as an environment variable on the service.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps:
#   - tesseract-ocr           recognition engine
#   - tesseract-ocr-rus/-eng  language packs
#   - libgl1 / libglib2.0-0   needed by opencv-python-headless at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so changes to source don't bust the layer cache.
COPY requirements.txt ./
RUN pip install -r requirements.txt

ARG INSTALL_EASYOCR=0
RUN if [ "$INSTALL_EASYOCR" = "1" ]; then pip install easyocr ; fi

COPY . .

# Render passes env vars (BOT_TOKEN, OCR_ENGINE, OCR_LAYOUT, …) via the
# service settings — no .env file inside the image.
CMD ["python", "-m", "ocrloop.bot"]
