# syntax=docker/dockerfile:1

# ---- Base image ----
FROM python:3.12-slim

# ---- Runtime env ----
# PYTHONUNBUFFERED=1        -> logs flushed immediately (otherwise Azure log stream lags)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- Working directory ----
# Every COPY/RUN/CMD below is relative to /app inside the image.
WORKDIR /app

# ---- Install deps FIRST (copy only requirements.txt) ----

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- Copy the rest of the code ----
# .dockerignore prevents .venv, .env, .git, __pycache__, etc. from being copied in.
COPY . .

# ---- Non-root user ----
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# ---- Port ----
EXPOSE 8000

# ---- Start command ----
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
