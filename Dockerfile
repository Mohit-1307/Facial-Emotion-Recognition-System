FROM python:3.12-slim

# System libs required by opencv-python-headless / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Put your trained checkpoint at saved_models/transfer_finetune_best.keras
# (or scratch_best.keras) before building, or mount it as a volume.
ENV MODEL_KIND=transfer
ENV CHECKPOINT_PATH=saved_models/transfer_finetune_best.keras
ENV PORT=7860

EXPOSE 7860

# Hugging Face Spaces expects the app to listen on $PORT (default 7860).
# Render/Railway also inject $PORT - this respects either.
CMD gunicorn --bind 0.0.0.0:${PORT} --timeout 120 --workers 1 \
    "webapp.app:build_app()"