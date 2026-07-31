"""

Production WSGI entry point for gunicorn (used by Render / Hugging Face /
any platform that runs `gunicorn wsgi:app` instead of `python webapp/app.py`).

webapp/app.py's model-loading logic lives inside main(), which only runs
under `if __name__ == "__main__":` -- gunicorn imports the module directly
and never calls main(), so app.py's CLASSIFIER global would stay None under
gunicorn. This file loads the model at import time instead, then exposes
the same `app` Flask instance for gunicorn to serve. app.py itself is
untouched and still works exactly as before for local development
(`python webapp/app.py`).

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import webapp.app as webapp_app  # noqa: E402
from deepfer import config  # noqa: E402
from realtime_webcam import EmotionClassifier  # noqa: E402

_CHECKPOINT = config.SAVED_MODELS_DIR / "transfer_finetune_best.keras"

if not _CHECKPOINT.exists():
    
    raise SystemExit(f"No trained checkpoint found at {_CHECKPOINT}")

print(f"[wsgi] Loading transfer model from {_CHECKPOINT}")

webapp_app.CLASSIFIER = EmotionClassifier(str(_CHECKPOINT), "transfer")

print("[wsgi] Model loaded.")

app = webapp_app.app