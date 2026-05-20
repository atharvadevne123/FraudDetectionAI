"""Gunicorn WSGI entry point — loads models once at worker startup.

Run with:
    gunicorn api.wsgi:app --workers 4 --bind 0.0.0.0:8001 --timeout 120
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import _load_models, app
from utils.logging import configure_logging

configure_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE"),
)

_load_models()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    app.run(host="0.0.0.0", port=port, debug=False)
