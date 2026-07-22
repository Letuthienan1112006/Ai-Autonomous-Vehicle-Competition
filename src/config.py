"""Runtime configuration for the competition controller.

Paths and simulator settings can be overridden with environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = Path(os.getenv("AGV_WEIGHTS_DIR", PROJECT_ROOT / "weights"))

SEGMENTATION_WEIGHTS = Path(
    os.getenv("SEGMENTATION_WEIGHTS", WEIGHTS_DIR / "segmentation.tar")
)
TRAFFIC_SIGN_WEIGHTS = Path(
    os.getenv("TRAFFIC_SIGN_WEIGHTS", WEIGHTS_DIR / "detectsign.pt")
)

# Passing the standard Ultralytics model name allows automatic download when
# the model is not already cached. Override this value for an offline setup.
OBJECT_DETECTION_WEIGHTS = os.getenv("OBJECT_DETECTION_WEIGHTS", "yolov8n.pt")

SIMULATOR_HOST = os.getenv("SIMULATOR_HOST", "127.0.0.1")
SIMULATOR_PORT = int(os.getenv("SIMULATOR_PORT", "54321"))

def validate_custom_weights() -> None:
    """Raise a readable error when project-specific weights are missing."""
    missing = [
        path for path in (SEGMENTATION_WEIGHTS, TRAFFIC_SIGN_WEIGHTS)
        if not path.is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing project-specific model weights:\n"
            f"{formatted}\n\n"
            "Place the files in the weights/ directory or set the corresponding "
            "environment variables. See weights/README.md."
        )
