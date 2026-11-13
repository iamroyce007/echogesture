"""Evaluator entry point for Track 2.

    from predict import predict
    predict("clips/test_0412.mp4")
    # {"clip_id": "test_0412", "prediction": "HAPPY", "top5": [...], "confidence": 0.81, "latency_ms": 0.9}

Runs entirely on-device (MediaPipe keypoints + INT8 ONNX classifier on CPU); nothing is uploaded.
Set ECHOGESTURE_VOCAB=/path/to/vocab.txt (one word per line) to restrict outputs to the official vocabulary.

CLI:  python predict.py path/to/clip.mp4 [more clips...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from echogesture.runtime import Predictor  # noqa: E402

_PREDICTOR: Predictor | None = None


def _get() -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        vocab_file = os.environ.get("ECHOGESTURE_VOCAB")
        allowed = Path(vocab_file).read_text().split() if vocab_file else None
        _PREDICTOR = Predictor(allowed_vocab=allowed)
    return _PREDICTOR


def predict(video_path: str) -> dict:
    """Return the predicted gloss/word for one video clip. Must run fully on-device."""
    return _get().predict_video(video_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        print(json.dumps(predict(p)))
