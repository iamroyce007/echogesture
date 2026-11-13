"""On-device inference runtime: video/landmarks -> gloss, using the INT8 ONNX model. No network access."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import build_features
from .landmarks import LandmarkExtractor, extract_video

MODELS = Path(os.environ.get("ECHOGESTURE_MODELS", Path(__file__).resolve().parent.parent / "models"))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class Predictor:
    def __init__(self, model_path: Path | str = MODELS / "signnet_int8.onnx",
                 vocab_path: Path | str = MODELS / "vocab.json",
                 allowed_vocab: list[str] | None = None, threads: int = 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads  # single core: realistic for a wearable / phone little-core budget
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.vocab = [v.upper() for v in json.loads(Path(vocab_path).read_text())]
        # optionally restrict outputs to the organisers' fixed vocabulary list
        self.mask = None
        if allowed_vocab:
            allowed = {v.strip().upper() for v in allowed_vocab}
            self.mask = np.array([v in allowed for v in self.vocab])
        self._extractor: LandmarkExtractor | None = None

    @property
    def extractor(self) -> LandmarkExtractor:
        if self._extractor is None:
            self._extractor = LandmarkExtractor()
        return self._extractor

    def classify_features(self, feats: np.ndarray, k: int = 5):
        """feats (T, F) -> (top-k glosses, top-k probs, model latency ms)."""
        t0 = time.perf_counter()
        logits = self.session.run(None, {"features": feats[None].astype(np.float32)})[0][0]
        ms = (time.perf_counter() - t0) * 1000
        if self.mask is not None:
            logits = np.where(self.mask, logits, -1e9)
        p = _softmax(logits)
        top = np.argsort(-p)[:k]
        return [self.vocab[i] for i in top], [float(p[i]) for i in top], ms

    def classify_landmarks(self, raw: np.ndarray, k: int = 5):
        t0 = time.perf_counter()
        feats = build_features(raw)
        words, probs, model_ms = self.classify_features(feats, k)
        return words, probs, model_ms, (time.perf_counter() - t0) * 1000

    def predict_video(self, video_path: str | Path, frame_transform=None) -> dict:
        raw = extract_video(video_path, self.extractor, frame_transform=frame_transform)
        words, probs, model_ms, _ = self.classify_landmarks(raw)
        return {
            "clip_id": Path(video_path).stem,
            "prediction": words[0],
            "top5": words,
            "confidence": round(probs[0], 4),
            # latency of the recognition model on one clip (CPU, 1 thread, INT8). Per-frame keypoint
            # tracking runs in a streaming fashion as frames arrive and is reported separately in results.csv.
            "latency_ms": round(model_ms, 2),
        }
