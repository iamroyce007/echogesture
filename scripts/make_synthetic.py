"""SMOKE TEST ONLY: generate fake landmark sequences so the full pipeline (train -> export -> predict ->
benchmark) can be exercised without the real dataset. Accuracy on this data means nothing.

  python scripts/make_synthetic.py --data-dir data_synth --classes 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echogesture.landmarks import LH, N_POINTS, POSE, RH  # noqa: E402

# canonical open right hand, wrist at origin (x right, y down), unit ~ palm length
_FINGERS = [(-0.35, -0.1), (-0.2, -1.0), (0.0, -1.05), (0.2, -1.0), (0.35, -0.85)]


def hand_shape(curl: np.ndarray) -> np.ndarray:
    pts = [np.zeros(3)]
    for f, (dx, dy) in enumerate(_FINGERS):
        base = np.array([dx * 0.6, dy * 0.6 if f else -0.2, 0.0])
        for j in range(1, 5):
            c = curl[f] * j / 4
            pts.append(base + np.array([dx * 0.15 * j, dy * 0.3 * j * (1 - c), 0.15 * j * c]))
    return np.array(pts[:21], dtype=np.float32)


def sample(cls_params, rng, T):
    path_r, path_l, curl_r, curl_l, two_hands = cls_params
    t = np.linspace(0, 1, T)[:, None]
    seq = np.full((T, N_POINTS, 3), np.nan, np.float32)
    scale = rng.uniform(0.18, 0.28)
    centre = np.array([rng.uniform(0.6, 1.1), rng.uniform(0.45, 0.6)])
    pose = np.zeros((13, 3), np.float32)
    pose[0, :2] = centre + [0, -1.1 * scale]
    pose[1:5, :2] = centre + np.array([[-0.15, -1.2], [0.15, -1.2], [-0.3, -1.1], [0.3, -1.1]]) * scale
    pose[5, :2], pose[6, :2] = centre + [0.7 * scale, 0], centre + [-0.7 * scale, 0]
    pose[11, :2], pose[12, :2] = centre + [0.5 * scale, 1.6 * scale], centre + [-0.5 * scale, 1.6 * scale]
    seq[:, POSE] = pose
    for sl, path, curl, side, on in ((RH, path_r, curl_r, -1, True), (LH, path_l, curl_l, 1, two_hands)):
        if not on:
            continue
        p0, p1, p2 = path
        wrist = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2  # quadratic bezier
        wrist = centre + (wrist + rng.normal(0, 0.05, 2)) * scale
        shape = hand_shape(np.clip(curl + rng.normal(0, 0.08, 5), 0, 1)) * scale * 0.5
        shape[:, 0] *= -side
        seq[:, sl] = shape[None] + np.concatenate([wrist, np.zeros((T, 1))], 1)[:, None]
        seq[:, POSE.start + (10 if side < 0 else 9), :2] = wrist
    return seq + rng.normal(0, 0.002, seq.shape).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data_synth")
    ap.add_argument("--classes", type=int, default=20)
    ap.add_argument("--per-class", type=int, default=40)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    (args.data_dir / "landmarks").mkdir(parents=True, exist_ok=True)
    vocab = [f"SIGN{i:03d}" for i in range(args.classes)]
    items = []
    for c in range(args.classes):
        params = (rng.uniform(-1.2, 1.2, (3, 2)), rng.uniform(-1.2, 1.2, (3, 2)), rng.uniform(0, 1, 5),
                  rng.uniform(0, 1, 5), rng.random() < 0.4)
        for k in range(args.per_class):
            seq = sample(params, rng, int(rng.integers(20, 50)))
            vid = f"syn_{c:03d}_{k:03d}"
            np.save(args.data_dir / "landmarks" / f"{vid}.npy", seq.astype(np.float16))
            split = "train" if k < args.per_class * 0.7 else ("val" if k < args.per_class * 0.85 else "test")
            items.append(dict(id=vid, label=c, split=split, npy=f"landmarks/{vid}.npy"))
    (args.data_dir / "splits.json").write_text(json.dumps(dict(vocab=vocab, items=items)))
    print(f"{len(items)} synthetic sequences, {args.classes} classes -> {args.data_dir}")


if __name__ == "__main__":
    main()
