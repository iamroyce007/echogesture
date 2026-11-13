"""Write predictions.jsonl (one JSON object per clip) for the organisers' evaluator.

Inputs (pick one):
  --clips-dir DIR        every video in DIR; clip_id = file stem
  --manifest FILE        text/CSV file with one clip path per line (or a "path" / "clip_id,path" CSV)
  --split test           self-evaluation on our cached WLASL test split (fast, prints accuracy)

  python scripts/make_predictions.py --clips-dir /path/to/test_clips --out predictions.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echogesture.runtime import Predictor  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def clips_from_manifest(path: Path) -> list[tuple[str, Path]]:
    rows = []
    with path.open() as fh:
        sample = fh.read(2048)
        fh.seek(0)
        if "," in sample:
            for r in csv.DictReader(fh):
                p = Path(r.get("path") or r.get("video_path") or r.get("file"))
                if not p.is_absolute():
                    p = path.parent / p
                rows.append((r.get("clip_id") or p.stem, p))
        else:
            for line in fh:
                if line.strip():
                    p = Path(line.strip())
                    p = p if p.is_absolute() else path.parent / p
                    rows.append((p.stem, p))
    return rows


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--clips-dir", type=Path)
    g.add_argument("--manifest", type=Path)
    g.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--vocab", type=Path, default=os.environ.get("ECHOGESTURE_VOCAB"),
                    help="official vocabulary list (one word per line) to restrict predictions")
    ap.add_argument("--out", type=Path, default=ROOT / "predictions.jsonl")
    args = ap.parse_args()

    allowed = Path(args.vocab).read_text().split() if args.vocab else None
    pred = Predictor(allowed_vocab=allowed)
    n = correct = 0
    with args.out.open("w") as fh:
        if args.split:
            splits = json.loads((args.data_dir / "splits.json").read_text())
            items = [i for i in splits["items"] if i["split"] == args.split]
            for it in tqdm(items, desc=args.split):
                raw = np.load(args.data_dir / it["npy"]).astype(np.float32)
                words, probs, model_ms, _ = pred.classify_landmarks(raw)
                fh.write(json.dumps(dict(clip_id=it["id"], prediction=words[0], top5=words,
                                         confidence=round(probs[0], 4), latency_ms=round(model_ms, 2))) + "\n")
                correct += words[0] == splits["vocab"][it["label"]].upper()
                n += 1
            print(f"top-1 accuracy on {args.split}: {correct / max(n, 1):.4f} ({correct}/{n})")
        else:
            clips = (sorted((p.stem, p) for p in args.clips_dir.iterdir() if p.suffix.lower() in VIDEO_EXT)
                     if args.clips_dir else clips_from_manifest(args.manifest))
            for clip_id, path in tqdm(clips, desc="clips"):
                try:
                    out = pred.predict_video(path)
                except Exception as e:  # keep every clip id present, as the evaluator requires
                    tqdm.write(f"failed {path}: {e}")
                    out = dict(prediction=pred.vocab[0], top5=pred.vocab[:5], confidence=0.0, latency_ms=0.0)
                out["clip_id"] = clip_id
                fh.write(json.dumps({k: out[k] for k in ("clip_id", "prediction", "top5", "confidence", "latency_ms")}) + "\n")
                n += 1
    print(f"wrote {n} predictions -> {args.out}")


if __name__ == "__main__":
    main()
