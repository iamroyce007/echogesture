"""Run MediaPipe over every clip in data/manifest.jsonl and cache landmarks as .npy.

Writes data/landmarks/<video_id>.npy (T, 55, 3) and data/splits.json (vocab + split assignment).
Safe to re-run: existing .npy files are skipped.

  python scripts/extract_landmarks.py --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_EXTRACTOR = None


def _init():
    global _EXTRACTOR
    from echogesture.landmarks import LandmarkExtractor

    _EXTRACTOR = LandmarkExtractor()


def _work(job):
    from echogesture.landmarks import extract_video

    row, out = job
    try:
        seq = extract_video(row["path"], _EXTRACTOR, row["frame_start"], row["frame_end"])
        np.save(out, seq.astype(np.float16))
        return row["video_id"], True
    except Exception as e:  # corrupt / unreadable clip
        return row["video_id"], f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (args.data_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    lm_dir = args.data_dir / "landmarks"
    lm_dir.mkdir(exist_ok=True)
    jobs = [(r, lm_dir / f"{r['video_id']}.npy") for r in rows]
    todo = [j for j in jobs if not j[1].exists()]
    print(f"{len(rows)} clips, {len(todo)} to extract")

    failed = 0
    if todo:
        with Pool(args.workers, initializer=_init) as pool:
            for vid, ok in tqdm(pool.imap_unordered(_work, todo), total=len(todo), desc="landmarks"):
                if ok is not True:
                    failed += 1
                    tqdm.write(f"skip {vid}: {ok}")

    # vocab in manifest order (WLASL order = frequency order), uppercase glosses
    vocab = list(dict.fromkeys(r["gloss"] for r in rows))
    items = []
    for r, out in jobs:
        if out.exists():
            items.append(dict(id=r["video_id"], label=vocab.index(r["gloss"]), split=r["split"],
                              npy=str(out.relative_to(args.data_dir))))
    splits = dict(vocab=vocab, items=items)
    (args.data_dir / "splits.json").write_text(json.dumps(splits, indent=1))
    counts = {s: sum(i["split"] == s for i in items) for s in ("train", "val", "test")}
    print(f"vocab={len(vocab)} {counts} failed={failed} -> {args.data_dir / 'splits.json'}")


if __name__ == "__main__":
    main()
