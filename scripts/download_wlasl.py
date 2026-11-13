"""Build a WLASL clip manifest, either from an existing preprocessed video folder or by downloading.

Two sources:
  --from-dir DIR   folder of preprocessed clips named <video_id>.mp4 (e.g. the Kaggle "WLASL processed"
                   release, or WLASL's own start_kit output). No network needed.
  (default)        download the original source URLs listed in WLASL_v0.3.json. Many links are dead,
                   so expect 50-70% availability. YouTube sources need `pip install yt-dlp`.

Output: data/manifest.jsonl with one {video_id, gloss, split, path, frame_start, frame_end} per clip.

Examples:
  python scripts/download_wlasl.py --subset 100 --from-dir ~/datasets/wlasl/videos
  python scripts/download_wlasl.py --subset 100 --workers 16
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
UA = {"User-Agent": "Mozilla/5.0 (EchoGesture dataset fetcher)"}


def load_index(data_dir: Path) -> list[dict]:
    path = data_dir / "WLASL_v0.3.json"
    if not path.exists():
        print(f"fetching {WLASL_JSON_URL}")
        with urllib.request.urlopen(urllib.request.Request(WLASL_JSON_URL, headers=UA), timeout=60) as r:
            path.write_bytes(r.read())
    return json.loads(path.read_text())


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def fetch(inst: dict, raw_dir: Path) -> Path | None:
    url = inst["url"]
    if is_youtube(url):
        yt_id = url.split("v=")[-1].split("&")[0].split("/")[-1]
        out = raw_dir / f"yt_{yt_id}.mp4"
        if out.exists():
            return out
        if not shutil.which("yt-dlp"):
            return None
        cmd = ["yt-dlp", "-q", "--no-warnings", "--socket-timeout", "30", "-f", "mp4/best", "-o", str(out), url]
        ok = subprocess.run(cmd, capture_output=True, timeout=300).returncode == 0
        return out if ok and out.exists() else None
    ext = Path(url.split("?")[0]).suffix.lower() or ".mp4"
    out = raw_dir / f"{inst['video_id']}{ext}"
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            data = r.read()
        if len(data) < 1024:
            return None
        out.write_bytes(data)
        if ext not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):  # e.g. .swf -> convert
            mp4 = out.with_suffix(".mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), str(mp4)], timeout=300)
            return mp4 if mp4.exists() else None
        return out
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=100, help="first N glosses (WLASL100/300/1000/2000)")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--from-dir", type=Path, help="folder of preprocessed <video_id>.mp4 clips")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(args.data_dir)[: args.subset]
    instances = [dict(inst, gloss=e["gloss"]) for e in index for inst in e["instances"]]
    print(f"{len(index)} glosses, {len(instances)} instances")

    rows = []
    if args.from_dir:
        src = args.from_dir.expanduser()
        for inst in instances:
            p = src / f"{inst['video_id']}.mp4"
            if p.exists():
                rows.append(dict(path=str(p), frame_start=1, frame_end=-1, inst=inst))
    else:
        raw_dir = args.data_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        if not shutil.which("yt-dlp"):
            print("note: yt-dlp not found, YouTube-hosted clips will be skipped (pip install yt-dlp)")
        with ThreadPoolExecutor(args.workers) as pool:
            futs = {pool.submit(fetch, inst, raw_dir): inst for inst in instances}
            for f in tqdm(as_completed(futs), total=len(futs), desc="download"):
                inst, p = futs[f], f.result()
                if p is not None:
                    rows.append(dict(path=str(p), frame_start=inst["frame_start"], frame_end=inst["frame_end"], inst=inst))

    out = args.data_dir / "manifest.jsonl"
    with out.open("w") as fh:
        for r in rows:
            i = r["inst"]
            fh.write(json.dumps(dict(video_id=i["video_id"], gloss=i["gloss"].upper(), split=i["split"],
                                     path=r["path"], frame_start=r["frame_start"], frame_end=r["frame_end"],
                                     signer_id=i.get("signer_id"))) + "\n")
    print(f"wrote {len(rows)}/{len(instances)} clips -> {out}")


if __name__ == "__main__":
    main()
