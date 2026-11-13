"""Measure every row of the Track 2 results sheet and write results.csv (+ benchmark_report.json).

  python scripts/benchmark.py                       # quick run (thermal test shortened to 60 s)
  python scripts/benchmark.py --thermal-seconds 600 # full 10-minute sustained-load test
  python scripts/benchmark.py --power-mw 410 --device "Galaxy S25" # merge a power reading taken on-device

Definitions used:
  top1_accuracy             INT8 model on the held-out WLASL test split (signers unseen in training)
  latency_p95               per-frame streaming latency = keypoint extraction (hands+pose) + classifier, p95
  model_size                total on-device model footprint (2 MediaPipe landmarkers + INT8 classifier)
  accuracy_drop_robustness  top-1 drop when test videos are degraded (low light, occlusion, blur, tilt);
                            falls back to landmark-level simulated occlusion/noise if videos are unavailable
  avg_power                 not measurable on the dev laptop without root; pass --power-mw from a device reading
  thermal_drift_10min       % slowdown of p50 per-frame latency, first vs last minute of sustained load
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echogesture.features import augment  # noqa: E402
from echogesture.landmarks import ASSETS, LandmarkExtractor, extract_video, read_frames  # noqa: E402
from echogesture.runtime import MODELS, Predictor  # noqa: E402

TARGETS = {  # metric: (op, target, unit)
    "top1_accuracy": (">=", 70, "%"),
    "latency_p95": ("<=", 50, "ms"),
    "model_size": ("<=", 25, "MB"),
    "accuracy_drop_robustness": ("<=", 5, "pt"),
    "avg_power": ("<=", 500, "mW"),
    "thermal_drift_10min": ("<=", 20, "%"),
}


def degrade(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exocentric capture conditions: low light + sensor noise, motion blur, occluder, camera roll."""
    f = frame.astype(np.float32) / 255.0
    f = np.power(f, 1.8) * 0.55  # dark scene
    f = f + rng.normal(0, 0.03, f.shape)  # high-ISO noise
    f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    f = cv2.blur(f, (5, 1))  # horizontal motion blur
    h, w = f.shape[:2]
    ow, oh = int(w * 0.18), int(h * 0.18)  # occluding object somewhere in the frame
    x, y = int(rng.integers(0, w - ow)), int(rng.integers(0, h - oh))
    f[y:y + oh, x:x + ow] = 40
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0)  # tilted phone / head
    return cv2.warpAffine(f, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def check(metric, value):
    if value is None:
        return ""
    op, target, _ = TARGETS[metric]
    return str(value >= target if op == ">=" else value <= target).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--robust-clips", type=int, default=100, help="test videos to re-run under degradation")
    ap.add_argument("--latency-frames", type=int, default=300)
    ap.add_argument("--thermal-seconds", type=int, default=60)
    ap.add_argument("--power-mw", type=float, default=None)
    ap.add_argument("--device", default=f"{platform.node().split('.')[0]} ({platform.machine()}, {platform.processor() or 'CPU'})")
    ap.add_argument("--out", type=Path, default=ROOT / "results.csv")
    args = ap.parse_args()

    pred = Predictor()
    splits = json.loads((args.data_dir / "splits.json").read_text())
    vocab = [v.upper() for v in splits["vocab"]]
    test = [i for i in splits["items"] if i["split"] == "test"] or [i for i in splits["items"] if i["split"] == "val"]
    report: dict = {}

    # ---- accuracy on cached test landmarks
    raws = [np.load(args.data_dir / i["npy"]).astype(np.float32) for i in test]
    hits = [pred.classify_landmarks(r)[0][0] == vocab[i["label"]] for r, i in zip(raws, test)]
    top1 = 100 * float(np.mean(hits))
    report["top1"] = dict(n=len(test), top1=top1)

    # ---- size
    files = [MODELS / "signnet_int8.onnx", ASSETS / "hand_landmarker.task", ASSETS / "pose_landmarker_lite.task"]
    sizes = {f.name: round(f.stat().st_size / 1e6, 3) for f in files}
    report["size_mb"] = sizes
    size_total = round(sum(sizes.values()), 2)

    # ---- per-frame streaming latency on a real clip if we have one
    manifest = args.data_dir / "manifest.jsonl"
    videos = []
    if manifest.exists():
        byid = {json.loads(l)["video_id"]: json.loads(l) for l in manifest.read_text().splitlines() if l.strip()}
        videos = [byid[i["id"]] for i in test if i["id"] in byid and Path(byid[i["id"]]["path"]).exists()]
    frames = []
    for v in videos[:20]:
        frames += [f for f, _ in read_frames(v["path"], v["frame_start"], v["frame_end"])]
        if len(frames) >= args.latency_frames:
            break
    if not frames:  # no videos available locally: worst case = detector on every frame
        frames = [np.full((480, 640, 3), 128, np.uint8)] * 50
    frames = frames[: args.latency_frames]
    feats = np.zeros((1, 32, pred.session.get_inputs()[0].shape[2]), np.float32)

    def frame_cost(ex, f):
        t0 = time.perf_counter()
        ex.process_frame(f, 66)
        kp = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        pred.session.run(None, {"features": feats})
        return kp, (time.perf_counter() - t1) * 1000

    with LandmarkExtractor() as ex:
        for f in frames[:10]:
            frame_cost(ex, f)  # warm-up
        costs = np.array([frame_cost(ex, f) for f in tqdm(frames, desc="latency")])
    total = costs.sum(1)
    report["latency_ms"] = dict(frames=len(frames), keypoints_p50=float(np.percentile(costs[:, 0], 50)),
                                keypoints_p95=float(np.percentile(costs[:, 0], 95)),
                                classifier_p50=float(np.percentile(costs[:, 1], 50)),
                                classifier_p95=float(np.percentile(costs[:, 1], 95)),
                                total_p50=float(np.percentile(total, 50)), total_p95=float(np.percentile(total, 95)))

    # ---- robustness
    rng = np.random.default_rng(0)
    if videos:
        sub = videos[: args.robust_clips]
        label = {i["id"]: vocab[i["label"]] for i in test}
        clean = deg = 0
        for v in tqdm(sub, desc="robustness"):
            r_clean = extract_video(v["path"], pred.extractor, v["frame_start"], v["frame_end"])
            r_deg = extract_video(v["path"], pred.extractor, v["frame_start"], v["frame_end"],
                                  frame_transform=lambda f: degrade(f, rng))
            clean += pred.classify_landmarks(r_clean)[0][0] == label[v["video_id"]]
            deg += pred.classify_landmarks(r_deg)[0][0] == label[v["video_id"]]
        drop = 100 * (clean - deg) / len(sub)
        report["robustness"] = dict(mode="video-degradation", n=len(sub), clean=100 * clean / len(sub),
                                    degraded=100 * deg / len(sub))
    else:
        deg_hits = [pred.classify_landmarks(augment(r, rng))[0][0] == vocab[i["label"]] for r, i in zip(raws, test)]
        drop = top1 - 100 * float(np.mean(deg_hits))
        report["robustness"] = dict(mode="landmark-simulation", n=len(test), degraded=100 * float(np.mean(deg_hits)))

    # ---- thermal drift: sustained per-frame load, compare first vs last window
    window = max(10, args.thermal_seconds // 6)
    samples, t_end, k = [], time.time() + args.thermal_seconds, 0
    with LandmarkExtractor() as ex:
        with tqdm(total=args.thermal_seconds, desc="sustained load (s)") as bar:
            start = last = time.time()
            while time.time() < t_end:
                samples.append((time.time() - start, sum(frame_cost(ex, frames[k % len(frames)]))))
                k += 1
                if time.time() - last >= 1:
                    bar.update(int(time.time() - last))
                    last = time.time()
    s = np.array(samples)
    first = np.median(s[s[:, 0] < window, 1])
    last_w = np.median(s[s[:, 0] > s[-1, 0] - window, 1])
    drift = max(0.0, 100 * (last_w - first) / first)
    report["thermal"] = dict(seconds=args.thermal_seconds, frames=len(s), first_p50=first, last_p50=last_w)

    runtime = f"ONNX Runtime {ort.__version__} INT8 CPU 1-thread + MediaPipe Tasks (CPU)"
    rows = [
        ("top1_accuracy", round(top1, 2)),
        ("latency_p95", round(report["latency_ms"]["total_p95"], 2)),
        ("model_size", size_total),
        ("accuracy_drop_robustness", round(drop, 2)),
        ("avg_power", args.power_mw),
        ("thermal_drift_10min", round(drift, 2)),
    ]
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "target", "measured", "unit", "device", "runtime", "pass"])
        for m, val in rows:
            op, tgt, unit = TARGETS[m]
            w.writerow([m, f"{op}{tgt}", "" if val is None else val, unit, args.device, runtime, check(m, val)])
    (args.out.parent / "benchmark_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))
    print(args.out.read_text())


if __name__ == "__main__":
    main()
