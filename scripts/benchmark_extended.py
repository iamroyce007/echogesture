"""Extended benchmark: accuracy analysis, ablations, latency distribution, per-condition robustness.

Everything is measured on the same held-out WLASL100 test split (161 clips, signers unseen in training).
Writes docs/benchmark/benchmark_extended.json, which scripts/make_benchmark_report.py turns into
BENCHMARK.md and figures.

  python scripts/benchmark_extended.py                 # ~10 min on an M1 (robustness re-extracts video)
  python scripts/benchmark_extended.py --robust-clips 0   # skip the slow per-condition robustness part
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echogesture.features import build_features  # noqa: E402
from echogesture.landmarks import LandmarkExtractor, extract_video, read_frames  # noqa: E402
from echogesture.model import SignNet  # noqa: E402
from echogesture.runtime import MODELS, Predictor  # noqa: E402

# (name, checkpoint dirs, training clips description)
ABLATIONS = [
    ("Baseline: 1 model, direct-link clips only", ["checkpoints_baseline"], "407 train"),
    ("Small model (0.5 M params)", ["checkpoints_small"], "407 train"),
    ("1 model, seed 0", ["checkpoints"], "680 train"),
    ("1 model, seed 1", ["checkpoints_s1"], "680 train"),
    ("1 model, seed 2", ["checkpoints_s2"], "680 train"),
    ("3-seed ensemble", ["checkpoints", "checkpoints_s1", "checkpoints_s2"], "680 train"),
    ("1 model, train+val, seed 0", ["checkpoints_full_s0"], "858 train+val"),
    ("1 model, train+val, seed 1", ["checkpoints_full_s1"], "858 train+val"),
    ("1 model, train+val, seed 2", ["checkpoints_full_s2"], "858 train+val"),
    ("Final: 3-seed ensemble, train+val (FP32)", ["checkpoints_full_s0", "checkpoints_full_s1", "checkpoints_full_s2"],
     "858 train+val"),
]


# ----------------------------------------------------------------------------- degradations (one at a time)
def low_light(f, rng):
    x = np.power(f.astype(np.float32) / 255.0, 1.8) * 0.55 + rng.normal(0, 0.03, f.shape)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def motion_blur(f, rng):
    return cv2.blur(f, (9, 1))


def occluder(f, rng):
    f = f.copy()
    h, w = f.shape[:2]
    ow, oh = int(w * 0.18), int(h * 0.18)
    x, y = int(rng.integers(0, w - ow)), int(rng.integers(0, h - oh))
    f[y:y + oh, x:x + ow] = 40
    return f


def tilt(f, rng):
    h, w = f.shape[:2]
    return cv2.warpAffine(f, cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0), (w, h), borderMode=cv2.BORDER_REFLECT)


def combined(f, rng):
    return tilt(occluder(motion_blur(low_light(f, rng), rng), rng), rng)


CONDITIONS = [("Clean", None), ("Low light + sensor noise", low_light), ("Motion blur", motion_blur),
              ("Occluding object", occluder), ("Camera tilt 8°", tilt), ("All combined", combined)]


def load_member(d):
    ck = torch.load(ROOT / d / "model.pt", map_location="cpu")
    c = ck["config"]
    m = SignNet(c["num_classes"], dim=c["dim"], depth=c["depth"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--robust-clips", type=int, default=60)
    ap.add_argument("--latency-frames", type=int, default=300)
    ap.add_argument("--e2e-clips", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "benchmark" / "benchmark_extended.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    splits = json.loads((args.data_dir / "splits.json").read_text())
    vocab = [v.upper() for v in splits["vocab"]]
    counts = Counter(i["split"] for i in splits["items"])
    test = [i for i in splits["items"] if i["split"] == "test"]
    y = np.array([i["label"] for i in test])
    raws = [np.load(args.data_dir / i["npy"]).astype(np.float32) for i in test]
    X = np.stack([build_features(r) for r in raws])
    man = {}
    for l in (args.data_dir / "manifest.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            man[r["video_id"]] = r
    signers = {s: len({man[i["id"]]["signer_id"] for i in splits["items"] if i["split"] == s and i["id"] in man})
               for s in ("train", "val", "test")}
    test_signers = {man[i["id"]]["signer_id"] for i in test}
    train_signers = {man[i["id"]]["signer_id"] for i in splits["items"] if i["split"] != "test"}
    out: dict = {}

    out["environment"] = dict(
        machine="Mac mini (Apple M1, 8 GB RAM)" if platform.machine() == "arm64" and platform.system() == "Darwin" else platform.platform(),
        os=f"{platform.system()} {platform.release()}", python=platform.python_version(),
        onnxruntime=ort.__version__, torch=torch.__version__,
        mediapipe=__import__("mediapipe").__version__, opencv=cv2.__version__,
        execution="CPU only, ONNX Runtime 1 intra-op thread for the classifier",
    )
    out["data"] = dict(vocab=len(vocab), clips=dict(counts), clips_total=sum(counts.values()),
                       signers=signers, test_signers_overlapping_train=len(test_signers & train_signers),
                       wlasl100_instances=2038)

    # ------------------------------------------------------------------ accuracy of the shipped INT8 model
    pred = Predictor()
    probs8 = []
    for f in X:
        logits = pred.session.run(None, {"features": f[None]})[0][0]
        e = np.exp(logits - logits.max())
        probs8.append(e / e.sum())
    P = np.array(probs8)
    rank = np.argsort(-P, 1)
    top = lambda k: float((rank[:, :k] == y[:, None]).any(1).mean())  # noqa: E731
    pred1 = rank[:, 0]
    conf = P.max(1)
    per_class = defaultdict(lambda: [0, 0])
    for t, p in zip(y, pred1):
        per_class[vocab[t]][1] += 1
        per_class[vocab[t]][0] += int(t == p)
    macro = float(np.mean([c / n for c, n in per_class.values()]))
    confusions = Counter((vocab[t], vocab[p]) for t, p in zip(y, pred1) if t != p)
    bins = np.linspace(0, 1, 6)
    calib, ece = [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf <= hi)
        if m.sum():
            acc, mc = float((pred1[m] == y[m]).mean()), float(conf[m].mean())
            calib.append(dict(bin=f"{lo:.1f}-{hi:.1f}", n=int(m.sum()), accuracy=acc, mean_confidence=mc))
            ece += m.sum() / len(y) * abs(acc - mc)
    # FP32 graph vs INT8 graph agreement
    fp32_path = MODELS / "signnet_fp32.onnx"
    agree = fp32_top1 = None
    if fp32_path.exists():
        s32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        p32 = np.array([s32.run(None, {"features": f[None]})[0][0].argmax() for f in X])
        agree, fp32_top1 = float((p32 == pred1).mean()), float((p32 == y).mean())
    out["accuracy"] = dict(
        n=len(y), top1=top(1), top5=top(5), top10=top(10), macro_top1=macro,
        mean_confidence_correct=float(conf[pred1 == y].mean()), mean_confidence_wrong=float(conf[pred1 != y].mean()),
        calibration=calib, ece=float(ece), fp32_top1=fp32_top1, fp32_int8_agreement=agree,
        classes_perfect=sum(1 for c, n in per_class.values() if c == n),
        classes_zero=sorted(k for k, (c, n) in per_class.items() if c == 0),
        top_confusions=[dict(true=a, predicted=b, count=c) for (a, b), c in confusions.most_common(8)],
        per_class={k: dict(correct=c, total=n) for k, (c, n) in sorted(per_class.items())},
    )
    print("accuracy", {k: out["accuracy"][k] for k in ("top1", "top5", "top10", "macro_top1", "ece")})

    # ------------------------------------------------------------------ ablations on the same test split
    Xt = torch.from_numpy(X)
    abl = []
    cache = {}
    for name, dirs, train_desc in ABLATIONS:
        if not all((ROOT / d / "model.pt").exists() for d in dirs):
            continue
        ps, params = [], 0
        for d in dirs:
            if d not in cache:
                m, n = load_member(d)
                with torch.no_grad():
                    cache[d] = (torch.softmax(m(Xt), 1).numpy(), n)
            ps.append(cache[d][0])
            params += cache[d][1]
        Pm = np.mean(ps, 0)
        r = np.argsort(-Pm, 1)
        abl.append(dict(name=name, train=train_desc, members=len(dirs), params_m=round(params / 1e6, 2),
                        top1=float((r[:, 0] == y).mean()), top5=float((r[:, :5] == y[:, None]).any(1).mean())))
    abl.append(dict(name="Shipped: final ensemble, INT8 ONNX", train="858 train+val", members=3,
                    params_m=abl[-1]["params_m"] if abl else None, top1=top(1), top5=top(5)))
    out["ablations"] = abl
    for a in abl:
        print(f"  {a['name']:48s} top1 {a['top1']:.3f} top5 {a['top5']:.3f}")

    # ------------------------------------------------------------------ latency distribution
    vids = [man[i["id"]] for i in test if i["id"] in man and Path(man[i["id"]]["path"]).exists()]
    frames = []
    for v in vids[:25]:
        frames += [f for f, _ in read_frames(v["path"], v["frame_start"], v["frame_end"])]
        if len(frames) >= args.latency_frames:
            break
    frames = frames[: args.latency_frames]
    feat = X[:1]
    kp, cl = [], []
    with LandmarkExtractor() as ex:
        for f in frames[:10]:
            ex.process_frame(f, 66)
        for f in frames:
            t0 = time.perf_counter()
            ex.process_frame(f, 66)
            kp.append((time.perf_counter() - t0) * 1000)
    for _ in range(10):
        pred.session.run(None, {"features": feat})
    for _ in range(1000):
        t0 = time.perf_counter()
        pred.session.run(None, {"features": feat})
        cl.append((time.perf_counter() - t0) * 1000)
    kp, cl = np.array(kp), np.array(cl)
    total = kp + np.percentile(cl, 50)
    pct = lambda a: {f"p{q}": float(np.percentile(a, q)) for q in (50, 90, 95, 99)} | {"mean": float(a.mean()), "max": float(a.max())}  # noqa: E731
    # end-to-end predict() on whole clips (offline mode the evaluator uses)
    e2e = []
    for v in vids[: args.e2e_clips]:
        cap = cv2.VideoCapture(v["path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        n = sum(1 for _ in read_frames(v["path"], v["frame_start"], v["frame_end"], target_fps=None))
        # time exactly what predict() does on a trimmed clip: decode + keypoints at ~15 fps + classify.
        # (Many WLASL clips are segments of a longer source video, so pass the segment's frame range.)
        t0 = time.perf_counter()
        raw = extract_video(v["path"], pred.extractor, v["frame_start"], v["frame_end"])
        pred.classify_landmarks(raw)
        e2e.append(dict(seconds_of_video=n / fps, seconds_to_process=time.perf_counter() - t0))
    rtf = [e["seconds_to_process"] / max(e["seconds_of_video"], 1e-6) for e in e2e]
    out["latency"] = dict(frames=len(frames), keypoints_ms=pct(kp), classifier_ms=pct(cl), per_frame_total_ms=pct(total),
                          keypoints_samples=[round(float(v), 2) for v in kp],
                          e2e_clips=len(e2e), e2e_mean_video_s=float(np.mean([e["seconds_of_video"] for e in e2e])),
                          e2e_mean_process_s=float(np.mean([e["seconds_to_process"] for e in e2e])),
                          e2e_real_time_factor=float(np.mean(rtf)))
    print("latency per-frame p95", out["latency"]["per_frame_total_ms"]["p95"], "rtf", out["latency"]["e2e_real_time_factor"])

    # ------------------------------------------------------------------ robustness, one degradation at a time
    if args.robust_clips:
        sub = vids[: args.robust_clips]
        label = {i["id"]: vocab[i["label"]] for i in test}
        rob = []
        for cname, fn in CONDITIONS:
            rng = np.random.default_rng(0)
            hits = 0
            t0 = time.perf_counter()
            for v in sub:
                raw = extract_video(v["path"], pred.extractor, v["frame_start"], v["frame_end"],
                                    frame_transform=(lambda f, fn=fn: fn(f, rng)) if fn else None)
                hits += pred.classify_landmarks(raw)[0][0] == label[v["video_id"]]
            rob.append(dict(condition=cname, n=len(sub), top1=hits / len(sub)))
            print(f"  {cname:26s} {hits}/{len(sub)}  ({time.perf_counter() - t0:.0f}s)")
        out["robustness"] = rob

    elif args.out.exists():  # keep the previous per-condition run when skipping the slow part
        prev = json.loads(args.out.read_text()).get("robustness")
        if prev:
            out["robustness"] = prev
            print("kept previous robustness results")

    # ------------------------------------------------------------------ carry over the full-run numbers
    rep = ROOT / "benchmark_report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        out["thermal"] = r.get("thermal")
        out["size_mb"] = r.get("size_mb")
        out["robustness_headline"] = r.get("robustness")
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
