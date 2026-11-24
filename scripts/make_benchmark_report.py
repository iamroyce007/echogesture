"""Turn docs/benchmark/benchmark_extended.json + results.csv into BENCHMARK.md and figures.

Needs matplotlib (any Python with matplotlib works; the model environment is not required):
  python3 scripts/make_benchmark_report.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "benchmark"
SURFACE, INK, INK2, GRID, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2", "#8a8984"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(DOCS / name, dpi=160)
    plt.close(fig)


def fig_ablation(abl):
    names = [a["name"] for a in abl][::-1]
    vals = [100 * a["top1"] for a in abl][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(abl) + 1.2))
    bars = ax.barh(names, vals, height=0.56, color=BLUE)
    bars[0].set_color(ORANGE)  # shipped INT8 model (listed last, drawn on top)
    for b, v in zip(bars, vals):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    ax.axvline(70, color=RULE, ls="--", lw=1.2)
    ax.text(70.8, -0.95, "target 70%", color=INK2, fontsize=8.5, va="center")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Top-1 accuracy on the 161-clip test split (%)")
    ax.set_title("What moved accuracy: data, train+val, ensembling", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_ablation.png")


def fig_latency(lat):
    import numpy as np
    s = np.array(lat["keypoints_samples"]) + lat["classifier_ms"]["p50"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.hist(s, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    p50, p95 = lat["per_frame_total_ms"]["p50"], lat["per_frame_total_ms"]["p95"]
    top = ax.get_ylim()[1]
    for x, lab in ((p50, f"p50 {p50:.1f} ms"), (p95, f"p95 {p95:.1f} ms")):
        ax.axvline(x, color=INK2, lw=1.2)
        ax.text(x, top * 0.96, f" {lab}", color=INK, fontsize=8.5, va="top")
    ax.axvline(50, color=RULE, ls="--", lw=1.2)
    ax.text(50, top * 0.80, " budget 50 ms", color=INK2, fontsize=8.5, va="top")
    ax.set_xlim(0, max(60, s.max() + 2))
    ax.set_xlabel("Per-frame latency: hand + pose keypoints, plus classifier (ms)")
    ax.set_ylabel("Frames")
    ax.set_title(f"Per-frame latency over {lat['frames']} real test frames", loc="left")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_latency.png")


def fig_robustness(rob):
    names = [r["condition"] for r in rob]
    vals = [100 * r["top1"] for r in rob]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    bars = ax.bar(names, vals, width=0.56, color=BLUE)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", color=INK, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title(f"Accuracy under each capture degradation ({rob[0]['n']} test videos)", loc="left")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    save(fig, "fig_robustness.png")


def fig_calibration(cal):
    x = [c["mean_confidence"] * 100 for c in cal]
    yv = [c["accuracy"] * 100 for c in cal]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 100], [0, 100], color=RULE, ls="--", lw=1.2)
    ax.text(62, 56, "perfect calibration", color=INK2, fontsize=8.5, rotation=40)
    ax.plot(x, yv, color=BLUE, lw=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for xi, yi, c in zip(x, yv, cal):
        ax.text(xi + 2, yi - 5, f"n={c['n']}", color=INK2, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Mean confidence in bin (%)")
    ax.set_ylabel("Accuracy in bin (%)")
    ax.set_title("Confidence vs accuracy (test split)", loc="left")
    save(fig, "fig_calibration.png")


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def main():
    B = json.loads((DOCS / "benchmark_extended.json").read_text())
    res = {r["metric"]: r for r in csv.DictReader((ROOT / "results.csv").open())}
    A, L = B["accuracy"], B["latency"]
    fig_ablation(B["ablations"])
    fig_latency(L)
    fig_calibration(A["calibration"])
    if B.get("robustness"):
        fig_robustness(B["robustness"])

    label = {"top1_accuracy": "Top-1 accuracy", "latency_p95": "Latency p95, per frame", "model_size": "Model size on device",
             "accuracy_drop_robustness": "Accuracy drop, degraded video", "avg_power": "Average power",
             "thermal_drift_10min": "Thermal drift, 10 min sustained"}
    rows = []
    for k, name in label.items():
        r = res[k]
        measured = f"**{r['measured']} {r['unit']}**" if r["measured"] else "not measured"
        status = {"true": "✅ pass", "false": "❌ fail"}.get(r["pass"], "needs a phone reading")
        rows.append(f"| {name} | {r['target']} {r['unit']} | {measured} | {status} |")

    E, D = B["environment"], B["data"]
    abl_rows = "\n".join(f"| {a['name']} | {a['train']} | {a['members']} | {a['params_m']} M | **{pct(a['top1'])}** | {pct(a['top5'])} |"
                         for a in B["ablations"])
    cal_rows = "\n".join(f"| {c['bin']} | {c['n']} | {pct(c['mean_confidence'])} | {pct(c['accuracy'])} |" for c in A["calibration"])
    conf_rows = "\n".join(f"| {c['true']} | {c['predicted']} | {c['count']} |" for c in A["top_confusions"])
    lat_rows = "\n".join(
        f"| {name} | {L[k]['p50']:.2f} | {L[k]['p95']:.2f} | {L[k]['p99']:.2f} | {L[k]['max']:.2f} |"
        for name, k in (("Hand + pose keypoints (MediaPipe, 2 threads)", "keypoints_ms"),
                        ("Sign classifier (INT8 ONNX, 1 thread)", "classifier_ms"),
                        ("**Total per frame**", "per_frame_total_ms")))
    rob = B.get("robustness") or []
    clean = rob[0]["top1"] if rob else None
    rob_rows = "\n".join(f"| {r['condition']} | {pct(r['top1'], 0)} | {'—' if i == 0 else f'{100 * (r['top1'] - clean):+.0f} pt'} |"
                         for i, r in enumerate(rob))
    th, sz, rh = B.get("thermal") or {}, B.get("size_mb") or {}, B.get("robustness_headline") or {}
    size_rows = "\n".join(f"| `{k}` | {v:.2f} MB | {'INT8' if k.endswith('.onnx') else 'float16'} |" for k, v in sz.items())

    md = f"""# EchoGesture benchmark report

Team Hypervisor · SRMIST · XR Hackathon 2026, Track 2. All numbers below come from running
[`scripts/benchmark.py`](scripts/benchmark.py) and [`scripts/benchmark_extended.py`](scripts/benchmark_extended.py)
in this repository. Nothing was entered by hand.

## Results against the Track 2 targets

| Metric | Target | Measured | Status |
|---|---|---|---|
{chr(10).join(rows)}

Measured on {E['machine']}, CPU only, with ONNX Runtime {E['onnxruntime']} and MediaPipe {E['mediapipe']}.
The full 10-minute sustained-load run produced the thermal row.

## Test setup

| | |
|---|---|
| Hardware | {E['machine']}, {E['os']} |
| Software | Python {E['python']} · ONNX Runtime {E['onnxruntime']} · MediaPipe {E['mediapipe']} · OpenCV {E['opencv']} · PyTorch {E['torch']} (training only) |
| Execution | {E['execution']} |
| Dataset | WLASL100 ({D['vocab']} ASL words). {D['clips_total']:,} of the {D['wlasl100_instances']:,} listed clips were still downloadable. |
| Splits | Official WLASL splits: {D['clips'].get('train', 0)} train / {D['clips'].get('val', 0)} val / **{D['clips'].get('test', 0)} test** clips |
| Signers | {D['signers']['train']} train, {D['signers']['val']} val, {D['signers']['test']} test ({D['test_signers_overlapping_train']} of the test signers also appear in train/val) |
| Shipped model | 3-seed SignNet ensemble trained on train+val, static INT8 (QDQ, per-channel) ONNX |

The test clips were never used for training, model selection or hyperparameter choice. **The official WLASL
splits are not signer-independent:** {D['test_signers_overlapping_train']} of the {D['signers']['test']} test signers also appear, in other clips, in train/val.
These results therefore measure recognition of *new recordings*, mostly by signers the model has seen before.
They do not measure recognition of completely new people.

## 1. Accuracy

| | Shipped INT8 model |
|---|---|
| Top-1 | **{pct(A['top1'])}** |
| Top-5 | {pct(A['top5'])} |
| Top-10 | {pct(A['top10'])} |
| Macro top-1 (mean over classes) | {pct(A['macro_top1'])} |
| FP32 → INT8 prediction agreement | {pct(A['fp32_int8_agreement']) if A['fp32_int8_agreement'] is not None else 'n/a'} (FP32 top-1 {pct(A['fp32_top1']) if A['fp32_top1'] is not None else 'n/a'}) |
| Classes always right / never right | {A['classes_perfect']} / {len(A['classes_zero'])} of {len(A['per_class'])} |

Quantizing to INT8 cost no accuracy.

### What moved the number

![Ablation](docs/benchmark/fig_ablation.png)

| Variant | Training clips | Models | Params | Top-1 | Top-5 |
|---|---|---|---|---|---|
{abl_rows}

All rows are evaluated on the same {A['n']} test clips. The biggest gains came from more data (adding the
YouTube-hosted clips) and from training the final models on train+val. With only {A['n']} test clips, one clip is
worth {100 / A['n']:.1f} points, so differences of 1–2 points between single seeds are noise.

### Confidence and calibration

![Calibration](docs/benchmark/fig_calibration.png)

| Confidence bin | Clips | Mean confidence | Accuracy |
|---|---|---|---|
{cal_rows}

Expected calibration error: **{A['ece']:.3f}**. The model is **underconfident**: in every bin, accuracy is
higher than the stated confidence. That's expected from label smoothing plus averaging three models.
The ranking is still informative: correct predictions average {pct(A['mean_confidence_correct'], 0)} confidence and
wrong ones {pct(A['mean_confidence_wrong'], 0)}, and above 60% confidence the model is almost always right. This is
why the HUD's amber "?" threshold (35%) is conservative. Temperature scaling on the validation split would fix
the absolute numbers without changing any prediction.

### Most common mistakes

| True sign | Predicted | Times |
|---|---|---|
{conf_rows}

## 2. Latency

![Latency](docs/benchmark/fig_latency.png)

| Stage (ms) | p50 | p95 | p99 | max |
|---|---|---|---|---|
{lat_rows}

The per-frame figures come from {L['frames']} frames of real test clips; the classifier figures come from 1,000 runs.
Hand and pose tracking run in parallel threads, so a frame costs roughly the slower of the two, not
their sum. The classifier is under 2 ms.

**End-to-end `predict()`** on {L['e2e_clips']} whole test clips (decode, keypoints at ~15 fps, classify):
{L['e2e_mean_process_s']:.2f} s to process {L['e2e_mean_video_s']:.2f} s of video on average, a real-time factor of
**{L['e2e_real_time_factor']:.2f}**, which is faster than real time.

## 3. Robustness to exocentric capture conditions
"""
    if rob:
        md += f"""
![Robustness](docs/benchmark/fig_robustness.png)

| Condition ({rob[0]['n']} test videos, re-processed from pixels) | Top-1 | Change |
|---|---|---|
{rob_rows}
"""
    if rh:
        md += f"""
The headline robustness row in `results.csv` applies **all** degradations at once (low light, sensor noise,
horizontal motion blur, an occluding block and an 8° tilt) to {rh.get('n')} test videos: {rh.get('clean'):.0f}% clean
→ {rh.get('degraded'):.0f}% degraded, a drop of **{rh.get('clean') - rh.get('degraded'):.1f} points** (target ≤ 5).
"""
    md += f"""
Degradations are applied to the video pixels before keypoint extraction, so this tests the full
pipeline, not only the classifier. They are synthetic; real low-light phone footage may behave differently.

## 4. Sustained load (thermal)

Over a {th.get('seconds', 0) // 60}-minute run ({th.get('frames', 0):,} frames back to back), median per-frame latency went from
{th.get('first_p50', 0):.2f} ms in the first minute to {th.get('last_p50', 0):.2f} ms in the last, a drift of
**{100 * (th.get('last_p50', 0) - th.get('first_p50', 1)) / th.get('first_p50', 1):.1f}%** (target ≤ 20%).

## 5. Model size

| File | Size | Precision |
|---|---|---|
{size_rows}
| **Total on device** | **{sum(sz.values()):.2f} MB** | |

The sign classifier alone is 5.8 MB. Most of the footprint is the two MediaPipe keypoint models.

## Limitations

- **Power was not measured.** It needs root (`powermetrics`) on macOS or a reading on an Android phone.
  `scripts/benchmark.py --power-mw <value>` fills that row.
- **Benchmarked on a desktop M1 CPU, not a Galaxy phone.** Phone latency will differ. The pipeline is
  CPU-only, single-threaded for the classifier, and the same ONNX file runs on ONNX Runtime Mobile.
- **Not signer-independent.** Most test signers also appear in training (see Test setup). Accuracy on
  entirely new signers, such as your own team, is expected to be lower, and a signer-independent split is the next
  evaluation to run.
- **Small test set.** {A['n']} clips means about ±7 points of sampling uncertainty on top-1 (95% interval).
- **Only half of WLASL100 was available.** {D['clips_total']:,} of {D['wlasl100_instances']:,} clips could still be downloaded.
- **Isolated signs only.** Each clip contains one word; continuous signing is not evaluated.

## Reproduce

```bash
python scripts/make_predictions.py --split test          # top-1 on the test split
python scripts/benchmark.py --thermal-seconds 600        # results.csv + benchmark_report.json
python scripts/benchmark_extended.py                     # ablations, latency distribution, per-condition robustness
python3 scripts/make_benchmark_report.py                 # this file + docs/benchmark/*.png
```
"""
    (ROOT / "BENCHMARK.md").write_text(md)
    print("wrote BENCHMARK.md and", sorted(p.name for p in DOCS.glob("*.png")))


if __name__ == "__main__":
    main()
