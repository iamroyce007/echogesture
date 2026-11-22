# AI Disclosure

**Team Hypervisor · SRMIST · XR Hackathon 2026, Track 2 (EchoGesture)**

This project was built with substantial help from an AI coding assistant: **Claude Code** (Anthropic),
running the Claude Opus 5.5 model. We disclose this in full.

## What the team did
- Chose the track, set the goals and the scope of the work, and directed every step.
- Set the product direction: an exocentric, glasses-style AR caption experience, and a demo built around it.
- Reviewed the outputs, results, slides and demo video, and made the decisions about what to keep.
- Provided all team, college and submission details.

## What the AI assistant did, under the team's direction
- Wrote most of the source code: keypoint extraction, feature engineering, the SignNet model, training,
  INT8 export, `predict()`, the benchmark, the live HUD and the demo-video generator.
- Ran the data download, landmark extraction, training, quantization and benchmarks on our machine, and
  reported the measured results.
- Designed the HUD and generated the demo video from WLASL test clips. The video's **voice-over is synthetic**, produced with macOS text-to-speech (voice "Samantha") from a script written with the assistant. The spoken words during the signing scenes are also text-to-speech of the model's predictions.
- Drafted the README, the submission deck content and layout, and the video recording script.

## What was not AI-generated
- **The training and evaluation data:** WLASL (Li et al., 2020), used under its research license.
- **The keypoint models:** Google MediaPipe hand and pose landmarkers (Apache 2.0).
- **All reported numbers** (accuracy, latency, size, robustness, thermal drift) were measured by running
  the code in this repository (`scripts/benchmark.py`). None were estimated or written by hand.
  Raw outputs are in `results.csv`, `benchmark_report.json` and `predictions_wlasl100_test.jsonl`.

## Third-party assets
- Inter font (SIL Open Font License), in `assets/fonts/`.
- Python libraries: PyTorch, ONNX Runtime, MediaPipe, OpenCV, Pillow, NumPy (their respective licenses).
