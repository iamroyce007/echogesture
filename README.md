# EchoGesture — on-device sign recognition (XR Hackathon 2026 · Track 2)

Exocentric isolated-sign recognition: **live camera video → ASL gloss → captions (HUD) / speech**, running
fully on the device. No frame, keypoint or result is uploaded anywhere.

```
camera frame ──► MediaPipe Hand + Pose landmarkers ──► 55 3D keypoints / frame
                 (on-device, float16, ~13.6 MB)            │
                                                           ▼
            pose-invariant features (body frame, roll-corrected, hand-local shape, velocity, occlusion flags)
                                                           │  32-frame window
                                                           ▼
                 SignNet: depthwise-separable dilated temporal CNN + attention pooling
                 (1.7 M params, INT8 ONNX ≈ 1.9 MB, ~0.5 ms on 1 CPU thread)
                                                           │
                                                           ▼
                     gloss + top-5 + confidence ──► AR caption HUD / text-to-speech
```

Why keypoints, not pixels: the classifier never sees faces or backgrounds, which is privacy-friendly. It is
also largely invariant to lighting, clothing, and camera position, and it is tiny enough for a wearable
budget.

## Results (WLASL100, held-out test signers · Mac mini M1, CPU only)

| Metric | Target | Measured | Pass |
| --- | --- | --- | --- |
| Top-1 accuracy | ≥ 70 % | **71.4 %** (top-5 88.8 %) | ✅ |
| Latency p95, per frame (keypoints + classifier) | ≤ 50 ms | **39.7 ms** (classifier alone 1.6 ms) | ✅ |
| Model size on device | ≤ 25 MB | **19.4 MB** total; sign model 5.8 MB INT8 | ✅ |
| Accuracy drop, degraded video | ≤ 5 pt | **3.0 pt** (76 → 73 % on 100 clips) | ✅ |
| Thermal drift, 10 min sustained | ≤ 20 % | **2.1 %** | ✅ |
| Average power | ≤ 500 mW | needs an on-device reading (`--power-mw`) | — |

Data: 1,019 of the 2,038 WLASL100 clips were still downloadable. The final model is an ensemble of 3 seeds
trained on train + val (858 clips), and test (161 clips) was never used for training or model selection.
Full numbers are in `results.csv` and `benchmark_report.json`.

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate     # or python3.11 -m venv .venv
pip install -r requirements.txt
# MediaPipe landmarker models (already in assets/ if you cloned with them)
curl -L -o assets/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o assets/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

Docker (CPU inference / evaluation):
```bash
docker build -t echogesture .
docker run --rm -v /path/to/test_clips:/clips -v $PWD/out:/out echogesture \
    python scripts/make_predictions.py --clips-dir /clips --out /out/predictions.jsonl
```

## Evaluator interface

```python
from predict import predict
predict("clips/test_0412.mp4")
# {"clip_id": "test_0412", "prediction": "HAPPY", "top5": ["HAPPY", ...], "confidence": 0.81, "latency_ms": 0.6}
```

* `ECHOGESTURE_VOCAB=vocab.txt` (one word per line) restricts outputs to the official vocabulary.
* `latency_ms` is the recognition-model time per clip. Per-frame keypoint cost is streamed as frames arrive
  and is reported, along with the end-to-end per-frame p95, in `results.csv`.

Batch file for submission:
```bash
python scripts/make_predictions.py --clips-dir /path/to/test_clips --out predictions.jsonl
```

## Reproduce training

```bash
python scripts/download_wlasl.py --subset 100            # or --from-dir <preprocessed WLASL videos/>
python scripts/extract_landmarks.py --workers 4          # -> data/landmarks/*.npy, data/splits.json
# final submission model = 3 seeds trained on train+val, fused into one INT8 graph
for s in 0 1 2; do python scripts/train.py --epochs 200 --seed $s --train-on-val --out-dir checkpoints_full_s$s; done
python scripts/export.py --ckpt checkpoints_full_s{0,1,2}/model.pt --calib checkpoints_full_s0/calib.npy
                                                         # -> models/signnet_int8.onnx (+ fp32, vocab.json)
python scripts/make_predictions.py --split test          # self-check on held-out signers
python scripts/benchmark.py --thermal-seconds 600        # -> results.csv, benchmark_report.json
```

`scripts/make_synthetic.py` generates fake landmark data for a 1-minute pipeline smoke test
(`--data-dir data_synth`). Its accuracy numbers are meaningless.

## Live demo (AR caption HUD)

```bash
python demo/live_hud.py                      # webcam; q quit · c clear · t speech · m skeleton
python demo/live_hud.py --source clip.mp4 --record demo_out.mp4 --headless   # render for the demo video
```
The glasses-style HUD ([echogesture/hud.py](echogesture/hud.py)) is designed for the wearer's
(exocentric) point of view:

* **Caption bubble anchored to the signer.** A frosted-glass card floats beside the signer's head, with a
  tail pointing at them. It never covers the face, which carries ASL grammar; when there is no room beside
  the signer, it drops to a lower-third subtitle. New words pop in with a short ease-out, and the bubble
  shows a confidence meter and the running sentence.
* **Signer focus brackets** follow the tracked body and hands, with a "Signing / Signer in view" state.
* **Recognition dock** with an animated waveform while signing and the live top-3 guesses as chips.
* **Status bar** with an on-device indicator, ASL → EN, and live per-frame latency.
* Optional text-to-speech (`t`) for two-way conversation.

Words are segmented from the continuous stream using hand presence and motion energy.

### Demo video

```bash
python scripts/make_demo_video.py --voice     # -> demo/echogesture_demo.mp4 (≈55 s, 1280x720, H.264 + AAC)
```
The video has intro and "how it works" cards, followed by three POV scenes: two different signers, then a
harsh-conditions scene with low light, motion blur and an occluder. A results card at the end is read from
`results.csv`. Every caption in the video is produced live by the model on WLASL **test** clips, from signers
not seen in training. For the final submission, record your own team member signing with
`demo/live_hud.py --record`.

## Robustness measures (exocentric conditions)

| Condition | Handling |
| --- | --- |
| Angled / moving signer | Body-centric normalization, shoulder-line roll correction, yaw/shear/scale augmentation |
| Left-handed signers | Mirror augmentation with left/right body-part swap |
| Occlusion | Per-frame hand dropout, fingertip dropout, presence flags; hands assigned via pose wrists |
| Low light / noise | Low landmarker thresholds + temporal tracking, keypoint jitter augmentation |
| Fast cadence | Speed/crop augmentation, fixed 32-step resampling of each sign segment |

`scripts/benchmark.py` re-runs test videos under synthetic low light, sensor noise, motion blur, an
occluding block, and an 8° camera tilt, then reports the accuracy drop.

## Deployment to phone / glasses

* `models/signnet_int8.onnx` runs with `onnxruntime-android` (CPU, or the NNAPI/QNN execution providers on
  Snapdragon). The MediaPipe Tasks hand/pose landmarkers have native Android SDKs that use the same
  `.task` files.
* Ops used: Conv1d, BatchNorm (folded), SiLU, Softmax, MatMul. These convert directly to TFLite / LiteRT
  if required.

## Submission deck

`docs/CollegeName_TeamName_Submission.pptx` (and a PDF copy). Fill in the team, college, member and demo-link fields marked `[...]`.

## Layout

```
predict.py                  evaluator entry point
echogesture/landmarks.py    MediaPipe keypoint extraction, hand <-> body assignment
echogesture/features.py     normalization, features, augmentation
echogesture/model.py        SignNet
echogesture/runtime.py      INT8 ONNX predictor
scripts/                    data download, landmark caching, training, export, predictions, benchmark
demo/live_hud.py            live AR caption HUD
```
