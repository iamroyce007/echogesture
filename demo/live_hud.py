"""EchoGesture live HUD: camera -> on-device sign recognition -> AR captions (+ optional speech).

  python demo/live_hud.py                                   # webcam 0, glasses-style HUD
  python demo/live_hud.py --source clip.mp4 --record out.mp4 --headless

Keys: q quit · c clear captions · t toggle text-to-speech · m toggle hand skeleton
Everything runs locally; no frame ever leaves the device.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echogesture.hud import GlassHUD, HudState  # noqa: E402
from echogesture.landmarks import N_HAND  # noqa: E402
from echogesture.runtime import Predictor  # noqa: E402


class Speaker:
    def __init__(self):
        self.cmd = "say" if platform.system() == "Darwin" and shutil.which("say") else (
            "espeak" if shutil.which("espeak") else None)
        self.enabled = False

    def __call__(self, text: str):
        if self.enabled and self.cmd:
            subprocess.Popen([self.cmd, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Segmenter:
    """Splits the continuous stream into sign segments using hand presence + motion energy."""

    def __init__(self, min_len=10, max_len=60, gap=6, still_frames=12):
        self.buf: list[np.ndarray] = []
        self.min_len, self.max_len, self.gap, self.still_frames = min_len, max_len, gap, still_frames
        self.missing = 0
        self.still = 0

    def push(self, lm: np.ndarray):
        hands = ~np.isnan(lm[: 2 * N_HAND, 0])
        if hands.any():
            if self.buf:
                motion = np.nanmean(np.abs(lm[: 2 * N_HAND, :2] - self.buf[-1][: 2 * N_HAND, :2]))
                self.still = self.still + 1 if (np.isfinite(motion) and motion < 0.002) else 0
            self.buf.append(lm)
            self.missing = 0
        elif self.buf:
            self.missing += 1
        if self.buf and (self.missing >= self.gap or self.still >= self.still_frames or len(self.buf) >= self.max_len):
            seg = np.stack(self.buf) if len(self.buf) >= self.min_len else None
            self.buf, self.missing, self.still = [], 0, 0
            return seg
        return None

    def current(self):
        return np.stack(self.buf) if len(self.buf) >= self.min_len else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--record", type=Path, help="write the HUD output to this mp4")
    ap.add_argument("--threshold", type=float, default=0.45, help="min confidence to commit a word")
    ap.add_argument("--mirror", action="store_true", help="mirror display (selfie camera)")
    ap.add_argument("--tts", action="store_true")
    ap.add_argument("--headless", action="store_true", help="no window (for recording)")
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"cannot open source {args.source} (camera permission?)")
    is_file = isinstance(src, str)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    pred = Predictor()
    ex = pred.extractor
    seg = Segmenter()
    speak = Speaker()
    speak.enabled = args.tts
    state = HudState()
    hud = None
    writer = None
    t_start, k = time.perf_counter(), 0

    def commit(segment):
        words, probs, model_ms, _ = pred.classify_landmarks(segment, 3)
        state.model_ms = model_ms
        state.hypotheses = list(zip(words, probs))
        last = state.captions[-1][0] if state.captions else None
        if probs[0] >= args.threshold and last != words[0]:
            state.captions.append((words[0], probs[0], state.t))
            speak(words[0].lower())

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        k += 1
        if args.mirror:
            frame = cv2.flip(frame, 1)  # mirroring = left/right swap, which the model is trained to handle
        if hud is None:
            h, w = frame.shape[:2]
            hud = GlassHUD(args.width, int(args.width * h / w))
        view = cv2.resize(frame, (hud.w, hud.h))
        state.t = k / fps_in if is_file else time.perf_counter() - t_start
        t0 = time.perf_counter()
        lm = ex.process_frame(view, int(1000 / fps_in))
        state.frame_ms = 0.85 * state.frame_ms + 0.15 * (time.perf_counter() - t0) * 1000
        state.landmarks = lm
        state.signing = bool(seg.buf) or not np.isnan(lm[: 2 * N_HAND, 0]).all()
        segment = seg.push(lm)
        if segment is not None:
            commit(segment)
        elif k % 4 == 0 and (cur := seg.current()) is not None:
            words, probs, state.model_ms, _ = pred.classify_landmarks(cur, 3)
            state.hypotheses = list(zip(words, probs))

        out = hud.render(view, state)
        if args.record:
            if writer is None:
                args.record.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (hud.w, hud.h))
            writer.write(out)
        if not args.headless:
            cv2.imshow("EchoGesture", out)
            key = cv2.waitKey(1 if not is_file else max(1, int(1000 / fps_in))) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                state.captions.clear()
            if key == ord("t"):
                speak.enabled = not speak.enabled
            if key == ord("m"):
                state.show_skeleton = not state.show_skeleton

    if (cur := seg.current()) is not None:  # flush a sign still in progress when the stream ends
        commit(cur)
    cap.release()
    if writer:
        writer.release()
        print(f"saved {args.record}")
    cv2.destroyAllWindows()
    print("captions:", " ".join(w for w, _, _ in state.captions))


if __name__ == "__main__":
    main()
