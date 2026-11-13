"""Render the EchoGesture demo video (1280x720, H.264): title cards + POV scenes through the live pipeline.

Every caption in the video is produced by the real system (MediaPipe keypoints + INT8 ONNX model) running
frame by frame on held-out WLASL test clips, i.e. signers never seen during training.

  python scripts/make_demo_video.py --out demo/echogesture_demo.mp4 --voice
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from echogesture.hud import ACCENT, AMBER, GLASS, GREEN, SUB, WHITE, GlassHUD, HudState, font, rr_mask, text_w  # noqa: E402
from echogesture.landmarks import LH, N_HAND, RH, LandmarkExtractor, read_frames  # noqa: E402
from echogesture.runtime import Predictor  # noqa: E402

W, H, FPS = 1280, 720, 30

# (scene tag, signer_id, glosses, degrade?)  — all clips come from the WLASL *test* split
SCENES = [
    ("Conversation", 9, ["MOTHER", "WORK", "FINISH"], False),
    ("New signer", 5, ["WOMAN", "DRINK", "HOT"], False),
    ("Low light · blur · occlusion", 9, ["TABLE", "BED"], True),
]


# ----------------------------------------------------------------------------- helpers

def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def backdrop(t: float) -> Image.Image:
    """Dark gradient with a slowly drifting accent glow."""
    y = np.linspace(0, 1, H)[:, None, None]
    top, bot = np.array([16, 19, 28]), np.array([5, 6, 9])
    img = (top * (1 - y) + bot * y) * np.ones((1, W, 1))
    im = Image.fromarray(img.astype(np.uint8)).convert("RGBA")
    glow = Image.new("RGBA", (W // 4, H // 4), (0, 0, 0, 0))
    gx = W // 8 + int(40 * math.sin(t * 0.6))
    ImageDraw.Draw(glow).ellipse((gx - 70, -40, gx + 110, 120), fill=ACCENT + (70,))
    glow = glow.filter(ImageFilter.GaussianBlur(30)).resize((W, H), Image.BILINEAR)
    im.alpha_composite(glow)
    return im


def chip(d, x, y, text, f, fg=WHITE, bg=(255, 255, 255, 26), outline=(255, 255, 255, 60), h=40, anchor="l"):
    w = text_w(text, f) + 36
    if anchor == "c":
        x -= w // 2
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=bg, outline=outline, width=1)
    d.text((x + w / 2, y + h / 2), text, font=f, fill=fg, anchor="mm")
    return x + w


def fade(img: np.ndarray, a: float) -> np.ndarray:
    return (img.astype(np.float32) * a).astype(np.uint8) if a < 1 else img


def to_bgr(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------- title cards

def card_intro(t, dur):
    im = backdrop(t)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    a = ease(t / 0.8)
    lift = int((1 - a) * 20)
    al = int(255 * a)
    d.text((W / 2, 250 + lift), "EchoGesture", font=font(84, "Bold"), fill=WHITE + (al,), anchor="mm")
    d.text((W / 2, 330 + lift), "Real-time sign language captions, seen through your glasses.",
           font=font(26, "Regular"), fill=SUB + (al,), anchor="mm")
    a2 = int(255 * ease((t - 0.6) / 0.8))
    x = W / 2 - 330
    for txt, col in (("On-device", GREEN), ("No cloud · 0 bytes uploaded", ACCENT), ("ASL → English", AMBER)):
        w = text_w(txt, font(18, "Medium")) + 52
        d.rounded_rectangle((x, 410, x + w, 452), radius=21, fill=(255, 255, 255, int(a2 * 0.1)),
                            outline=(255, 255, 255, int(a2 * 0.25)))
        d.ellipse((x + 18, 426, x + 28, 436), fill=col + (a2,))
        d.text((x + 36, 431), txt, font=font(18, "Medium"), fill=WHITE + (a2,), anchor="lm")
        x += w + 14
    d.text((W / 2, 640), "XR Hackathon 2026  ·  Track 2  ·  EchoGesture", font=font(16, "Medium"),
           fill=(150, 156, 170, a2), anchor="mm")
    im.alpha_composite(ov)
    return to_bgr(im)


def card_pipeline(t, dur):
    im = backdrop(t + 5).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    d.text((W / 2, 118), "How it works", font=font(44, "Bold"), fill=WHITE, anchor="mm")
    d.text((W / 2, 168), "Every step runs on the wearable. The video never leaves the device.",
           font=font(20, "Regular"), fill=SUB, anchor="mm")
    steps = [("Camera", "Exocentric view of\nthe person you talk to", "15–30 fps"),
             ("Keypoints", "Hands + upper body\n55 points per frame", "MediaPipe · on-device"),
             ("Sign model", "Temporal CNN ensemble\nover 32-frame window", "INT8 · 5.8 MB"),
             ("AR caption", "Word anchored next\nto the signer + voice", "Glanceable HUD")]
    bw, bh, gap = 250, 230, 40
    x0 = (W - 4 * bw - 3 * gap) // 2
    for i, (title, body, foot) in enumerate(steps):
        a = ease((t - 0.25 * i) / 0.6)
        if a <= 0:
            continue
        x = x0 + i * (bw + gap)
        y = 250 + int((1 - a) * 24)
        im.paste(Image.new("RGB", (bw, bh), GLASS), (x, y), rr_mask(bw, bh, 26).point(lambda v: int(v * 0.86 * a)))
        dd = ImageDraw.Draw(im, "RGBA")
        dd.rounded_rectangle((x, y, x + bw - 1, y + bh - 1), radius=26, outline=(255, 255, 255, int(60 * a)))
        dd.ellipse((x + 24, y + 26, x + 64, y + 66), fill=ACCENT)
        dd.text((x + 44, y + 46), str(i + 1), font=font(20, "Bold"), fill=WHITE, anchor="mm")
        dd.text((x + 24, y + 92), title, font=font(26, "Bold"), fill=WHITE)
        dd.multiline_text((x + 24, y + 134), body, font=font(16, "Regular"), fill=SUB, spacing=6)
        dd.text((x + 24, y + bh - 34), foot, font=font(15, "SemiBold"), fill=ACCENT)
        if i < 3:
            ax = x + bw + 8
            dd.line((ax, y + bh / 2, ax + gap - 16, y + bh / 2), fill=(255, 255, 255, int(120 * a)), width=2)
            dd.polygon([(ax + gap - 16, y + bh / 2 - 6), (ax + gap - 8, y + bh / 2), (ax + gap - 16, y + bh / 2 + 6)],
                       fill=(255, 255, 255, int(120 * a)))
    return to_bgr(im)


def card_scene(title, sub, t):
    im = backdrop(t + 9)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    a = int(255 * ease(t / 0.5))
    d.text((W / 2, H / 2 - 24), title, font=font(48, "Bold"), fill=WHITE + (a,), anchor="mm")
    d.text((W / 2, H / 2 + 32), sub, font=font(20, "Regular"), fill=SUB + (a,), anchor="mm")
    im.alpha_composite(ov)
    return to_bgr(im)


def card_results(t, results: dict):
    im = backdrop(t + 13).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    d.text((W / 2, 100), "Measured on-device", font=font(44, "Bold"), fill=WHITE, anchor="mm")
    d.text((W / 2, 148), f"{results.get('_device', '')}  ·  WLASL100 held-out signers",
           font=font(18, "Regular"), fill=SUB, anchor="mm")
    tiles = [("Top-1 accuracy", "top1_accuracy", "{:.1f}%", "target ≥ 70%"),
             ("Latency p95 / frame", "latency_p95", "{:.0f} ms", "target ≤ 50 ms"),
             ("All models on device", "model_size", "{:.1f} MB", "sign model 5.8 MB INT8 · target ≤ 25"),
             ("Robustness drop", "accuracy_drop_robustness", "{:.1f} pt", "low light, blur, occlusion ≤ 5"),
             ("Thermal drift, 10 min", "thermal_drift_10min", "{:.1f}%", "sustained load ≤ 20%"),
             ("Cloud uploads", None, "0 bytes", "fully on-device")]
    tw, th, gx, gy = 340, 160, 30, 28
    x0, y0 = (W - 3 * tw - 2 * gx) // 2, 205
    for i, (label, key, fmt, foot) in enumerate(tiles):
        a = ease((t - 0.12 * i) / 0.5)
        if a <= 0:
            continue
        x, y = x0 + (i % 3) * (tw + gx), y0 + (i // 3) * (th + gy) + int((1 - a) * 18)
        im.paste(Image.new("RGB", (tw, th), GLASS), (x, y), rr_mask(tw, th, 24).point(lambda v: int(v * 0.88 * a)))
        dd = ImageDraw.Draw(im, "RGBA")
        dd.rounded_rectangle((x, y, x + tw - 1, y + th - 1), radius=24, outline=(255, 255, 255, int(55 * a)))
        val = results.get(key) if key else None
        ok = results.get(f"{key}_pass", "true") == "true" if key else True
        txt = fmt.format(val) if key and val is not None else ("—" if key else fmt)
        dd.text((x + 24, y + 26), label, font=font(17, "Medium"), fill=SUB)
        dd.text((x + 24, y + 58), txt, font=font(44, "Bold"), fill=WHITE)
        dd.ellipse((x + tw - 40, y + 28, x + tw - 26, y + 42), fill=GREEN if ok else AMBER)
        dd.text((x + 24, y + th - 32), foot, font=font(14, "Regular"), fill=(150, 156, 170))
    return to_bgr(im)


def card_outro(t):
    im = backdrop(t + 20)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    a = int(255 * ease(t / 0.6))
    d.text((W / 2, 300), "EchoGesture", font=font(64, "Bold"), fill=WHITE + (a,), anchor="mm")
    d.text((W / 2, 362), "Conversation without the gap. Privacy without compromise.",
           font=font(22, "Regular"), fill=SUB + (a,), anchor="mm")
    d.text((W / 2, 640), "Demo clips: WLASL test split (signers unseen in training). All captions produced live by the model.",
           font=font(14, "Regular"), fill=(140, 146, 160, a), anchor="mm")
    im.alpha_composite(ov)
    return to_bgr(im)


# ----------------------------------------------------------------------------- POV composition

def compose_pov(frame: np.ndarray) -> np.ndarray:
    """Place the clip like a person standing in front of the wearer, slightly left of centre, so the
    caption can float beside them. Surroundings are a soft, darkened extension of the scene."""
    h, w = frame.shape[:2]
    sc = max(W / w, H / h)
    bg = cv2.resize(frame, (int(w * sc), int(h * sc)))
    bg = bg[(bg.shape[0] - H) // 2:(bg.shape[0] - H) // 2 + H, (bg.shape[1] - W) // 2:(bg.shape[1] - W) // 2 + W]
    bg = (cv2.GaussianBlur(bg, (0, 0), 30).astype(np.float32) * 0.8).astype(np.uint8)
    fh = H
    fw = min(int(w * fh / h), W)
    fg = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_CUBIC)
    if fw >= W:  # widescreen source: full bleed
        return fg
    # left-aligned, fading into the out-of-focus surroundings on the right (shallow depth of field)
    ramp = int(fw * 0.28)
    xs = np.ones(fw, np.float32)
    r = np.linspace(1, 0, ramp)
    xs[-ramp:] = r * r * (3 - 2 * r)
    m = xs[None, :, None]
    roi = bg[:, :fw].astype(np.float32)
    bg[:, :fw] = (fg * m + roi * (1 - m)).astype(np.uint8)
    return bg


def degrade(frame: np.ndarray, rng, k: int) -> np.ndarray:
    f = frame.astype(np.float32) / 255.0
    f = np.power(f, 1.8) * 0.55 + rng.normal(0, 0.03, f.shape)
    f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    f = cv2.blur(f, (5, 1))
    h, w = f.shape[:2]
    ox = int(w * (0.62 + 0.04 * math.sin(k / 9)))  # occluding object drifting across the right side
    cv2.rectangle(f, (ox, int(h * 0.55)), (ox + int(w * 0.16), int(h * 0.80)), (35, 35, 38), -1)
    return f


def has_hand(lm):
    return not (np.isnan(lm[LH, 0]).all() and np.isnan(lm[RH, 0]).all())


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "demo" / "echogesture_demo.mp4")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--voice", action="store_true", help="speak each committed word (macOS `say`)")
    args = ap.parse_args()

    man = {json.loads(l)["video_id"]: json.loads(l) for l in (args.data_dir / "manifest.jsonl").read_text().splitlines() if l.strip()}
    splits = json.loads((args.data_dir / "splits.json").read_text())
    test_ids = {i["id"] for i in splits["items"] if i["split"] == "test"}
    results = {}
    if (ROOT / "results.csv").exists():
        for r in csv.DictReader((ROOT / "results.csv").open()):
            if r["measured"]:
                results[r["metric"]] = float(r["measured"])
                results[f"{r['metric']}_pass"] = r["pass"]
            results["_device"] = r["device"]

    pred = Predictor()
    hud = GlassHUD(W, H)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    silent = tmp / "video.mp4"
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                            "-crf", "18", "-pix_fmt", "yuv420p", str(silent)], stdin=subprocess.PIPE)
    n_frames = 0
    speech: list[tuple[float, str]] = []

    def emit(img):
        nonlocal n_frames
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
        n_frames += 1

    def section(fn, dur, fade_in=0.4, fade_out=0.4):
        n = int(dur * FPS)
        for i in range(n):
            t = i / FPS
            a = min(1.0, ease(t / fade_in) if fade_in else 1.0, ease((dur - t) / fade_out) if fade_out else 1.0)
            emit(fade(fn(t), a))

    section(lambda t: card_intro(t, 4.5), 4.5)
    section(lambda t: card_pipeline(t, 6.0), 6.0)

    rng = np.random.default_rng(0)
    for si, (tag, signer, glosses, bad) in enumerate(SCENES):
        sub = "Same model, harsh capture conditions" if bad else f"Signer unseen during training  ·  {' · '.join(g.title() for g in glosses)}"
        section(lambda t: card_scene(f"Scene {si + 1}  ·  {tag}", sub, t), 2.2)
        state = HudState(clock="17:42", tag=tag if bad else None)
        k_scene = 0
        with LandmarkExtractor() as ex:
            for g in glosses:
                cands = [m for vid, m in man.items() if vid in test_ids and m["signer_id"] == signer and m["gloss"] == g]
                if not cands:
                    print(f"skip {g}: no test clip for signer {signer}")
                    continue
                m = cands[0]
                frames = [f for f, _ in read_frames(m["path"], m["frame_start"], m["frame_end"], target_fps=None)]
                src_fps = cv2.VideoCapture(m["path"]).get(cv2.CAP_PROP_FPS) or 30
                stride = max(1, round(src_fps / 15))  # classifier sees ~15 fps, as in predict()
                buf = []
                for k, f in enumerate(frames):
                    if bad:
                        f = degrade(f, rng, k_scene)
                    view = compose_pov(f)
                    t0 = time.perf_counter()
                    lm = ex.process_frame(view, int(1000 / src_fps))
                    state.frame_ms = 0.8 * state.frame_ms + 0.2 * (time.perf_counter() - t0) * 1000 if state.frame_ms else (time.perf_counter() - t0) * 1000
                    if k % stride == 0:
                        buf.append(lm)
                    state.landmarks = lm
                    state.signing = has_hand(lm)
                    if k % 3 == 0 and len(buf) >= 6:
                        words, probs, model_ms, _ = pred.classify_landmarks(np.stack(buf), 3)
                        state.hypotheses, state.model_ms = list(zip(words, probs)), model_ms
                    state.t = n_frames / FPS
                    emit(hud.render(view, state))
                    k_scene += 1
                # end of sign: commit the word from the full clip
                words, probs, _, _ = pred.classify_landmarks(np.stack(buf), 3)
                state.hypotheses = list(zip(words, probs))
                state.captions.append((words[0], probs[0], n_frames / FPS))
                speech.append((n_frames / FPS, words[0].lower()))
                print(f"scene {si + 1}: {g:8s} -> {words[0]:8s} ({probs[0]:.2f})")
                state.signing = False
                for _ in range(int(1.1 * FPS)):  # hold so the caption animation is readable
                    state.t = n_frames / FPS
                    emit(hud.render(view, state))
        # short fade to black at scene end
        last = hud.render(view, state)
        for i in range(int(0.4 * FPS)):
            emit(fade(last, 1 - (i + 1) / (0.4 * FPS)))

    section(lambda t: card_results(t, results), 6.0)
    section(card_outro, 4.0, fade_out=0.8)
    enc.stdin.close()
    enc.wait()

    if args.voice and shutil.which("say") and speech:
        inputs, filters = [], []
        for i, (ts, word) in enumerate(speech):
            wav = tmp / f"w{i}.aiff"
            subprocess.run(["say", "-o", str(wav), word], check=True)
            inputs += ["-i", str(wav)]
            ms = int(ts * 1000)
            filters.append(f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]")
        mix = "".join(f"[a{i}]" for i in range(len(speech)))
        filters.append(f"{mix}amix=inputs={len(speech)}:normalize=0[aout]")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), *inputs, "-filter_complex",
                        ";".join(filters), "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "128k", str(args.out)], check=True)
    else:
        shutil.copy(silent, args.out)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {args.out}  ({n_frames / FPS:.1f}s)")


if __name__ == "__main__":
    main()
