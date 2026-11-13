"""Glass-style AR caption HUD, rendered from the wearer's (exocentric) point of view.

Design language: frosted-glass rounded cards, one accent colour, large high-contrast type and
soft motion, built so a caption can be read at a glance through smart glasses.

    hud = GlassHUD(1280, 720)
    out_bgr = hud.render(frame_bgr, HudState(...))
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .landmarks import ASSETS, LH, N_HAND, P_LSH, P_NOSE, P_RSH, POSE, RH

FONT_PATH = Path(ASSETS) / "fonts" / "Inter.ttf"

ACCENT = (62, 145, 255)
GREEN = (52, 199, 89)
AMBER = (255, 176, 32)
WHITE = (255, 255, 255)
SUB = (196, 202, 214)
GLASS = (18, 20, 26)

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


@dataclass
class HudState:
    landmarks: np.ndarray | None = None          # (55, 3) current frame keypoints (x scaled by aspect)
    captions: list[tuple[str, float, float]] = field(default_factory=list)  # (word, confidence, t_committed)
    hypotheses: list[tuple[str, float]] = field(default_factory=list)       # live top-3
    signing: bool = False
    t: float = 0.0                                # clock used for animation (seconds)
    frame_ms: float = 0.0
    model_ms: float = 0.0
    tag: str | None = None                        # e.g. "Low light · occlusion" condition label
    clock: str | None = None                      # status-bar time text
    show_skeleton: bool = True


# ----------------------------------------------------------------------------- drawing primitives

@lru_cache(maxsize=64)
def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    try:
        f = ImageFont.truetype(str(FONT_PATH), size)
        f.set_variation_by_name(weight)
        return f
    except Exception:  # bundled font missing: fall back to PIL's default
        return ImageFont.load_default(size)


@lru_cache(maxsize=256)
def rr_mask(w: int, h: int, r: int) -> Image.Image:
    """Anti-aliased rounded-rectangle mask (4x supersampled)."""
    ss = 4
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=r * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def text_w(txt: str, f: ImageFont.FreeTypeFont) -> int:
    return int(f.getlength(txt))


class GlassHUD:
    UNSURE = 0.35  # below this confidence a caption is shown as a tentative guess (amber, with "?")

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.s = min(width / 1280, height / 720)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        d = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
        vig = np.clip(1.08 - 0.32 * d ** 2.2, 0.55, 1.0)
        lens = np.array(rr_mask(width, height, int(36 * self.s)), dtype=np.float32) / 255  # glasses viewport
        self.shade = (vig * lens)[..., None]
        self._box = None   # smoothed signer box
        self._anchor = None

    # ------------------------------------------------------------------ glass card
    def card(self, base: Image.Image, blurred: Image.Image, box, r, alpha=150, border=True):
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.w, x1), min(self.h, y1)
        w, h = x1 - x0, y1 - y0
        if w < 4 or h < 4:
            return
        m = rr_mask(w, h, min(r, h // 2, w // 2))
        base.paste(blurred.crop((x0, y0, x1, y1)), (x0, y0), m)
        base.paste(Image.new("RGBA", (w, h), GLASS + (255,)), (x0, y0), m.point(lambda v: v * alpha // 255))
        if border:
            ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(ov).rounded_rectangle((0, 0, w - 1, h - 1), radius=min(r, h // 2, w // 2),
                                                 outline=(255, 255, 255, 46), width=1)
            base.alpha_composite(ov, (x0, y0))

    def pill(self, base, blurred, x, y, parts, h=None, pad=None, align="left"):
        """parts: list of (text, font, color) or ('dot', color). Returns (x0, x1)."""
        s = self.s
        h = h or int(38 * s)
        pad = pad or int(16 * s)
        gap = int(8 * s)
        widths = [int(10 * s) if p[0] == "dot" else text_w(p[0], p[1]) for p in parts]
        w = sum(widths) + gap * (len(parts) - 1) + 2 * pad
        x0 = x - w if align == "right" else (x - w // 2 if align == "center" else x)
        self.card(base, blurred, (x0, y, x0 + w, y + h), h // 2)
        d = ImageDraw.Draw(base)
        cx = x0 + pad
        for p, pw in zip(parts, widths):
            if p[0] == "dot":
                r = int(4.5 * s)
                d.ellipse((cx, y + h / 2 - r, cx + 2 * r, y + h / 2 + r), fill=p[1])
            else:
                d.text((cx, y + h / 2), p[0], font=p[1], fill=p[2], anchor="lm")
            cx += pw + gap
        return x0, x0 + w

    # ------------------------------------------------------------------ main render
    def render(self, frame_bgr: np.ndarray, st: HudState) -> np.ndarray:
        s, W, H = self.s, self.w, self.h
        if frame_bgr.shape[1] != W or frame_bgr.shape[0] != H:
            frame_bgr = cv2.resize(frame_bgr, (W, H))
        shaded = (frame_bgr.astype(np.float32) * self.shade).astype(np.uint8)
        small = cv2.resize(shaded, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        blurred_np = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.2), (W, H), interpolation=cv2.INTER_LINEAR)
        base = Image.fromarray(cv2.cvtColor(shaded, cv2.COLOR_BGR2RGB)).convert("RGBA")
        blurred = Image.fromarray(cv2.cvtColor(blurred_np, cv2.COLOR_BGR2RGB)).convert("RGBA")

        pts = self._to_px(st.landmarks) if st.landmarks is not None else None
        if pts is not None and st.show_skeleton:
            self._skeleton(base, pts)
        self._signer_frame(base, blurred, pts, st)
        self._status_bar(base, blurred, st)
        self._caption_bubble(base, blurred, pts, st)
        self._dock(base, blurred, st)
        return cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)

    def _to_px(self, lm: np.ndarray) -> np.ndarray:
        aspect = self.w / self.h
        out = lm[..., :2].copy()
        out[:, 0] = out[:, 0] / aspect * self.w
        out[:, 1] = out[:, 1] * self.h
        return out

    # ------------------------------------------------------------------ components
    def _skeleton(self, base, pts):
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        lw = max(2, int(2.5 * self.s))
        for sl in (LH, RH):
            hp = pts[sl]
            if np.isnan(hp[:, 0]).all():
                continue
            for a, b in HAND_EDGES:
                if np.isfinite(hp[[a, b]]).all():
                    d.line((*hp[a], *hp[b]), fill=ACCENT + (190,), width=lw)
            r = 2.6 * self.s
            for p in hp:
                if np.isfinite(p).all():
                    d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(255, 255, 255, 230))
        base.alpha_composite(ov)

    def _signer_box(self, pts):
        if pts is None:
            return None
        pose = pts[POSE]
        cand = [pose[i] for i in (P_NOSE, P_LSH, P_RSH)] + list(pts[: 2 * N_HAND])
        cand = np.array([p for p in cand if np.isfinite(p).all()])
        if len(cand) < 3 or not np.isfinite(pose[[P_LSH, P_RSH]]).all():
            return None
        sw = abs(pose[P_LSH, 0] - pose[P_RSH, 0])
        x0, y0 = cand.min(0)
        x1, y1 = cand.max(0)
        x0, x1 = min(x0, pose[P_RSH, 0] - 0.5 * sw, pose[P_LSH, 0] - 0.5 * sw), max(x1, pose[P_RSH, 0] + 0.5 * sw, pose[P_LSH, 0] + 0.5 * sw)
        y0 = min(y0, pose[P_NOSE, 1] - 0.55 * sw) if np.isfinite(pose[P_NOSE]).all() else y0
        y1 = max(y1, max(pose[P_LSH, 1], pose[P_RSH, 1]) + 0.9 * sw)
        s = self.s
        return np.array([max(x0, 16 * s), max(y0, 112 * s), min(x1, self.w - 16 * s), min(y1, self.h - 130 * s)])

    def _signer_frame(self, base, blurred, pts, st):
        box = self._signer_box(pts)
        if box is not None:
            self._box = box if self._box is None else 0.75 * self._box + 0.25 * box
        if self._box is None:
            return
        x0, y0, x1, y1 = self._box
        s = self.s
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        L, lw = int(30 * s), max(2, int(3 * s))
        col = (255, 255, 255, 215)
        for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            d.line((cx, cy, cx + dx * L, cy), fill=col, width=lw)
            d.line((cx, cy, cx, cy + dy * L), fill=col, width=lw)
        base.alpha_composite(ov)
        label = "Signing" if st.signing else "Signer in view"
        on_left = getattr(self, "_side", None) == "left"  # keep the label clear of the caption bubble
        self.pill(base, blurred, int(x1 if on_left else x0), int(y0 - 44 * s),
                  [("dot", ACCENT if st.signing else GREEN), (label, font(int(15 * s), "Medium"), WHITE)],
                  h=int(32 * s), pad=int(12 * s), align="right" if on_left else "left")

    def _status_bar(self, base, blurred, st):
        s = self.s
        y = int(22 * s)
        clock = st.clock or time.strftime("%H:%M")
        self.pill(base, blurred, int(26 * s), y,
                  [(clock, font(int(17 * s), "SemiBold"), WHITE), ("EchoGesture", font(int(15 * s), "Regular"), SUB)])
        x = W = self.w - int(26 * s)
        x0, _ = self.pill(base, blurred, x, y, [(f"{st.frame_ms:.0f} ms", font(int(15 * s), "Medium"), WHITE)],
                          align="right")
        x0, _ = self.pill(base, blurred, x0 - int(10 * s), y, [("ASL  →  EN", font(int(15 * s), "Medium"), WHITE)],
                          align="right")
        self.pill(base, blurred, x0 - int(10 * s), y,
                  [("dot", GREEN), ("On-device", font(int(15 * s), "Medium"), WHITE)], align="right")
        if st.tag:
            self.pill(base, blurred, W // 2 + int(13 * s), y,
                      [("dot", AMBER), (st.tag, font(int(15 * s), "Medium"), WHITE)], align="center")

    def _place_bubble(self, pts, bw, bh):
        """Float the caption beside the signer's head without covering the face or either hand.

        Candidates: right of head, left of head, lower third. The previous side is kept while it stays
        free (hysteresis), and the position is smoothed, so the bubble does not jump between frames.
        """
        s = self.s
        lower = ((self.w - bw) / 2, self.h - bh - 118 * s, None)
        pose = pts[POSE] if pts is not None else None
        if pose is not None and np.isfinite(pose[[P_NOSE, P_LSH, P_RSH]]).all():
            head = pose[P_NOSE]
            self._anchor = head if self._anchor is None else 0.8 * self._anchor + 0.2 * head
            self._sw = abs(pose[P_LSH, 0] - pose[P_RSH, 0])
        if self._anchor is None:
            return lower
        hx, hy = self._anchor
        sw = getattr(self, "_sw", 200 * s)
        by = float(np.clip(hy - bh * 0.45, 84 * s, self.h - bh - 120 * s))
        face = (hx - 0.45 * sw, hy - 0.6 * sw, hx + 0.45 * sw, hy + 0.5 * sw)
        hands = pts[: 2 * N_HAND] if pts is not None else np.zeros((0, 2))
        hands = hands[np.isfinite(hands).all(1)]

        def free(x0):
            x1, y1 = x0 + bw, by + bh
            if x0 < 16 * s or x1 > self.w - 16 * s:
                return False
            if not (x1 < face[0] or x0 > face[2] or y1 < face[1] or by > face[3]):
                return False
            pad = 18 * s
            return not ((hands[:, 0] > x0 - pad) & (hands[:, 0] < x1 + pad) &
                        (hands[:, 1] > by - pad) & (hands[:, 1] < y1 + pad)).any()

        cands = {"right": hx + 0.75 * sw, "left": hx - 0.75 * sw - bw}
        prev = getattr(self, "_side", None)
        order = [prev] + [k for k in cands if k != prev] if prev in cands else list(cands)
        for side in order:
            if free(cands[side]):
                self._side = side
                target = np.array([cands[side], by])
                break
        else:
            # keep the last free spot briefly before giving up (hands sweep through quickly)
            self._blocked = getattr(self, "_blocked", 0) + 1
            if prev in cands and self._blocked < 8:
                target = np.array([cands[prev], by])
                side = prev
            else:
                self._side = None
                return lower
        self._blocked = 0
        self._bpos = target if getattr(self, "_bpos", None) is None else 0.7 * self._bpos + 0.3 * target
        return self._bpos[0], self._bpos[1], side

    def _caption_bubble(self, base, blurred, pts, st):
        s = self.s
        bw, bh = int(370 * s), int(178 * s)
        bx, by, side = self._place_bubble(pts, bw, bh)
        bx, by = int(bx), int(by)

        last = st.captions[-1] if st.captions else None
        e = ease_out((st.t - last[2]) / 0.35) if last else 1.0
        lift = int((1 - e) * 14 * s)
        self.card(base, blurred, (bx, by + lift, bx + bw, by + bh + lift), int(26 * s), alpha=165)
        if side is not None:  # tail pointing at the signer
            tail = Image.new("RGBA", base.size, (0, 0, 0, 0))
            ty = by + lift + int(48 * s)
            tx = bx if side == "right" else bx + bw
            dx = -int(12 * s) if side == "right" else int(12 * s)
            ImageDraw.Draw(tail).polygon([(tx, ty - 10 * s), (tx + dx, ty), (tx, ty + 10 * s)], fill=GLASS + (200,))
            base.alpha_composite(tail)
        d = ImageDraw.Draw(base)

        px, py = bx + int(24 * s), by + lift + int(22 * s)
        d.ellipse((px, py + 4 * s, px + 9 * s, py + 13 * s), fill=ACCENT)
        d.text((px + int(16 * s), py), "Live caption", font=font(int(14 * s), "Medium"), fill=SUB)
        d.text((bx + bw - int(24 * s), py), "ASL", font=font(int(14 * s), "SemiBold"), fill=ACCENT, anchor="ra")

        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        do = ImageDraw.Draw(ov)
        if last:
            unsure = last[1] < self.UNSURE
            word = last[0].capitalize() + ("?" if unsure else "")
            meter = AMBER if unsure else ACCENT
            fsz = int(46 * s)
            while text_w(word, font(fsz, "Bold")) > bw - 48 * s and fsz > 20:
                fsz -= 2
            do.text((px, py + int(30 * s) + lift), word, font=font(fsz, "Bold"), fill=WHITE + (int(255 * e),))
            # confidence meter
            my = py + int(92 * s) + lift
            mw = int(120 * s)
            do.rounded_rectangle((px, my, px + mw, my + 5 * s), radius=3, fill=(255, 255, 255, 50))
            do.rounded_rectangle((px, my, px + max(6, int(mw * last[1] * e)), my + 5 * s), radius=3,
                                 fill=meter + (255,))
            note = f"Not sure · {last[1] * 100:.0f}%" if unsure else f"{last[1] * 100:.0f}% confident"
            do.text((px + mw + 10 * s, my + 2.5 * s), note,
                    font=font(int(13 * s), "Medium" if unsure else "Regular"),
                    fill=(AMBER if unsure else SUB) + (255,), anchor="lm")
            sentence = " ".join(w.lower() + ("?" if c < self.UNSURE else "") for w, c, _ in st.captions[-6:])
            sentence = sentence[0].upper() + sentence[1:]
            f = font(int(18 * s), "Regular")
            while text_w(sentence, f) > bw - 48 * s and " " in sentence:
                sentence = "… " + sentence.split(" ", 2)[-1] if sentence.startswith("…") else "… " + sentence.split(" ", 1)[1]
            do.text((px, py + int(112 * s) + lift), sentence, font=f, fill=WHITE + (215,))
        else:
            do.text((px, py + int(36 * s)), "Listening for signs…", font=font(int(30 * s), "SemiBold"),
                    fill=(255, 255, 255, 150))
            do.text((px, py + int(84 * s)), "Captions appear here as the signer signs.",
                    font=font(int(15 * s), "Regular"), fill=SUB + (200,))
        base.alpha_composite(ov)

    def _dock(self, base, blurred, st):
        s = self.s
        h = int(64 * s)
        chips = st.hypotheses[:3]
        fchip = font(int(15 * s), "SemiBold")
        chip_ws = [text_w(f"{w.capitalize()}  {p * 100:.0f}%", fchip) + int(28 * s) for w, p in chips]
        label = "Recognizing" if st.signing else ("Ready" if not chips else "Last result")
        flab = font(int(16 * s), "SemiBold")
        wave_w = int(46 * s)
        w = int(24 * s) + wave_w + int(14 * s) + text_w(label, flab) + int(20 * s) + sum(chip_ws) + \
            int(8 * s) * max(0, len(chips) - 1) + int(20 * s)
        w = max(w, int(300 * s))
        x0 = (self.w - w) // 2
        y0 = self.h - h - int(28 * s)
        self.card(base, blurred, (x0, y0, x0 + w, y0 + h), h // 2, alpha=170)
        d = ImageDraw.Draw(base)
        # animated voice-style waveform: alive while signing, flat when idle
        cx, cy = x0 + int(24 * s), y0 + h // 2
        for i in range(5):
            amp = (0.35 + 0.65 * abs(math.sin(st.t * 7 + i * 1.3))) if st.signing else 0.18
            bh = int(30 * s * amp)
            bx = cx + i * int(10 * s)
            d.rounded_rectangle((bx, cy - bh / 2, bx + int(5 * s), cy + bh / 2), radius=int(3 * s),
                                fill=ACCENT if st.signing else (150, 156, 170))
        tx = cx + wave_w + int(14 * s)
        d.text((tx, cy), label, font=flab, fill=WHITE, anchor="lm")
        cx = tx + text_w(label, flab) + int(20 * s)
        for k, ((word, p), cw) in enumerate(zip(chips, chip_ws)):
            ch = int(36 * s)
            box = (cx, cy - ch / 2, cx + cw, cy + ch / 2)
            if k == 0:
                d.rounded_rectangle(box, radius=ch // 2, fill=ACCENT)
            else:
                d.rounded_rectangle(box, radius=ch // 2, outline=(112, 118, 132), width=max(1, int(1.5 * s)))
            d.text((cx + cw / 2, cy), f"{word.capitalize()}  {p * 100:.0f}%", font=fchip,
                   fill=WHITE if k == 0 else SUB, anchor="mm")
            cx += cw + int(8 * s)
