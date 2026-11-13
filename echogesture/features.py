"""Pose-invariant feature construction + training augmentations, all on raw (T, N_POINTS, 3) landmarks.

Invariances baked in:
  * translation / scale: body frame centered on the shoulder midpoint, scaled by shoulder width
  * in-plane roll: shoulder line rotated to horizontal (tilted heads / angled phones)
  * hand shape: each hand also expressed in its own wrist frame, scaled by palm size
  * occlusion: missing points are zero-filled and flagged with per-hand presence bits
"""
from __future__ import annotations

import numpy as np

from .landmarks import LH, N_HAND, N_POINTS, P_LSH, P_RSH, POSE, RH

SEQ_LEN = 32
FEAT_DIM = N_POINTS * 3 * 2 + 2 * N_HAND * 3 + 2  # global xyz + velocity, hand-local xyz, 2 presence flags

# pose block indices to swap when mirroring (left<->right): eyes, ears, shoulders, elbows, wrists, hips
_POSE_MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]


def _hand_present(seq: np.ndarray, sl: slice) -> np.ndarray:
    return ~np.isnan(seq[:, sl, 0]).all(axis=1)


def trim_and_resample(seq: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Drop leading/trailing frames without hands, then linearly resample to a fixed length."""
    present = _hand_present(seq, LH) | _hand_present(seq, RH)
    if present.any():
        idx = np.where(present)[0]
        lo, hi = max(0, idx[0] - 1), min(len(seq), idx[-1] + 2)
        seq = seq[lo:hi]
    t = len(seq)
    if t == length:
        return seq.copy()
    pos = np.linspace(0, t - 1, length)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (pos - lo)[:, None, None].astype(np.float32)
    a, b = seq[lo], seq[hi]
    out = a * (1 - w) + b * w
    # where one neighbour is NaN, fall back to the nearest frame instead of propagating NaN
    nearest = np.where(w < 0.5, a, b)
    return np.where(np.isnan(out), nearest, out).astype(np.float32)


def normalize(seq: np.ndarray) -> np.ndarray:
    """Body-centric normalization. Returns a new array, NaNs preserved."""
    seq = seq.astype(np.float32, copy=True)
    pose = seq[:, POSE]
    ls, rs = pose[:, P_LSH, :2], pose[:, P_RSH, :2]
    mid = np.nanmean((ls + rs) / 2, axis=0) if not np.isnan(ls + rs).all() else None
    if mid is not None and not np.isnan(mid).any():
        width = np.nanmedian(np.linalg.norm(ls - rs, axis=1))
        vec = np.nanmean(rs - ls, axis=0)  # from signer-left shoulder to signer-right shoulder
        angle = np.arctan2(vec[1], vec[0])
    else:  # no body visible: fall back to point-cloud statistics
        pts = seq[..., :2].reshape(-1, 2)
        mid = np.nanmean(pts, axis=0) if not np.isnan(pts).all() else np.zeros(2, np.float32)
        width = np.nanstd(pts) * 4 if not np.isnan(pts).all() else 1.0
        angle = 0.0
    if not np.isfinite(width) or width < 1e-4:
        width = 1.0
    # rotate the shoulder line so it is horizontal; keep orientation (do not flip direction)
    ref = np.pi if abs(angle) > np.pi / 2 else 0.0
    c, s = np.cos(ref - angle), np.sin(ref - angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    seq[..., :2] = (seq[..., :2] - mid) @ rot.T / width
    seq[..., 2] = seq[..., 2] / width
    return seq


def _hand_local(hand: np.ndarray) -> np.ndarray:
    """(T, 21, 3) -> wrist-centred, palm-size-normalised hand shape."""
    wrist = hand[:, :1]
    size = np.linalg.norm(hand[:, 9, :2] - hand[:, 0, :2], axis=1)  # wrist -> middle MCP
    size = np.where(np.isfinite(size) & (size > 1e-4), size, 1.0)[:, None, None]
    return (hand - wrist) / size


def build_features(raw: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """raw (T, N_POINTS, 3) with NaNs -> (length, FEAT_DIM) float32, no NaNs."""
    seq = normalize(trim_and_resample(raw, length))
    flags = np.stack([_hand_present(seq, LH), _hand_present(seq, RH)], axis=1).astype(np.float32)
    local = np.concatenate([_hand_local(seq[:, LH]), _hand_local(seq[:, RH])], axis=1)
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    feats = np.concatenate(
        [seq.reshape(length, -1), vel.reshape(length, -1), local.reshape(length, -1), flags], axis=1
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(feats, -10, 10).astype(np.float32)


# ----------------------------------------------------------------------------- augmentation

def mirror(seq: np.ndarray) -> np.ndarray:
    """Left-handed signer simulation: flip x and swap left/right body parts."""
    out = seq.copy()
    out[..., 0] = -out[..., 0]
    left = out[:, LH].copy()
    out[:, LH] = out[:, RH]
    out[:, RH] = left
    out[:, POSE] = out[:, POSE][:, _POSE_MIRROR]
    return out


def augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augment raw landmarks (before normalization) to mimic exocentric capture conditions."""
    seq = seq.astype(np.float32, copy=True)
    t = len(seq)
    # temporal crop + speed change (fast conversational cadence)
    if t > 8 and rng.random() < 0.8:
        keep = rng.uniform(0.75, 1.0)
        n = max(4, int(t * keep))
        start = rng.integers(0, t - n + 1)
        seq = seq[start : start + n]
    if rng.random() < 0.5:
        seq = mirror(seq)
    # camera geometry: roll, yaw (foreshortening), scale, shear, shift
    centre = np.nanmean(seq[..., :2].reshape(-1, 2), axis=0)
    if np.isnan(centre).any():
        centre = np.zeros(2, np.float32)
    a = np.deg2rad(rng.uniform(-15, 15))
    yaw = np.cos(np.deg2rad(rng.uniform(-35, 35)))
    sc = rng.uniform(0.8, 1.2)
    sh = rng.uniform(-0.15, 0.15)
    m = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([[yaw, sh], [0, 1]]) * sc
    seq[..., :2] = (seq[..., :2] - centre) @ m.T.astype(np.float32) + centre + rng.normal(0, 0.03, 2)
    seq[..., 2] *= sc
    # landmark jitter (low light makes keypoints noisy)
    seq += rng.normal(0, rng.uniform(0, 0.006), seq.shape).astype(np.float32)
    # occlusion: drop random frames of a hand, or the whole of one hand
    for sl in (LH, RH):
        if rng.random() < 0.3:
            drop = rng.random(len(seq)) < rng.uniform(0.1, 0.4)
            seq[drop, sl] = np.nan
    if rng.random() < 0.05:
        seq[:, LH if rng.random() < 0.5 else RH] = np.nan
    # random finger-tip occlusion (partial hands)
    if rng.random() < 0.2:
        tips = rng.choice([4, 8, 12, 16, 20], size=rng.integers(1, 3), replace=False)
        for sl in (LH, RH):
            seq[:, sl.start + tips] = np.nan
    return seq
