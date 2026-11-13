"""Per-frame 3D keypoint extraction with MediaPipe Tasks (runs fully on-device).

Every frame becomes a (N_POINTS, 3) float32 array laid out as:
    [0:21]   left hand   (signer's left, assigned via the pose wrists)
    [21:42]  right hand
    [42:55]  upper-body pose subset (POSE_IDX)
Missing points are NaN. x is rescaled by the frame aspect ratio so x and y share units.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe/LiteRT INFO spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

N_HAND = 21
# nose, eyes (inner-ish), ears, shoulders, elbows, wrists, hips
POSE_IDX = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
N_POSE = len(POSE_IDX)
N_POINTS = 2 * N_HAND + N_POSE
LH = slice(0, N_HAND)
RH = slice(N_HAND, 2 * N_HAND)
POSE = slice(2 * N_HAND, N_POINTS)
# indices inside the POSE block
P_NOSE, P_LSH, P_RSH, P_LWR, P_RWR = 0, 5, 6, 9, 10

ASSETS = Path(os.environ.get("ECHOGESTURE_ASSETS", Path(__file__).resolve().parent.parent / "assets"))


class LandmarkExtractor:
    """Wraps MediaPipe HandLandmarker + PoseLandmarker in VIDEO mode (temporal tracking)."""

    def __init__(self, assets_dir: Path | str = ASSETS, min_conf: float = 0.3):
        assets_dir = Path(assets_dir)
        # Low thresholds help in low light / angled signers; tracking smooths jitter.
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(assets_dir / "hand_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf,
                min_tracking_confidence=min_conf,
            )
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(assets_dir / "pose_landmarker_lite.task")),
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=min_conf,
                min_pose_presence_confidence=min_conf,
                min_tracking_confidence=min_conf,
            )
        )
        self._ts = 0  # MediaPipe needs strictly increasing timestamps across the instance lifetime
        # hand and pose graphs run concurrently (native code releases the GIL): ~max(hand, pose) per frame
        self._pool = ThreadPoolExecutor(max_workers=1)

    def close(self):
        self._pool.shutdown()
        self.hands.close()
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def process_frame(self, bgr: np.ndarray, dt_ms: int = 33) -> np.ndarray:
        h, w = bgr.shape[:2]
        aspect = w / h
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts += max(1, int(dt_ms))
        pose_job = self._pool.submit(self.pose.detect_for_video, image, self._ts)
        hres = self.hands.detect_for_video(image, self._ts)
        pres = pose_job.result()

        out = np.full((N_POINTS, 3), np.nan, dtype=np.float32)
        if pres.pose_landmarks:
            lm = pres.pose_landmarks[0]
            out[POSE] = [[lm[i].x * aspect, lm[i].y, lm[i].z * aspect] for i in POSE_IDX]

        hands = []
        for i, hl in enumerate(hres.hand_landmarks):
            arr = np.array([[p.x * aspect, p.y, p.z * aspect] for p in hl], dtype=np.float32)
            label = hres.handedness[i][0].category_name if hres.handedness else ""
            hands.append((arr, label))
        self._assign_hands(out, hands)
        return out

    @staticmethod
    def _assign_hands(out: np.ndarray, hands: list[tuple[np.ndarray, str]]):
        """Assign detected hands to the signer's left/right using the pose wrists when available.

        This is camera-orientation agnostic (works for mirrored selfie view and third-person view),
        unlike MediaPipe's handedness label which assumes a mirrored image.
        """
        if not hands:
            return
        lw, rw = out[POSE][P_LWR, :2], out[POSE][P_RWR, :2]
        if not (np.isnan(lw).any() or np.isnan(rw).any()):
            if len(hands) == 1:
                d_l = np.linalg.norm(hands[0][0][0, :2] - lw)
                d_r = np.linalg.norm(hands[0][0][0, :2] - rw)
                out[LH if d_l < d_r else RH] = hands[0][0]
                return
            a, b = hands[0][0], hands[1][0]
            cost_ab = np.linalg.norm(a[0, :2] - lw) + np.linalg.norm(b[0, :2] - rw)
            cost_ba = np.linalg.norm(b[0, :2] - lw) + np.linalg.norm(a[0, :2] - rw)
            out[LH], out[RH] = (a, b) if cost_ab <= cost_ba else (b, a)
            return
        # fallback: MediaPipe handedness labels (mirrored convention -> swap)
        for arr, label in hands[:2]:
            out[RH if label == "Left" else LH] = arr


def read_frames(video_path: str | Path, frame_start: int = 1, frame_end: int = -1, target_fps: float = 15.0):
    """Yield (bgr_frame, dt_ms) subsampled to ~target_fps. frame_start/frame_end are 1-based, -1 = until end."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    stride = max(1, round(fps / target_fps)) if target_fps else 1
    dt_ms = int(1000 * stride / fps)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx < frame_start:
                continue
            if frame_end > 0 and idx > frame_end:
                break
            if (idx - frame_start) % stride == 0:
                yield frame, dt_ms
    finally:
        cap.release()


def extract_video(
    video_path: str | Path,
    extractor: LandmarkExtractor | None = None,
    frame_start: int = 1,
    frame_end: int = -1,
    target_fps: float = 15.0,
    frame_transform=None,
) -> np.ndarray:
    """Return (T, N_POINTS, 3) landmarks for a clip. `frame_transform` lets benchmarks inject degradations."""
    own = extractor is None
    extractor = extractor or LandmarkExtractor()
    try:
        seq = []
        for frame, dt in read_frames(video_path, frame_start, frame_end, target_fps):
            if frame_transform is not None:
                frame = frame_transform(frame)
            seq.append(extractor.process_frame(frame, dt))
    finally:
        if own:
            extractor.close()
    if not seq:
        return np.full((1, N_POINTS, 3), np.nan, dtype=np.float32)
    return np.stack(seq).astype(np.float32)
