# utils/drawing.py
import time
import cv2
import numpy as np
from typing import List

from core.tracker import TrackedFace

BOX_THICKNESS = 2
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.55
LABEL_THICKNESS = 1
FPS_COLOR = (0, 200, 255)
FPS_POSITION = (15, 35)

COLORS = [
    (0, 255, 0),   (255, 100, 0), (0, 100, 255), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (100, 255, 100),(255, 100, 100),
    (100, 100, 255),(200, 200, 0),(0, 200, 200),  (200, 0, 200),
]


def get_color(track_id: int):
    return COLORS[track_id % len(COLORS)]


class FPSCounter:
    def __init__(self, window: int = 30):
        self._window = window
        self._timestamps: list = []

    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def draw_tracked_faces(
    frame: np.ndarray,
    tracked_faces: List[TrackedFace],
    show_confidence: bool = True,
) -> np.ndarray:
    for tf in tracked_faces:
        x1, y1, x2, y2 = tf.bbox
        color = get_color(tf.track_id)
        thickness = 3 if tf.is_new else BOX_THICKNESS
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"ID:{tf.track_id}"
        if show_confidence:
            label += f" {tf.confidence:.2f}"

        (lw, lh), baseline = cv2.getTextSize(
            label, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS
        )
        cv2.rectangle(
            frame,
            (x1, y1 - lh - baseline - 4),
            (x1 + lw + 4, y1),
            color, cv2.FILLED,
        )
        cv2.putText(
            frame, label, (x1 + 2, y1 - baseline - 2),
            LABEL_FONT, LABEL_SCALE, (0, 0, 0),
            LABEL_THICKNESS, cv2.LINE_AA,
        )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(
        frame, f"FPS: {fps:.1f}", FPS_POSITION,
        LABEL_FONT, 0.8, FPS_COLOR, 2, cv2.LINE_AA,
    )
    return frame


def draw_info_bar(
    frame: np.ndarray,
    frame_idx: int,
    face_count: int,
) -> np.ndarray:
    h = frame.shape[0]
    text = f"Frame: {frame_idx}  |  Faces: {face_count}"
    cv2.putText(
        frame, text, (15, h - 15),
        LABEL_FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return frame