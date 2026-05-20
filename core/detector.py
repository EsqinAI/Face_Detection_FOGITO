# core/detector.py
# RetinaFace face detector.
# ResNet50 backbone — dəqiq, multi-face, Jetson TensorRT ilə sürətləndirilə bilər.

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

import torch
from retinaface import RetinaFace as RF

from config import DetectorConfig

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: List[int]           # [x1, y1, x2, y2]
    confidence: float
    landmarks: Optional[np.ndarray] = None   # 5 facial landmark

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self):
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2,
        )


class FaceDetector:
    """
    RetinaFace detector.
    ResNet50: dəqiq, amma ağır
    MobileNet: yüngül, Jetson edge üçün
    """

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._model = None

    def load(self) -> None:
        logger.info(f"RetinaFace yüklənir | backbone: {self.cfg.backbone}")

        self._model = RF(
            network=self.cfg.backbone,
            gpu_id=0 if self.cfg.device == "cuda" else -1,
        )

        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.detect(dummy, threshold=self.cfg.confidence)

        logger.info("RetinaFace hazırdır.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._model is None:
            raise RuntimeError("Əvvəlcə .load() çağırılmalıdır.")

        h, w = frame.shape[:2]

        try:
            faces = self._model.detect(
                frame,
                threshold=self.cfg.confidence,
            )
        except Exception as e:
            logger.warning(f"Detection xətası: {e}")
            return []

        detections = []

        if faces is None:
            return detections

        for face in faces:
            # RetinaFace output: [x1, y1, x2, y2, confidence]
            x1, y1, x2, y2, conf = (
                int(face[0]), int(face[1]),
                int(face[2]), int(face[3]),
                float(face[4]),
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            # Aspect ratio filtri
            aspect = (y2 - y1) / (x2 - x1) if (x2 - x1) > 0 else 0
            if not (0.6 <= aspect <= 1.6):
                continue

            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                confidence=conf,
            ))

        return detections