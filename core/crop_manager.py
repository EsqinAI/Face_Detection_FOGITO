# core/crop_manager.py
import logging
import os
from datetime import datetime

import cv2
import numpy as np

from config import CropConfig
from core.tracker import TrackedFace
from core.anti_spoof import AntiSpoof

logger = logging.getLogger(__name__)


class CropManager:
    def __init__(self, cfg: CropConfig, anti_spoof: AntiSpoof):
        self.cfg = cfg
        self.anti_spoof = anti_spoof
        self._saved_ids: set = set()
        os.makedirs(cfg.save_dir, exist_ok=True)
        logger.info(f"CropManager hazırdır → {os.path.abspath(cfg.save_dir)}")

    def process(
        self,
        tracked_faces: List[TrackedFace],
        frame: np.ndarray,
        depth_map: np.ndarray,
    ) -> None:
        for tf in tracked_faces:
            if not tf.is_new:
                continue
            if tf.track_id in self._saved_ids:
                continue

            # Depth-based anti-spoof
            if not self.anti_spoof.is_real(depth_map, tf.bbox):
                logger.info(f"FAKE atlandı → track_id={tf.track_id}")
                continue

            crop = self._crop_face(frame, tf)
            if crop is None:
                continue

            # Blur yoxlaması
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
                logger.debug(f"Blur atlandı → track_id={tf.track_id}")
                continue

            path = self._save(crop, tf.track_id)
            self._saved_ids.add(tf.track_id)
            logger.info(
                f"[SERVER] track_id={tf.track_id} | "
                f"conf={tf.confidence:.2f} | "
                f"crop={os.path.basename(path)} | "
                f"göndərildi ✓"
            )

    def _crop_face(self, frame: np.ndarray, tf: TrackedFace):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = tf.bbox

        x1 = max(0, x1 - self.cfg.padding)
        y1 = max(0, y1 - self.cfg.padding)
        x2 = min(w, x2 + self.cfg.padding)
        y2 = min(h, y2 + self.cfg.padding)

        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < self.cfg.min_size or crop.shape[1] < self.cfg.min_size:
            return None
        return crop

    def _save(self, crop: np.ndarray, track_id: int) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"face_{track_id:04d}_{timestamp}.jpg"
        path = os.path.join(self.cfg.save_dir, filename)
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        return path

    @property
    def saved_count(self) -> int:
        return len(self._saved_ids)