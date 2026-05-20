# core/tracker.py
# BoT-SORT tracker.
# CMC (Camera Motion Compensation) — hərəkət edən kamera üçün.

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict

from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace

from config import TrackerConfig
from core.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class TrackedFace:
    bbox: List[int]
    confidence: float
    track_id: int
    is_new: bool = False


class FaceTracker:
    """
    BoT-SORT tracker.
    - CMC: hərəkət edən kamera kompensasiyası
    - ReID: appearance feature ilə daha stabil tracking
    """

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self._tracker = None
        self._confirmed_ids: set = set()
        self._active_ids: set = set()

    def load(self) -> None:
        args = IterableSimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=self.cfg.track_high_thresh,
            track_low_thresh=self.cfg.track_low_thresh,
            new_track_thresh=self.cfg.new_track_thresh,
            track_buffer=self.cfg.track_buffer,
            match_thresh=self.cfg.match_thresh,
            proximity_thresh=self.cfg.proximity_thresh,
            appearance_thresh=self.cfg.appearance_thresh,
            with_reid=self.cfg.with_reid,
            fuse_score=True,
            # CMC — hərəkət edən kamera
            cmc_method=self.cfg.cmc_method,
            gmc_method=self.cfg.cmc_method,
        )

        self._tracker = BOTSORT(args, frame_rate=30)
        logger.info(f"BoT-SORT hazırdır | CMC: {self.cfg.cmc_method}")

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_idx: int,
    ) -> List[TrackedFace]:
        if self._tracker is None:
            raise RuntimeError("Əvvəlcə .load() çağırılmalıdır.")

        h, w = frame.shape[:2]

        if detections:
            dets = np.array([
                [*d.bbox, d.confidence, 0]
                for d in detections
            ], dtype=np.float32)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        # BoT-SORT update — frame də lazımdır (CMC + ReID üçün)
        tracks = self._tracker.update(dets, frame)

        tracked_faces: List[TrackedFace] = []
        current_ids = set()

        for track in tracks:
            x1, y1, x2, y2 = track.tlbr.astype(int)
            track_id = int(track.track_id)
            conf = float(track.score)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            is_new = track_id not in self._confirmed_ids
            if is_new:
                self._confirmed_ids.add(track_id)
                logger.info(f"Yeni üz təsdiqləndi → track_id={track_id}")

            current_ids.add(track_id)
            tracked_faces.append(TrackedFace(
                bbox=[x1, y1, x2, y2],
                confidence=conf,
                track_id=track_id,
                is_new=is_new,
            ))

        lost_ids = self._active_ids - current_ids
        for lid in lost_ids:
            logger.debug(f"Track itdi → track_id={lid}")

        self._active_ids = current_ids
        return tracked_faces

    def should_detect(self, frame_idx: int) -> bool:
        return True  # RetinaFace hər frame-də işləyir