# core/anti_spoof.py
# ZED X depth-based anti-spoofing.
# Real insan → bbox içində depth variasiyası yüksəkdir (3D forma)
# Foto/ekran/billboard → depth uniform-dur (düz sətih)

import logging
import numpy as np
from config import AntiSpoofConfig

logger = logging.getLogger(__name__)


class AntiSpoof:
    """
    ZED X depth map ilə anti-spoofing.
    Əlavə model lazım deyil — ZED X-in öz depth sensoru istifadə olunur.
    """

    def __init__(self, cfg: AntiSpoofConfig):
        self.cfg = cfg

    def load(self) -> None:
        logger.info(
            f"Depth AntiSpoof hazırdır | "
            f"range: {self.cfg.min_depth}-{self.cfg.max_depth}m | "
            f"std_thresh: {self.cfg.depth_std_threshold}"
        )

    def is_real(self, depth_map: np.ndarray, bbox: List[int]) -> bool:
        """
        Depth map əsasında real üz yoxlaması.

        Məntiq:
            1. Bbox mərkəzindəki depth məsafə range-də olmalıdır
            2. Bbox içindəki depth std-i threshold-dan yüksək olmalıdır
               (real üz 3D-dir — burun, alın, yanaqlar fərqli dərinlikdədir)

        Returns:
            True  → real üz
            False → fake (foto, ekran, billboard, çox uzaq)
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        region = depth_map[y1:y2, x1:x2]

        if region.size == 0:
            return True

        # Sıfır/invalid dəyərləri filtr et
        valid = region[(region > 0.1) & (region < 50.0)]

        if valid.size < self.cfg.sample_points:
            logger.debug(f"Kifayət qədər depth nöqtəsi yoxdur → real qəbul edildi")
            return True

        mean_depth = float(np.mean(valid))
        std_depth = float(np.std(valid))

        # 1. Məsafə yoxlaması
        if not (self.cfg.min_depth <= mean_depth <= self.cfg.max_depth):
            logger.info(
                f"Depth range xaricindədir → FAKE | "
                f"depth={mean_depth:.2f}m | "
                f"range={self.cfg.min_depth}-{self.cfg.max_depth}m"
            )
            return False

        # 2. Depth variasiyası yoxlaması
        if std_depth < self.cfg.depth_std_threshold:
            logger.info(
                f"Düz sətih aşkarlandı → FAKE | "
                f"std={std_depth:.3f} < {self.cfg.depth_std_threshold}"
            )
            return False

        logger.debug(
            f"REAL | depth={mean_depth:.2f}m | std={std_depth:.3f}"
        )
        return True