# core/video_reader.py
# ZED X kamera reader.
# RGB frame + Depth map eyni anda qaytarır.

import logging
import numpy as np
from dataclasses import dataclass
from typing import Generator, Tuple, Optional

import pyzed.sl as sl

from config import CameraConfig

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Hər frame üçün data strukturu."""
    frame_idx: int
    rgb: np.ndarray           # BGR, (H, W, 3)
    depth_map: np.ndarray     # float32, (H, W) — metr


class ZEDVideoReader:
    """
    ZED X kamera reader.
    RGB + Depth eyni timestamp-lə qaytarır.
    """

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._cam: Optional[sl.Camera] = None
        self._image = sl.Mat()
        self._depth = sl.Mat()
        self.fps: float = float(cfg.fps)

    def open(self) -> None:
        self._cam = sl.Camera()

        init = sl.InitParameters()

        # Resolution
        res_map = {
            "HD720":  sl.RESOLUTION.HD720,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD2K":   sl.RESOLUTION.HD2K,
        }
        init.camera_resolution = res_map.get(self.cfg.resolution, sl.RESOLUTION.HD1080)
        init.camera_fps = self.cfg.fps

        # Depth
        depth_map = {
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY":     sl.DEPTH_MODE.QUALITY,
            "ULTRA":       sl.DEPTH_MODE.ULTRA,
        }
        init.depth_mode = depth_map.get(self.cfg.depth_mode, sl.DEPTH_MODE.ULTRA)
        init.coordinate_units = sl.UNIT.METER
        init.depth_minimum_distance = self.cfg.depth_min_dist
        init.depth_maximum_distance = self.cfg.depth_max_dist

        status = self._cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED X açıla bilmədi: {status}")

        # Avtomatik exposure + white balance
        self._cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        self._cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)

        logger.info(
            f"ZED X açıldı | "
            f"Resolution: {self.cfg.resolution} | "
            f"FPS: {self.cfg.fps} | "
            f"Depth: {self.cfg.depth_mode}"
        )

    def read_frames(self) -> Generator[FrameData, None, None]:
        """Generator — hər frame-də FrameData qaytarır."""
        if self._cam is None:
            raise RuntimeError("Əvvəlcə .open() çağırılmalıdır.")

        runtime = sl.RuntimeParameters()
        runtime.confidence_threshold = 50     # depth confidence
        runtime.texture_confidence_threshold = 100

        frame_idx = 0

        while True:
            if self._cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                logger.warning("Frame alına bilmədi, keçilir...")
                continue

            # RGB frame — LEFT kamera
            self._cam.retrieve_image(self._image, sl.VIEW.LEFT)
            rgb_data = self._image.get_data()

            # BGRA → BGR
            rgb = rgb_data[:, :, :3].copy()
            rgb = self._resize(rgb)

            # Depth map
            self._cam.retrieve_measure(self._depth, sl.MEASURE.DEPTH)
            depth_raw = self._depth.get_data().copy()

            # NaN/Inf → 0
            depth_raw = np.nan_to_num(depth_raw, nan=0.0, posinf=0.0, neginf=0.0)
            depth_map = self._resize_depth(depth_raw)

            yield FrameData(
                frame_idx=frame_idx,
                rgb=rgb,
                depth_map=depth_map,
            )
            frame_idx += 1

    def get_imu_data(self) -> dict:
        """
        IMU məlumatı — BoT-SORT CMC üçün.
        Kamera hərəkətini kompensasiya etmək üçün istifadə olunur.
        """
        sensors_data = sl.SensorsData()
        self._cam.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
        imu = sensors_data.get_imu_data()

        return {
            "angular_velocity":    imu.get_angular_velocity(),
            "linear_acceleration": imu.get_linear_acceleration(),
        }

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.resize(frame, (self.cfg.display_width, self.cfg.display_height))

    def _resize_depth(self, depth: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.resize(
            depth,
            (self.cfg.display_width, self.cfg.display_height),
            interpolation=cv2.INTER_NEAREST,  # depth üçün nearest — interpolasiya etmə
        )

    def release(self) -> None:
        if self._cam is not None:
            self._cam.close()
            logger.info("ZED X bağlandı.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()