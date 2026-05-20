# main.py
import logging
import time
import argparse

import cv2

from config import AppConfig
from core.video_reader import ZEDVideoReader
from core.detector import FaceDetector
from core.tracker import FaceTracker
from core.anti_spoof import AntiSpoof
from core.crop_manager import CropManager
from utils.drawing import FPSCounter, draw_tracked_faces, draw_fps, draw_info_bar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(cfg: AppConfig) -> None:
    detector = FaceDetector(cfg.detector)
    detector.load()

    anti_spoof = AntiSpoof(cfg.anti_spoof)
    anti_spoof.load()

    tracker = FaceTracker(cfg.tracker)
    tracker.load()

    crop_manager = CropManager(cfg.crop, anti_spoof=anti_spoof)

    fps_counter = FPSCounter(window=30)
    frame_time = 1.0 / cfg.target_fps

    with ZEDVideoReader(cfg.camera) as vr:
        logger.info("Pipeline başladı. Çıxmaq üçün 'q' basın.")

        for frame_data in vr.read_frames():
            frame_start = time.perf_counter()
            fps = fps_counter.tick()

            # Detection
            detections = detector.detect(frame_data.rgb)

            # Tracking
            tracked_faces = tracker.update(
                detections, frame_data.rgb, frame_data.frame_idx
            )

            # Crop + Anti-spoof (depth ilə)
            crop_manager.process(
                tracked_faces,
                frame_data.rgb,
                frame_data.depth_map,
            )

            # Render
            draw_tracked_faces(frame_data.rgb, tracked_faces, cfg.show_confidence)
            draw_fps(frame_data.rgb, fps)
            draw_info_bar(frame_data.rgb, frame_data.frame_idx, len(tracked_faces))

            cv2.imshow("Face Tracker", frame_data.rgb)

            elapsed = time.perf_counter() - frame_start
            wait_ms = max(1, int((frame_time - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                logger.info("Dayandırıldı.")
                break

    cv2.destroyAllWindows()
    logger.info(f"Pipeline tamamlandı | Saxlanan üz: {crop_manager.saved_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    cfg = AppConfig()
    cfg.target_fps = args.fps

    run(cfg)