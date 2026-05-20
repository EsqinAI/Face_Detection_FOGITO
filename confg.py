# config.py
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CameraConfig:
    # ZED X parametrləri
    resolution: str = "HD1080"        # HD720 | HD1080 | HD2K
    fps: int = 30                     # 30 | 60
    depth_mode: str = "ULTRA"         # PERFORMANCE | QUALITY | ULTRA
    depth_min_dist: float = 0.3       # minimum metr
    depth_max_dist: float = 15.0      # maximum metr
    display_width: int = 1280
    display_height: int = 720


@dataclass
class DetectorConfig:
    # RetinaFace
    backbone: str = "resnet50"        # resnet50 | mobilenet
    confidence: float = 0.7
    device: str = "cuda"
    input_size: int = 1280 # eger performans lazim olarsa 960 veya 640 yoxlanila biler


    # VACIB!!! performansi dahada yaxsilasdirmaq ucun TensorRT-den istifade etmek lazim gelir, performansi 3-4 qatina cixara biler
    # TensorRT üçün:
    # use_tensorrt: bool = True
    # engine_path: str = "models/retinaface.engine"


@dataclass
class TrackerConfig:
    # BoT-SORT
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 15            # 30 FPS × 1 saniyə, mən 15 olaraq dəyişdirirəm lazım gələrsə 30 qoyula bilər.
    match_thresh: float = 0.8
    # CMC — hərəkət edən kamera kompensasiyası
    cmc_method: str = "sparseOptFlow" # sparseOptFlow | sof | ecc | orb / sof daha cox suret, ecc daha cox deqiqlik, orb balans
    with_reid: bool = True            # appearance feature / eger agir gelerse False etmek olar
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    device: str = "cuda"


@dataclass
class AntiSpoofConfig:
    # ZED X Depth-based
    min_depth: float = 0.3            # 30cm — çox yaxın
    max_depth: float = 8.0            # 8m — billboard adətən 10m+ , deyisdirmek olar muhite gore
    depth_std_threshold: float = 0.15 # bbox içində depth variasiyası / 0.1-0.2 arasi yoxlanila biler
    # Real insan: depth std > threshold (3D forma)
    # Foto/ekran: depth std < threshold (düz sətih)
    sample_points: int = 16           # bbox içindən neçə nöqtə yoxlansın


@dataclass
class CropConfig:
    save_dir: str = "faces"
    padding: int = 20 # uzun kesilmemesi ucun elave piksel
    min_size: int = 60 # crop-un minimum olcusu
    jpeg_quality: int = 90


@dataclass
class AppConfig:
    camera: CameraConfig = None
    detector: DetectorConfig = None
    tracker: TrackerConfig = None
    anti_spoof: AntiSpoofConfig = None
    crop: CropConfig = None
    show_fps: bool = True
    show_confidence: bool = True
    target_fps: int = 30

    def __post_init__(self):
        if self.camera is None:
            self.camera = CameraConfig()
        if self.detector is None:
            self.detector = DetectorConfig()
        if self.tracker is None:
            self.tracker = TrackerConfig()
        if self.anti_spoof is None:
            self.anti_spoof = AntiSpoofConfig()
        if self.crop is None:
            self.crop = CropConfig()