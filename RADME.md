# Face Detection & Tracking Pipeline

Real-time face detection, tracking və anti-spoofing sistemi.
NVIDIA Jetson AGX Orin + ZED X kamera üçün hazırlanmış production pipeline.

---

## Arxitektura

ZED X Kamera

↓

ZEDVideoReader — RGB frame + Depth map

↓

RetinaFace (ResNet50) — Face detection

↓

BoT-SORT + CMC — Multi-face tracking

↓

Depth Anti-Spoof — ZED X depth ilə real/fake ayrımı

↓

CropManager — Face crop + server notification

---

## Hardware

| Komponent | Model |
|---|---|
| Edge Computer | NVIDIA Jetson AGX Orin 64GB |
| Kamera | ZED X (Stereolabs) |

---

## Software Stack

| Komponent | Texnologiya |
|---|---|
| Face Detection | RetinaFace (ResNet50 / MobileNet) |
| Tracking | BoT-SORT + CMC |
| Anti-Spoofing | ZED X Depth-based |
| Video Input | ZED SDK (pyzed) |
| Framework | PyTorch, OpenCV, Ultralytics |

---

## Qurulum

### 1. Tələblər

- Python 3.10 (Jetson JetPack) / 3.11 (dev)
- CUDA 12.4+
- [ZED SDK 4.x](https://www.stereolabs.com/developers/release)

### 2. Virtual mühit

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / Jetson
.venv\Scripts\activate           # Windows
```

### 3. PyTorch (CUDA)

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

### 4. Digər paketlər

```bash
pip install -r requirements.txt
```

### 5. Model faylları

`models/` qovluğunu yaradıb model fayllarını əlavə edin
(`.gitignore`-da olduğu üçün repo-da yoxdur):

```bash
mkdir models
# model fayllarını buraya əlavə edin
```

---

## İstifadə

```bash
python main.py --fps 30
```

---

## Folder Structure

face_tracker_prod/
├── core/
├── video_reader.py      # ZED X kamera reader
│   ├── detector.py          # RetinaFace detection
│   ├── tracker.py           # BoT-SORT tracking
│   ├── anti_spoof.py        # Depth-based anti-spoofing
│   └── crop_manager.py      # Face crop + save
├── utils/
│   └── drawing.py           # Visualization
├── config.py                # Bütün hyperparametrlər
├── main.py                  # Entry point
└── requirements.txt

---

## Konfiqurasiya

Bütün parametrlər `config.py`-dadır:

| Parametr | Sinif | Default | Qeyd |
|---|---|---|---|
| `backbone` | `DetectorConfig` | `resnet50` | Jetson-da `mobilenet` ilə başlayın |
| `input_size` | `DetectorConfig` | `1280` | Jetson-da `640` ilə başlayın |
| `depth_mode` | `CameraConfig` | `ULTRA` | Jetson-da `QUALITY` ilə başlayın |
| `cmc_method` | `TrackerConfig` | `sparseOptFlow` | Hərəkət edən kamera üçün |
| `with_reid` | `TrackerConfig` | `True` | Jetson-da əvvəlcə `False` |
| `max_depth` | `AntiSpoofConfig` | `8.0` | Billboard filtrasiyası (metr) |
| `depth_std_threshold` | `AntiSpoofConfig` | `0.15` | 3D yoxlama həssaslığı |

---

## Roadmap

- [ ] TensorRT export (RetinaFace → `.engine`)
- [ ] Face Recognition (ArcFace / AdaFace)
- [ ] REST API — server inteqrasiyası
- [ ] Multi-camera dəstəyi



