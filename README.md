# Realtime AI Surveillance Analytics System

> **YouTube Stream → YOLOv8 → ByteTrack → Line Counter → PostgreSQL → Streamlit Dashboard → AI Chatbot**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE PROCESS                         │
│                                                                 │
│  YouTube URL ──► StreamHandler ──► PersonDetector (YOLOv8n)   │
│                    (yt-dlp +          │  sv.Detections           │
│                     OpenCV)           │                          │
│                                       ▼                          │
│                              PersonTracker (ByteTrack)          │
│                                       │  tracker_id assigned     │
│                                       ▼                          │
│                               LineCounter                        │
│                                       │  CrossingEvent           │
│                          ┌────────────┤                          │
│                          ▼            ▼                          │
│                     PostgreSQL    LiveState (in-proc)            │
└──────────────────────────┼──────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
    FastAPI REST      WebSocket         Streamlit
    /analytics/*      /events/ws        dashboard
    /events/latest                       + Plotly
    /chat                                + AI Chatbot
         │                                    │
         └──────────── Gemini API ────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 14+ running locally (or use Docker Compose)
- `ffmpeg` installed (`brew install ffmpeg` / `apt install ffmpeg`)

### 1 — Clone & install

```bash
git clone <repo>
cd surveillance_system
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
# Edit .env:
#   YOUTUBE_STREAM_URL=https://www.youtube.com/watch?v=<live_id>
#   POSTGRES_PASSWORD=your_password
#   GEMINI_API_KEY=your_gemini_api_key
```

### 3 — Start PostgreSQL & create tables

```bash
# Option A: Docker
docker compose up db -d

# Option B: local Postgres already running
# (set credentials in .env)

python scripts/init_db.py
```

### 4 — Start services (three terminals)

```bash
# Terminal 1 — FastAPI backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Detection pipeline
python scripts/run_pipeline.py

# Terminal 3 — Streamlit dashboard
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

### Docker Compose (all-in-one)

```bash
cp .env.example .env   # fill in secrets
docker compose up --build
```

| Service    | Port  | Description                    |
|------------|-------|--------------------------------|
| PostgreSQL | 5432  | Database                       |
| FastAPI    | 8000  | REST + WebSocket API           |
| Pipeline   | —     | YOLOv8 + ByteTrack worker      |
| Streamlit  | 8501  | Dashboard                      |

---

## Project Structure

```
surveillance_system/
├── dataset/
│   ├── dataset.yaml          # YOLOv8 training config
│   ├── images/train|val      # training and validation images
│   └── labels/train|val      # YOLO labels
├── training/
│   └── train.py              # YOLOv8 fine-tuning pipeline
├── inference/
│   └── model_loader.py       # pretrained/custom model loading
├── models/                   # versioned best.pt / last.pt outputs
├── experiments/              # Ultralytics runs, metrics, plots
├── config/
│   └── settings.py          # Pydantic-settings config (reads .env)
├── core/
│   ├── stream_handler.py    # yt-dlp + OpenCV frame iterator
│   ├── detector.py          # YOLOv8 person detector
│   ├── tracker.py           # ByteTrack multi-object tracker
│   ├── line_counter.py      # Virtual line + crossing detection
│   └── database.py          # SQLAlchemy models + CRUD helpers
├── api/
│   ├── main.py              # FastAPI app + lifespan
│   ├── state.py             # In-process live state store
│   ├── schemas.py           # Pydantic request/response models
│   └── routes/
│       ├── analytics.py     # GET /analytics/*
│       ├── events.py        # GET /events/latest + WS /events/ws
│       └── chatbot.py       # POST /chat  (Gemini)
├── dashboard/
│   ├── app.py               # Streamlit multi-section dashboard
│   └── utils.py             # API client helpers
├── scripts/
│   ├── init_db.py           # One-time DB schema creation
│   └── run_pipeline.py      # Main pipeline entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Fine-Tuning YOLOv8

This project can run with generic pretrained YOLO weights or a custom fine-tuned model. Fine-tuning starts from `yolov8n.pt` or `yolov8s.pt`, learns your camera/domain data, and exports versioned `best.pt` and `last.pt` weights.

### Dataset Format

Use standard YOLO format:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each image needs a matching `.txt` label file with the same relative name. A label row is:

```text
class_id x_center y_center width height
```

Coordinates must be normalized from `0` to `1`. For person-only training, use class `0`.

### Collect And Label Data

Collect frames from the same camera angles, lighting, distance, crowd density, and weather conditions where the system will run. Include hard examples: partial people, motion blur, night scenes, shadows, occlusion, reflections, and empty frames.

Label images with CVAT, Label Studio, Roboflow, or makesense.ai, then export in YOLOv8/YOLO format. Keep labels consistent: one box around each visible person.

### Prepare Dataset

Split raw images and labels:

```bash
python scripts/prepare_dataset.py --images raw/images --labels raw/labels --target dataset --val-ratio 0.2
```

Validate an existing dataset:

```bash
python scripts/prepare_dataset.py --target dataset --validate-only
```

Import a Roboflow YOLOv8 export:

```bash
python scripts/prepare_dataset.py --roboflow-dir path/to/roboflow-export --target dataset
```

The utility checks missing labels, empty labels, train/val counts, and class statistics. It also writes `dataset/dataset.yaml`.

### Train

Run transfer learning from pretrained YOLOv8:

```bash
python train.py --weights yolov8n.pt --data dataset/dataset.yaml --epochs 50 --batch 16 --imgsz 640
```

GPU is selected automatically when CUDA is available. You can force it:

```bash
python train.py --device 0
```

Outputs:

```text
experiments/<version>/weights/best.pt
experiments/<version>/weights/last.pt
models/<version>/best.pt
models/<version>/last.pt
models/latest/best.pt
models/registry.json
```

Ultralytics also saves `results.csv`, loss curves, precision/recall curves, PR curves, confusion matrix, and validation metrics. Important plots are copied to `models/<version>/visualizations/`.

### Validate Metrics

After training, validation runs automatically and writes `training_summary.json` with:

- precision
- recall
- mAP50
- mAP50-95
- loss-related Ultralytics metrics when available

### Use Custom Model For Inference

Set `.env`:

```env
YOLO_MODEL=models/latest/best.pt
YOLO_CONFIDENCE=0.40
YOLO_DEVICE=auto
```

Then run the normal realtime pipeline:

```bash
python scripts/run_pipeline.py
```

To switch back to pretrained inference:

```env
YOLO_MODEL=yolov8n.pt
```

The detector loads models with:

```python
model = YOLO("models/best.pt")
```

through the project loader, so both pretrained weights and custom `best.pt` files are supported without changing the realtime flow.

---

## API Reference

| Method | Path                      | Description                            |
|--------|---------------------------|----------------------------------------|
| GET    | `/analytics/today`        | Today's total IN / OUT / total counts  |
| GET    | `/analytics/hourly?days=` | Per-hour stats for last N days         |
| GET    | `/analytics/peak-hour`    | Busiest hour in the last N days        |
| GET    | `/analytics/live`         | Real-time counters (poll every 3 s)    |
| GET    | `/events/latest?limit=`   | Most recent crossing events            |
| WS     | `/events/ws`              | WebSocket stream of live events        |
| POST   | `/chat`                   | AI chatbot (Gemini)                    |
| GET    | `/docs`                   | Swagger UI                             |

---

## AI Chatbot

The chatbot at `POST /chat` injects live DB context (today's counts, hourly
stats, peak hour, latest events) into the system prompt before every call.

Example questions:
- *"How many people crossed today?"*
- *"What was the peak traffic hour this week?"*
- *"Show hourly analytics for yesterday"*
- *"Are there more arrivals or departures?"*
- *"Summarise today's activity in one sentence"*

---

## Configuration Reference

| Variable              | Default                     | Description                         |
|-----------------------|-----------------------------|-------------------------------------|
| `YOUTUBE_STREAM_URL`  | (lofi live stream)          | Any YouTube live URL or RTSP/HLS    |
| `TARGET_FPS`          | 10                          | Frames sent to detector per second  |
| `YOLO_MODEL`          | `yolov8n.pt`                | nano/small/medium/large/extra       |
| `YOLO_PRETRAINED_MODEL` | `yolov8n.pt`              | fallback and fine-tuning start point |
| `YOLO_CONFIDENCE`     | 0.40                        | Detection threshold 0–1             |
| `YOLO_DEVICE`         | `auto`                      | `auto` / `cpu` / `0` (GPU) / `mps` |
| `DATASET_YAML`        | `dataset/dataset.yaml`      | YOLOv8 dataset config              |
| `TRAINING_PROJECT`    | `experiments`               | Ultralytics run output root         |
| `TRAINED_MODELS_DIR`  | `models`                    | versioned production weights root   |
| `TRAIN_EPOCHS`        | 50                          | default fine-tuning epochs          |
| `TRAIN_BATCH`         | 16                          | default training batch size         |
| `TRAIN_IMGSZ`         | 640                         | default training image size         |
| `LINE_START_X/Y`      | 0.0 / 0.5                   | Normalised line start (0–1)         |
| `LINE_END_X/Y`        | 1.0 / 0.5                   | Normalised line end (0–1)           |
| `GEMINI_API_KEY`      | -                           | Required for chatbot                |
| `GEMINI_MODEL`        | `gemini-2.0-flash`          | Any Gemini text model               |

---

## Performance Tips

- Use `YOLO_MODEL=yolov8s.pt` for better accuracy at modest CPU cost.
- Set `YOLO_DEVICE=0` if a CUDA GPU is available (10× faster inference).
- Lower `TARGET_FPS=5` on slow hardware to reduce load.
- The pipeline stores frames only briefly — no video is persisted to disk.
