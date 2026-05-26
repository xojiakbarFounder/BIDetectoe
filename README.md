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
         └──────────── OpenAI API ────────────┘
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
#   OPENAI_API_KEY=sk-...
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
│       └── chatbot.py       # POST /chat  (OpenAI)
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

## API Reference

| Method | Path                      | Description                            |
|--------|---------------------------|----------------------------------------|
| GET    | `/analytics/today`        | Today's total IN / OUT / total counts  |
| GET    | `/analytics/hourly?days=` | Per-hour stats for last N days         |
| GET    | `/analytics/peak-hour`    | Busiest hour in the last N days        |
| GET    | `/analytics/live`         | Real-time counters (poll every 3 s)    |
| GET    | `/events/latest?limit=`   | Most recent crossing events            |
| WS     | `/events/ws`              | WebSocket stream of live events        |
| POST   | `/chat`                   | AI chatbot (OpenAI)                    |
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
| `YOLO_CONFIDENCE`     | 0.40                        | Detection threshold 0–1             |
| `YOLO_DEVICE`         | `cpu`                       | `cpu` / `0` (GPU) / `mps`          |
| `LINE_START_X/Y`      | 0.0 / 0.5                   | Normalised line start (0–1)         |
| `LINE_END_X/Y`        | 1.0 / 0.5                   | Normalised line end (0–1)           |
| `OPENAI_API_KEY`      | —                           | Required for chatbot                |
| `OPENAI_MODEL`        | `gpt-4o-mini`               | Any OpenAI chat model               |

---

## Performance Tips

- Use `YOLO_MODEL=yolov8s.pt` for better accuracy at modest CPU cost.
- Set `YOLO_DEVICE=0` if a CUDA GPU is available (10× faster inference).
- Lower `TARGET_FPS=5` on slow hardware to reduce load.
- The pipeline stores frames only briefly — no video is persisted to disk.
