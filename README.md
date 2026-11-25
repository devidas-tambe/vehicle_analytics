# Vehicle Analytics (Flask + React + PostgreSQL + YOLO)

Real-time vehicle line-crossing analytics built with:
- **Backend:** Flask + Gunicorn + psycopg3
- **Frontend:** React 18 + Vite + Nginx
- **Database:** PostgreSQL 16
- **Computer Vision:** Ultralytics YOLOv8 + ByteTrack + OpenCV HSV color estimation
- **Streaming:** MediaMTX (RTSP / HLS)

## Quick Start (Docker)

1. Start PostgreSQL, Flask API, React Frontend, and MediaMTX:
   ```bash
   docker compose up -d --build db api frontend mediamtx
   ```
   *(If port `5432` is already in use on your machine, change `127.0.0.1:5432:5432` to `127.0.0.1:5433:5432` in `compose.yaml` and set `DATABASE_URL` accordingly.)*

2. Open the dashboard:
   - React Dashboard: `http://localhost:3000`
   - Flask API Health: `http://localhost:8000/api/health`

3. Run synthetic demo mode (optional):
   ```bash
   docker compose --profile demo up -d demo
   ```

## Run Real Video Inference (Local Python)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r vision/requirements.txt
python -m vision.inference --source path\to\traffic.mp4
```

## Run Unit Tests

```bash
python -m unittest discover -s tests -v
```
