"""CPU vehicle detection, ByteTrack tracking, color estimation, line/instant counting, and RTSP publishing."""
import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.color import classify_color
from vision.counting import LineCounter
from vision.writer import EventWriter


def start_ffmpeg_publisher(url: str, width: int, height: int, fps: int = 15):
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-g", str(fps),
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        url,
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="rtsp://127.0.0.1:8554/traffic",
        help="RTSP URL, MP4 path, or webcam index (e.g. 0)",
    )
    parser.add_argument(
        "--publish",
        default="rtsp://127.0.0.1:8554/annotated",
        help="RTSP URL to publish annotated video (empty string to disable)",
    )
    parser.add_argument("--line-y", type=float, default=0.55)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--source-id", default="windows-camera-01")
    parser.add_argument(
        "--instant",
        action="store_true",
        help="Log vehicle immediately when detected instead of waiting for line crossing",
    )
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    counter = LineCounter(args.line_y)
    model = YOLO(args.model)

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    source = int(args.source) if args.source.isdecimal() else args.source

    if isinstance(source, str) and source.lower().startswith("rtsp://"):
        cap = cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                10000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                10000,
            ],
        )
    else:
        cap = cv2.VideoCapture(source)

    # Reduce latency and stale frames
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cap.release()
        raise SystemExit("Cannot open video source. Check file path or RTSP stream.")

    dsn = os.getenv(
        "DATABASE_URL", "postgresql://vehicle:vehicle_dev@127.0.0.1:5432/vehicles"
    )
    writer = EventWriter(dsn, mode="windows")
    run_id = uuid.uuid4()
    histories = defaultdict(lambda: deque(maxlen=7))
    last_seen = {}
    counted_tracks = set()
    streamer = None
    frames = 0
    total = 0
    started = time.monotonic()

    logging.info("Inference run started: %s (Press Q in preview or Ctrl+C to stop)", run_id)
    flushed = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logging.info("End of video or stream interrupted.")
                break

            frames += 1
            height, width = frame.shape[:2]

            # Start publishing annotated stream to MediaMTX on first frame
            if args.publish and streamer is None:
                logging.info("Publishing annotated stream to %s", args.publish)
                streamer = start_ffmpeg_publisher(args.publish, width, height, fps=15)

            # COCO vehicle classes: 2=car, 3=motorcycle, 5=bus, 7=truck
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[2, 3, 5, 7],
                device="cpu",
                imgsz=args.imgsz,
                conf=0.25,
                verbose=False,
            )[0]

            boxes = results.boxes
            if boxes is not None and boxes.id is not None:
                for coords, track, cls, conf in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.id.int().cpu().tolist(),
                    boxes.cls.int().cpu().tolist(),
                    boxes.conf.cpu().tolist(),
                ):
                    x1, y1, x2, y2 = map(int, coords)
                    x1, x2 = max(0, x1), min(width, x2)
                    y1, y2 = max(0, y1), min(height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    raw_color = classify_color(frame[y1:y2, x1:x2])
                    histories[track].append(raw_color)
                    last_seen[track] = frames
                    color = Counter(histories[track]).most_common(1)[0][0]

                    if args.instant:
                        if track not in counted_tracks:
                            counted_tracks.add(track)
                            direction = "down" if (y2 / height) >= args.line_y else "up"
                            writer.submit(
                                (
                                    run_id,
                                    args.source_id,
                                    str(track),
                                    color,
                                    float(conf),
                                    datetime.now(timezone.utc),
                                    direction,
                                )
                            )
                            total += 1
                    else:
                        direction = counter.observe(track, y2 / height)
                        if direction:
                            writer.submit(
                                (
                                    run_id,
                                    args.source_id,
                                    str(track),
                                    color,
                                    float(conf),
                                    datetime.now(timezone.utc),
                                    direction,
                                )
                            )
                            total += 1

                    # Draw border, category, confidence, color, and coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 80), 2)
                    label = f"{model.names[cls]} {conf:.2f} | #{track} {color} | ({x1},{y1},{x2},{y2})"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (60, 220, 80),
                        2,
                    )

            if frames % 300 == 0:
                for key in list(last_seen):
                    if frames - last_seen[key] > 300:
                        histories.pop(key, None)
                        last_seen.pop(key, None)
                        counter.side.pop(key, None)

            if writer.failed.is_set():
                raise RuntimeError("Database writer failed")

            # Draw line & HUD on frame (visible in both browser & OpenCV window)
            line = int(height * args.line_y)
            cv2.line(frame, (0, line), (width, line), (0, 220, 255), 2)
            fps = frames / max(time.monotonic() - started, 0.001)
            cv2.putText(
                frame,
                f"Windows CPU | Count {total} | {fps:.1f} FPS",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
            )

            # Push annotated frame to MediaMTX
            if streamer is not None and streamer.stdin is not None:
                try:
                    streamer.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    logging.warning("Annotated stream pipe closed; disabling publish.")
                    streamer = None

            if not args.no_preview:
                cv2.imshow("Vehicle Analytics - Flask + React", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        logging.info("Stopping inference...")
    finally:
        cap.release()
        if streamer is not None:
            try:
                if streamer.stdin:
                    streamer.stdin.close()
                streamer.terminate()
            except Exception:
                pass
        if not args.no_preview:
            cv2.destroyAllWindows()
        flushed = writer.close()

    if not flushed:
        raise SystemExit("Database writer did not flush all events; review logs.")

    logging.info("Processed %d frames, counted %d vehicles.", frames, total)


if __name__ == "__main__":
    main()