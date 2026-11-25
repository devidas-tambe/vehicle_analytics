"""Synthetic events ONLY. Does not run computer vision."""
import os
import random
import time
import uuid
from datetime import datetime, timezone
import psycopg

run_id = uuid.uuid4()
dsn = os.getenv(
    "DATABASE_URL", "postgresql://vehicle:vehicle_dev@127.0.0.1:5432/vehicles"
)
i = 0

while True:
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            while True:
                i += 1
                conn.execute(
                    """
                    INSERT INTO vehicle_events
                    (run_id, source_id, vehicle_id, color, confidence, crossed_at, direction, mode)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'demo')
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        run_id,
                        "demo-camera",
                        str(i),
                        random.choice(
                            ["white", "black", "silver", "red", "blue", "green", "yellow"]
                        ),
                        None,
                        datetime.now(timezone.utc),
                        random.choice(["up", "down"]),
                    ),
                )
                conn.commit()
                print(f"DEMO event {i}", flush=True)
                time.sleep(2)
    except psycopg.Error as exc:
        print(f"Demo database error: {exc}", flush=True)
        time.sleep(3)
