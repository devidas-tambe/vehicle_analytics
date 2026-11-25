"""Asynchronous PostgreSQL event writer with retries and idempotent inserts."""
import logging
import queue
import threading
import time
import psycopg

SQL = """
INSERT INTO vehicle_events
(run_id, source_id, vehicle_id, color, confidence, crossed_at, direction, mode)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (run_id, source_id, vehicle_id) DO NOTHING
"""


class EventWriter:
    def __init__(self, dsn, max_pending=10000, mode="windows"):
        if mode not in ("live", "windows"):
            raise ValueError("Unsupported inference mode")
        self.mode = mode
        self.dsn = dsn
        self.queue = queue.Queue(maxsize=max_pending)
        self.stop = threading.Event()
        self.failed = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, event):
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.failed.set()
            raise RuntimeError(
                "Database queue full; stopping inference to avoid silent event loss"
            )

    def _run(self):
        pending = None
        conn = None
        try:
            while not self.stop.is_set() or pending is not None or not self.queue.empty():
                if pending is None:
                    try:
                        pending = self.queue.get(timeout=0.25)
                    except queue.Empty:
                        continue
                try:
                    if conn is None or conn.closed:
                        conn = psycopg.connect(
                            self.dsn, connect_timeout=5, autocommit=True
                        )
                    conn.execute(SQL, (*pending, self.mode))
                    logging.info(
                        "Stored crossing: track=%s color=%s direction=%s",
                        pending[2],
                        pending[3],
                        pending[6],
                    )
                    pending = None
                    self.queue.task_done()
                except psycopg.Error as exc:
                    logging.error("Database write failed; retrying: %s", exc)
                    if conn is not None:
                        conn.close()
                    conn = None
                    time.sleep(2)
        finally:
            if conn is not None:
                conn.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            logging.error(
                "Writer could not flush all events: %s unacknowledged",
                self.queue.unfinished_tasks,
            )
            return False
        return not self.failed.is_set()
