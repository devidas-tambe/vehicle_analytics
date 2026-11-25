"""Read-only Flask Analytics API for React Dashboard."""
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg
from psycopg.rows import dict_row

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:5173"],
            "methods": ["GET"],
        }
    },
)

DSN = os.getenv(
    "DATABASE_URL", "postgresql://vehicle:vehicle_dev@127.0.0.1:5432/vehicles"
)
VALID_MODES = ("live", "demo", "windows")


@app.route("/api/health", methods=["GET"])
def health():
    try:
        with psycopg.connect(DSN, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
        return jsonify({"status": "ok"}), 200
    except psycopg.Error as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 503


@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    mode = request.args.get("mode", "windows")
    if mode not in VALID_MODES:
        return jsonify({"error": f"Invalid mode. Choose from {VALID_MODES}"}), 400

    try:
        hours = int(request.args.get("hours", 24))
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return jsonify({"error": "hours and limit must be integers"}), 400

    if not (1 <= hours <= 8760):
        return jsonify({"error": "hours must be between 1 and 8760"}), 400
    if not (1 <= limit <= 200):
        return jsonify({"error": "limit must be between 1 and 200"}), 400

    try:
        with psycopg.connect(DSN, row_factory=dict_row, connect_timeout=5) as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            where = (
                "WHERE mode = %s AND crossed_at >= now() - make_interval(hours => %s)"
            )
            params = (mode, hours)

            counts = conn.execute(
                "SELECT count(*) AS total, "
                "count(*) FILTER (WHERE direction='up') AS up, "
                "count(*) FILTER (WHERE direction='down') AS down "
                "FROM vehicle_events " + where,
                params,
            ).fetchone()

            colors = conn.execute(
                "SELECT color, count(*) AS count "
                "FROM vehicle_events " + where + " "
                "GROUP BY color ORDER BY count DESC, color",
                params,
            ).fetchall()

            raw_events = conn.execute(
                "SELECT id, run_id, source_id, vehicle_id, color, confidence, crossed_at, direction "
                "FROM vehicle_events " + where + " "
                "ORDER BY crossed_at DESC, id DESC LIMIT %s",
                (*params, limit),
            ).fetchall()

        events = [
            {
                "id": row["id"],
                "run_id": str(row["run_id"]),
                "source_id": row["source_id"],
                "vehicle_id": row["vehicle_id"],
                "color": row["color"],
                "confidence": row["confidence"],
                "crossed_at": row["crossed_at"].isoformat(),
                "direction": row["direction"],
            }
            for row in raw_events
        ]

        return jsonify(
            {
                "mode": mode,
                "hours": hours,
                "total": counts["total"],
                "up": counts["up"],
                "down": counts["down"],
                "colors": colors,
                "events": events,
            }
        ), 200
    except psycopg.Error as exc:
        return jsonify({"error": f"Database error: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
