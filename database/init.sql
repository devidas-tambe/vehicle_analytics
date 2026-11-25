CREATE TABLE IF NOT EXISTS vehicle_events (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    source_id TEXT NOT NULL,
    vehicle_id TEXT NOT NULL,
    color TEXT NOT NULL,
    confidence REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    crossed_at TIMESTAMPTZ NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('up', 'down')),
    mode TEXT NOT NULL CHECK (mode IN ('live', 'demo', 'windows')),
    UNIQUE (run_id, source_id, vehicle_id)
);

CREATE INDEX IF NOT EXISTS event_recent_idx ON vehicle_events (mode, crossed_at DESC);
CREATE INDEX IF NOT EXISTS event_color_idx ON vehicle_events (mode, color);
