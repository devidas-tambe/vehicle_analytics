import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const swatches = {
  white: '#f5f5f5',
  black: '#222',
  silver: '#a3aab6',
  gray: '#747b85',
  red: '#ee5f63',
  blue: '#4e8ce9',
  green: '#32ac84',
  yellow: '#f1c64e',
  orange: '#f18c49',
  brown: '#8d5a40',
  unknown: '#a6a6a6',
};

function App() {
  const [mode, setMode] = useState('windows');
  const [hours, setHours] = useState(24);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [updated, setUpdated] = useState(null);
  const [video, setVideo] = useState(false);
  const [stream, setStream] = useState('annotated');
  const iframeRef = useRef(null);

  const handleFullScreen = () => {
    const el = iframeRef.current;
    if (!el) return;
    if (el.requestFullscreen) el.requestFullscreen();
    else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
  };

  useEffect(() => {
    let stopped = false;
    const controller = new AbortController();
    let timer;
    setData(null);
    setUpdated(null);
    setError('');

    async function refresh() {
      try {
        const r = await fetch(`/api/dashboard?mode=${mode}&hours=${hours}`, {
          signal: controller.signal,
          cache: 'no-store',
        });
        if (!r.ok) throw Error(`HTTP ${r.status}`);
        const payload = await r.json();
        if (!stopped) {
          setData(payload);
          setError('');
          setUpdated(new Date());
        }
      } catch (e) {
        if (!stopped && e.name !== 'AbortError') setError(e.message);
      } finally {
        if (!stopped) timer = setTimeout(refresh, 500);
      }
    }

    refresh();
    return () => {
      stopped = true;
      clearTimeout(timer);
      controller.abort();
    };
  }, [mode, hours]);

  const total = data?.total ?? 0;

  return (
    <main>
      <header>
        <div>
          <div className="eyebrow">VISION ENGINEERING / TRAFFIC MONITOR</div>
          <h1>Vehicle analytics</h1>
          <p>Line-crossing events · Color classification · Live monitoring</p>
        </div>
        <div className="status">
          <i className={error ? 'bad' : 'good'} />
          {error ? 'Connection error' : updated ? 'API connected' : 'Connecting'}
          <small>
            {updated ? `Updated ${updated.toLocaleTimeString()}` : 'Waiting for data'}
          </small>
        </div>
      </header>

      <section className="toolbar">
        <label>
          Data source{' '}
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="windows">Windows CPU (real video)</option>
            <option value="live">DeepStream (real video)</option>
            <option value="demo">Synthetic demo</option>
          </select>
        </label>
        <label>
          Time range{' '}
          <select value={hours} onChange={(e) => setHours(+e.target.value)}>
            <option value={1}>Last hour</option>
            <option value={24}>Last 24 hours</option>
            <option value={168}>Last 7 days</option>
            <option value={8760}>Last 365 days</option>
          </select>
        </label>
        <span>Auto-refresh: 0.5 seconds</span>
      </section>

      {mode === 'windows' && (
        <div className="notice">
          WINDOWS CPU — YOLO + ByteTrack + OpenCV color baseline.
        </div>
      )}
      {mode === 'demo' && (
        <div className="notice">
          DEMO MODE — randomly generated events, not camera detections.
        </div>
      )}
      {error && (
        <div role="alert" className="notice">
          API unavailable: {error}. Displayed data may be stale.
        </div>
      )}

      <section className="metrics">
        {[
          ['Total crossings', data?.total],
          ['Upward crossings', data?.up],
          ['Downward crossings', data?.down],
          ['Color categories', data?.colors.length],
        ].map(([label, n]) => (
          <article key={label}>
            <span>{label}</span>
            <strong>{n ?? '—'}</strong>
            <small>Selected time range</small>
          </article>
        ))}
      </section>

      <section className="split">
        <article>
          <h2>Vehicle color breakdown</h2>
          <p>Distribution of counted vehicles</p>
          {data?.colors.length ? (
            data.colors.map((c) => (
              <div className="barrow" key={c.color}>
                <span className="color">
                  <i style={{ background: swatches[c.color] || '#aaa' }} />
                  {c.color}
                </span>
                <div className="track">
                  <div
                    style={{
                      width: `${(100 * c.count) / Math.max(total, 1)}%`,
                      background: swatches[c.color] || '#aaa',
                    }}
                  />
                </div>
                <b>{c.count}</b>
              </div>
            ))
          ) : (
            <p className="empty">No crossing events yet.</p>
          )}
        </article>

        <article>
          <h2>Traffic stream</h2>
          <p>MediaMTX HLS · optional live video</p>
          <div className="videoControls">
            <select value={stream} onChange={(e) => setStream(e.target.value)}>
              <option value="annotated">Annotated stream</option>
              <option value="traffic">Original stream</option>
            </select>
            <button onClick={() => setVideo(!video)}>
              {video ? 'Stop preview' : 'Start preview'}
            </button>
            {video && (
              <button onClick={handleFullScreen}>Full Screen</button>
            )}
          </div>
          {video ? (
            <iframe
              ref={iframeRef}
              title="Live traffic video"
              src={`http://${location.hostname}:8888/${stream}/`}
              allow="autoplay; fullscreen"
              allowFullScreen
            />
          ) : (
            <div className="placeholder">
              Video preview paused
              <br />
              <small>Publish an RTSP stream to enable playback.</small>
            </div>
          )}
        </article>
      </section>

      <article className="events">
        <h2>Recent crossing events</h2>
        <p>Latest 50 records · timestamps shown in local time</p>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                {[
                  'Event',
                  'Track ID',
                  'Source',
                  'Color',
                  'Confidence',
                  'Direction',
                  'Timestamp',
                ].map((h) => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data?.events.map((e) => (
                <tr key={e.id}>
                  <td>#{e.id}</td>
                  <td>{e.vehicle_id}</td>
                  <td>{e.source_id}</td>
                  <td>
                    <span className="color">
                      <i style={{ background: swatches[e.color] || '#aaa' }} />
                      {e.color}
                    </span>
                  </td>
                  <td>
                    {e.confidence == null ? '—' : `${(100 * e.confidence).toFixed(0)}%`}
                  </td>
                  <td>{e.direction === 'up' ? '↑ Up' : '↓ Down'}</td>
                  <td>{new Date(e.crossed_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!data?.events.length && (
            <p className="empty">No events in this time range.</p>
          )}
        </div>
      </article>

      <footer>Vehicle Analytics / PostgreSQL + YOLO + Flask + React</footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);