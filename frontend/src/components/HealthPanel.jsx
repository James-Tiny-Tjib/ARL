import { useState } from "react";
import { fetchHealth, fetchPing } from "../api";

export default function HealthPanel() {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePing = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPing();
      setStatus({ type: "ping", ...data });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHealth();
      setStatus({ type: "health", ...data });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>🛰 Server Status</h2>

      <div style={styles.buttonRow}>
        <button style={styles.btn} onClick={handlePing} disabled={loading}>
          Ping
        </button>
        <button style={styles.btn} onClick={handleHealth} disabled={loading}>
          Health Check
        </button>
      </div>

      {loading && <p style={styles.muted}>Calling backend…</p>}

      {error && <p style={styles.error}>Error: {error}</p>}

      {status && !loading && (
        <div style={styles.result}>
          {status.type === "ping" ? (
            <>
              <p><span style={styles.label}>Message:</span> {status.message}</p>
              <p><span style={styles.label}>Timestamp:</span> {new Date(status.timestamp * 1000).toLocaleTimeString()}</p>
            </>
          ) : (
            <>
              <p><span style={styles.label}>Status:</span> {status.status}</p>
              <p><span style={styles.label}>Device:</span> {status.device}</p>
              <p style={styles.label}>Models loaded:</p>
              <ul style={styles.list}>
                {Object.entries(status.models_loaded).map(([name, loaded]) => (
                  <li key={name}>
                    <span style={loaded ? styles.ok : styles.notOk}>
                      {loaded ? "✓" : "✗"}
                    </span>{" "}
                    {name}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}

const styles = {
  card:      { background: "#1a1f2e", border: "1px solid #2a2f3a", borderRadius: 12, padding: 24, color: "#e2e8f0", fontFamily: "sans-serif" },
  title:     { margin: "0 0 16px", fontSize: 18, fontWeight: 700 },
  buttonRow: { display: "flex", gap: 10, marginBottom: 16 },
  btn:       { background: "#3b82f6", color: "white", border: "none", borderRadius: 8, padding: "8px 18px", cursor: "pointer", fontWeight: 600 },
  muted:     { color: "#6b7280", fontSize: 13 },
  error:     { color: "#ef4444", fontSize: 13 },
  result:    { background: "#0d1117", borderRadius: 8, padding: 14, fontSize: 13, lineHeight: 1.7 },
  label:     { color: "#9ca3af", fontWeight: 600 },
  list:      { margin: "4px 0 0 16px", padding: 0 },
  ok:        { color: "#22c55e", fontWeight: 700 },
  notOk:     { color: "#ef4444", fontWeight: 700 },
};
