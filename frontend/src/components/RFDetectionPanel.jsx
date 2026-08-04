import { useState } from "react";
import { detectRF, fetchLatestSnapshot } from "../api";

export default function RFDetectionPanel() {
  const [result, setResult]     = useState(null);
  const [snapshot, setSnapshot] = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  const handleDetect = async () => {
    setLoading(true);
    setError(null);
    try {
      // No IQ data passed — backend generates a synthetic sample
      const data = await detectRF();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSnapshot = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchLatestSnapshot();
      setSnapshot(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const threatColor = (isT) => isT ? "#ef4444" : "#22c55e";

  return (
    <div style={styles.card}>
      <h2 style={styles.title}>📡 RF Detection</h2>

      <div style={styles.buttonRow}>
        <button style={styles.btn} onClick={handleDetect} disabled={loading}>
          Run RF Detection
        </button>
        <button style={{ ...styles.btn, background: "#6366f1" }} onClick={handleSnapshot} disabled={loading}>
          Latest Snapshot
        </button>
      </div>

      {loading && <p style={styles.muted}>Running…</p>}
      {error   && <p style={styles.error}>Error: {error}</p>}

      {result && !loading && (
        <div style={styles.result}>
          <p style={styles.sectionLabel}>RF Result</p>
          <div style={{ ...styles.verdict, borderColor: threatColor(result.is_threat) }}>
            <span style={{ color: threatColor(result.is_threat), fontSize: 18, fontWeight: 700 }}>
              {result.is_threat ? "⚠ THREAT" : "✓ CLEAR"}
            </span>
          </div>
          <p><span style={styles.label}>Threat prob:</span>   {(result.threat_prob * 100).toFixed(1)}%</p>
          <p><span style={styles.label}>Friendly prob:</span> {(result.friendly_prob * 100).toFixed(1)}%</p>
          <p><span style={styles.label}>Timestamp:</span>     {new Date(result.timestamp * 1000).toLocaleTimeString()}</p>
        </div>
      )}

      {snapshot && !loading && (
        <div style={{ ...styles.result, marginTop: 12 }}>
          <p style={styles.sectionLabel}>Latest Snapshot</p>
          <p><span style={styles.label}>Fusion threat:</span>{" "}
            <span style={{ color: threatColor(snapshot.fusion?.is_threat) }}>
              {snapshot.fusion?.is_threat ? "THREAT" : "CLEAR"}
            </span>
          </p>
          <p><span style={styles.label}>Fusion prob:</span> {((snapshot.fusion?.threat_prob ?? 0) * 100).toFixed(1)}%</p>
          <p><span style={styles.label}>RF:</span> {snapshot.rf?.is_threat ? "Threat" : "Clear"} — {((snapshot.rf?.threat_prob ?? 0) * 100).toFixed(1)}%</p>
          <p><span style={styles.label}>Audio:</span> {snapshot.audio?.is_threat ? "Threat" : "Clear"}</p>
          <p><span style={styles.label}>Visual:</span> {snapshot.visual?.is_threat ? "Threat" : "Clear"} — {((snapshot.visual?.drone_prob ?? 0) * 100).toFixed(1)}%</p>
          <p><span style={styles.label}>Time:</span> {new Date((snapshot.timestamp ?? 0) * 1000).toLocaleTimeString()}</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  card:         { background: "#1a1f2e", border: "1px solid #2a2f3a", borderRadius: 12, padding: 24, color: "#e2e8f0", fontFamily: "sans-serif" },
  title:        { margin: "0 0 16px", fontSize: 18, fontWeight: 700 },
  buttonRow:    { display: "flex", gap: 10, marginBottom: 16 },
  btn:          { background: "#3b82f6", color: "white", border: "none", borderRadius: 8, padding: "8px 18px", cursor: "pointer", fontWeight: 600 },
  muted:        { color: "#6b7280", fontSize: 13 },
  error:        { color: "#ef4444", fontSize: 13 },
  result:       { background: "#0d1117", borderRadius: 8, padding: 14, fontSize: 13, lineHeight: 1.8 },
  sectionLabel: { color: "#6b7280", fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 },
  label:        { color: "#9ca3af", fontWeight: 600 },
  verdict:      { border: "2px solid", borderRadius: 8, padding: "8px 14px", marginBottom: 10, display: "inline-block" },
};
