import HealthPanel from "./components/HealthPanel";
import RFDetectionPanel from "./components/RFDetectionPanel";

export default function App() {
  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <span style={styles.icon}>📡</span>
        <h1 style={styles.title}>RF Threat Classification</h1>
        <span style={styles.subtitle}>FastAPI + React test harness</span>
      </header>

      <div style={styles.grid}>
        <HealthPanel />
        <RFDetectionPanel />
      </div>
    </div>
  );
}

const styles = {
  page:     { background: "#0d1117", minHeight: "100vh", padding: "24px 32px", boxSizing: "border-box" },
  header:   { display: "flex", alignItems: "center", gap: 12, marginBottom: 32 },
  icon:     { fontSize: 28 },
  title:    { color: "white", fontSize: 22, fontWeight: 700, margin: 0, fontFamily: "sans-serif" },
  subtitle: { color: "#6b7280", fontSize: 13, fontFamily: "sans-serif" },
  grid:     { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, maxWidth: 900 },
};
