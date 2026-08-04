const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Common headers — bypass Cloudflare tunnel browser challenge
const JSON_HEADERS = {
  "Content-Type": "application/json",
  "cf-access-client-id": "bypass",
};

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/health`, { headers: JSON_HEADERS });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function fetchPing() {
  const res = await fetch(`${BASE_URL}/ping`, { headers: JSON_HEADERS });
  if (!res.ok) throw new Error(`Ping failed: ${res.status}`);
  return res.json();
}

export async function detectRF(iqData = null) {
  // If no IQ data provided, send empty body — backend generates a synthetic sample
  const body = iqData
    ? { iq_data: iqData, noise_std: 0.1 }
    : { iq_data: null, noise_std: 0.1 };

  const res = await fetch(`${BASE_URL}/detect/rf`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`RF detection failed: ${res.status}`);
  return res.json();
}

export async function fetchLatestSnapshot() {
  const res = await fetch(`${BASE_URL}/snapshot/latest`, { headers: JSON_HEADERS });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Snapshot fetch failed: ${res.status}`);
  return res.json();
}
