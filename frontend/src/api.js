// One place that knows how to reach the API, so components stay declarative.
// Vite proxies /api to the FastAPI process (see vite.config.js).

const BASE = "/api";

async function get(path, params = {}) {
  const query = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== "")
  );
  const suffix = query.toString() ? `?${query}` : "";
  const response = await fetch(`${BASE}${path}${suffix}`);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json();
}

export const api = {
  summary: () => get("/dashboard/summary"),
  timeline: (buckets = 48) => get("/dashboard/timeline", { buckets }),
  incidents: (limit = 40) => get("/dashboard/incidents", { limit }),
  transactions: (limit = 60, recommendation = null) =>
    get("/dashboard/transactions", { limit, recommendation }),
  health: () => get("/health"),
  ask: (question, merchantId = null) =>
    post("/assistant/ask", { question, merchant_id: merchantId }),
  uploadCsv: async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${BASE}/transactions/upload-csv`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Upload failed with status ${response.status}`);
    }
    return response.json();
  },
};

async function post(path, body) {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    throw new Error(errData.detail || `${path} returned ${response.status}`);
  }
  return response.json();
}

// The dataset's clock is a seconds offset with no calendar origin, so elapsed
// time is the only honest way to show it. Never format it as a date.
export function elapsed(seconds) {
  if (seconds === null || seconds === undefined) return "-";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

export const money = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

export const count = new Intl.NumberFormat("en-US");
