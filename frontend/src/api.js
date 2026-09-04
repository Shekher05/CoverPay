// Frontend API layer for CoverPay Fraud Intelligence.
//
// Local dev: BASE is "/api", which vite.config.js proxies to the FastAPI process.
// Production (Render Static Site): set VITE_API_BASE to the backend Web Service
// URL, e.g. https://coverpay-api.onrender.com - it is baked in at build time.
// A base URL is not a secret; real secrets stay in Render's backend env.
const BASE = (import.meta.env.VITE_API_BASE || "/api").replace(/\/+$/, "");

export function getStoredApiKey() {
  try {
    return localStorage.getItem("coverpay_api_key") || "";
  } catch {
    return "";
  }
}

export function setStoredApiKey(key) {
  try {
    if (key) {
      localStorage.setItem("coverpay_api_key", key);
    } else {
      localStorage.removeItem("coverpay_api_key");
    }
  } catch {
    // Local storage unavailable (incognito / sandboxed)
  }
}

function getHeaders(extraHeaders = {}) {
  const headers = { ...extraHeaders };
  const key = getStoredApiKey();
  if (key) {
    headers["X-API-Key"] = key;
  }
  return headers;
}

async function handleResponse(response, path) {
  if (!response.ok) {
    let errorDetail = "";
    try {
      const errData = await response.json();
      errorDetail = errData.detail || "";
    } catch {
      // Body not JSON
    }

    if (response.status === 401) {
      throw new Error(
        errorDetail || "Authentication failed: missing or invalid X-API-Key. Configure your API key in System settings."
      );
    }
    if (response.status === 429) {
      throw new Error(
        errorDetail || "Rate limit reached (30 requests/minute). Please wait a moment before trying again."
      );
    }
    if (response.status === 413) {
      throw new Error(
        errorDetail || "Payload too large. CSV files must be under 200 MB and 2,000,000 rows."
      );
    }
    if (response.status === 422) {
      throw new Error(
        errorDetail || "Validation error: the submitted transaction or CSV is missing required model features."
      );
    }
    if (response.status === 503) {
      throw new Error(
        errorDetail || "Service unavailable: the ML model or AI Assistant is not initialized or unavailable."
      );
    }

    throw new Error(errorDetail || `${path} returned HTTP ${response.status}`);
  }

  return response.json();
}

async function get(path, params = {}) {
  const query = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== "")
  );
  const suffix = query.toString() ? `?${query}` : "";
  const response = await fetch(`${BASE}${path}${suffix}`, {
    method: "GET",
    headers: getHeaders(),
  });
  return handleResponse(response, path);
}

async function post(path, body) {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: getHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  return handleResponse(response, path);
}

export const api = {
  // System Health
  health: () => get("/health"),

  // Operational Dashboard
  summary: () => get("/dashboard/summary"),
  timeline: (buckets = 48) => get("/dashboard/timeline", { buckets }),
  incidents: (limit = 50) => get("/dashboard/incidents", { limit }),
  transactions: (limit = 100, recommendation = null, incident_id = null) =>
    get("/dashboard/transactions", { limit, recommendation, incident_id }),

  // Single Transaction Lookup & Scoring
  getTransaction: (transactionId) => get(`/transactions/${encodeURIComponent(transactionId)}`),
  scoreTransaction: (transactionPayload) => post("/transactions/score", transactionPayload),

  // Batch CSV Upload Analysis
  uploadCsv: async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${BASE}/transactions/upload-csv`, {
      method: "POST",
      headers: getHeaders(),
      body: formData,
    });
    return handleResponse(response, "/transactions/upload-csv");
  },

  // AI Fraud Analyst Assistant
  ask: (question, merchantId = null, hours = null) =>
    post("/assistant/ask", {
      question,
      merchant_id: merchantId || null,
      hours: hours ? Number(hours) : null,
    }),
};

// The dataset clock is an offset in seconds from the dataset epoch.
// It is not a Unix timestamp, so elapsed time is the only honest way to format it.
export function elapsed(seconds) {
  if (seconds === null || seconds === undefined) return "n/a";
  const abs = Math.abs(seconds);
  const h = Math.floor(abs / 3600);
  const m = Math.floor((abs % 3600) / 60);
  const s = Math.floor(abs % 60);
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

export function formatHours(hours) {
  if (hours === null || hours === undefined) return "n/a";
  return `+${Number(hours).toFixed(1)}h`;
}

export const money = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

export const count = new Intl.NumberFormat("en-US");

