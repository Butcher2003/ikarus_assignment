export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";
export const ASSET_BASE_URL = API_BASE_URL.replace(/\/api$/, "");

async function handleResponse(response) {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export async function chatWithAssistant(messages, topK = 3) {
  const payload = {
    messages: messages.map(({ role, content }) => ({ role, content })),
    top_k: topK,
  };
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return handleResponse(response);
}

export async function fetchAnalytics() {
  const response = await fetch(`${API_BASE_URL}/analytics/summary`);
  return handleResponse(response);
}

export async function fetchClusterDetail(clusterId, limit = 30) {
  const searchParams = new URLSearchParams({ limit: String(limit) });
  const response = await fetch(`${API_BASE_URL}/analytics/cluster/${clusterId}?${searchParams.toString()}`);
  return handleResponse(response);
}

export async function fetchRecommendations(query, topK = 6) {
  const payload = { query, top_k: topK };
  const response = await fetch(`${API_BASE_URL}/recommend`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return handleResponse(response);
}

export async function generateDescription({ query, productIds = [], topK = 6 }) {
  const payload = {
    query,
    product_ids: productIds,
    top_k: topK,
  };
  const response = await fetch(`${API_BASE_URL}/description/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return handleResponse(response);
}
