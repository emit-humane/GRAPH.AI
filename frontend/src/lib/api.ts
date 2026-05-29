// Single source of truth for backend calls. All paths route through the
// Next.js /api proxy which rewrites to NEXT_PUBLIC_BACKEND_URL.

const BASE = "/api";

async function _get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}

async function _post<T>(path: string, body?: any): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`POST ${path} → ${r.status}`);
  return r.json();
}

async function _patch<T>(path: string, body: any): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`PATCH ${path} → ${r.status}`);
  return r.json();
}

export const api = {
  // generator
  generatorStatus: () => _get<any>("/generator/status"),
  generatorStart: (cfg?: { min_amount?: number; max_amount?: number; interval_seconds?: number }) =>
    _post<any>("/generator/start", cfg),
  generatorStop: () => _post<any>("/generator/stop"),

  // alerts
  alerts: (params?: { risk_level?: string; alert_status?: string; limit?: number }) => {
    const q = new URLSearchParams();
    if (params?.risk_level) q.set("risk_level", params.risk_level);
    if (params?.alert_status) q.set("alert_status", params.alert_status);
    if (params?.limit) q.set("limit", String(params.limit));
    return _get<{ count: number; items: any[] }>(`/alerts?${q.toString()}`);
  },
  alert: (id: string) => _get<any>(`/alerts/${id}`),
  alertStatus: (id: string, status: string, assigned_to?: string) =>
    _patch<any>(`/alerts/${id}/status`, { alert_status: status, assigned_to }),

  // graph + geo
  caseGraph: (txId: string) => _get<any>(`/graph/case/${txId}`),
  subgraph: (nodeId: string, hops: number = 2) =>
    _get<any>(`/graph/subgraph?node_id=${encodeURIComponent(nodeId)}&hops=${hops}`),
  caseGeo: (txId: string) => _get<any>(`/geo/case/${txId}`),

  // report
  report: (txId: string) => _get<any>(`/report/${txId}`),
  reportPdfUrl: (txId: string) => `${BASE}/report/${txId}/pdf`,
  reportZipUrl: (txId: string) => `${BASE}/report/${txId}/zip`,

  // stats
  evaluation: () => _get<any>("/stats/evaluation"),
};

export const SSE_URL = `${BASE}/stream/events`;
