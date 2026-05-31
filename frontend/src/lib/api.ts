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
  generatorSampleProbeCsvUrl: () => `${BASE}/generator/sample_probe_csv`,
  generatorInjectCsv: async (file: File, autoStart: boolean = true) => {
    const form = new FormData();
    form.append("file", file);
    const url = `${BASE}/generator/inject_csv?auto_start=${autoStart}`;
    const r = await fetch(url, { method: "POST", body: form });
    if (!r.ok) {
      let detail = `${r.status}`;
      try { detail = JSON.stringify(await r.json()); } catch { /* ignore */ }
      throw new Error(`inject_csv failed: ${detail}`);
    }
    return r.json() as Promise<{
      filename: string;
      queued_events: number;
      rows_with_errors: number;
      errors: string[];
      queue_depth: number;
      generator_running: boolean;
      generator_auto_started: boolean;
    }>;
  },

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

  // scenario studio
  studioCatalog: () => _get<{ typologies: any[] }>("/studio/catalog"),
  studioPreview: (typology: string, params: Record<string, any>, seed: number, count: number = 1) => {
    const q = new URLSearchParams();
    q.set("typology", typology);
    q.set("seed", String(seed));
    q.set("count", String(count));
    q.set("params", JSON.stringify(params));
    return _get<any>(`/studio/preview?${q.toString()}`);
  },
  studioInject: (typology: string, params: Record<string, any>, seed: number, count: number = 1) =>
    _post<any>("/studio/inject", { typology, params, seed, count }),
  studioExport: (scenarios: Array<{ typology: string; params: Record<string, any>; seed: number; count: number }>) =>
    _post<any>("/studio/export", { scenarios }),
  studioExportDownloadUrl: () => "/api/studio/export/download",
  studioClearExport: async () => {
    const r = await fetch("/api/studio/export", { method: "DELETE" });
    return r.json();
  },
  studioActivity: () => _get<any>("/studio/activity"),
};

export const SSE_URL = `${BASE}/stream/events`;
