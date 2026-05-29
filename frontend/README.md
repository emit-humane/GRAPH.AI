# GRAPH.AI Frontend

Next.js 14 (App Router) + Tailwind + Cytoscape + Leaflet. Talks to the FastAPI
backend at `:8000` via `/api/*` proxy rewrites configured in `next.config.js`.

## Quick start

```bash
cd frontend
npm install
npm run dev          # → http://localhost:3000
```

Or, from the repo root: `make install-frontend && make frontend`.

## Pages

| Route | Purpose |
|---|---|
| `/overview` | Live SSE stream + Start/Stop generator + flagged cards |
| `/case` | 5-layer breakdown bars, Cytoscape 2-hop chain, investigation notes |
| `/geo` | Leaflet India map showing the case's account locations |
| `/report` | FIU filing preview + PDF / ZIP downloads |
| `/graph` | Ad-hoc k-hop subgraph explorer |
| `/analytics` | Live aggregates (risk distribution, rule counts, pattern counts) |

## Shared infra

- `src/hooks/useEventStream.ts` — SSE subscription with auto-reconnect
- `src/stores/selectionStore.ts` — Zustand store for cross-tab case selection
- `src/lib/api.ts` — single source of truth for backend calls
- `src/components/ScoreBreakdownBars.tsx` — the 5-layer breakdown widget,
  with **Supervised (L3) marked as PRIMARY**

## Environment

| Var | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_BACKEND_URL` | `http://localhost:8000` | Backend root (rewritten by `/api/*` → `BACKEND/*`) |
