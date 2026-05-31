# Deployment — Render (backend) + Vercel (frontend)

This guide walks the project end-to-end onto **Render** (FastAPI backend) and
**Vercel** (Next.js frontend). The repo ships with `render.yaml`,
`vercel.json`, and an env-driven CORS config — every config below is already
wired; you just need to choose a couple of values and click "Deploy".

---

## TL;DR — first deploy in 10 minutes

```
Backend on Render
  1. Push the repo to GitHub
  2. Render → "New" → "Blueprint" → pick this repo, branch=main
     (Render reads render.yaml and creates the web service + Postgres)
  3. Wait for the build (~15 min — torch + torch-geometric are heavy)
  4. Note the backend URL  → https://graphai-backend.onrender.com

Frontend on Vercel
  5. Vercel → "Add New" → "Project" → import the GitHub repo
     - Framework preset:  Next.js
     - Root directory:    frontend/
  6. Environment Variables:
       NEXT_PUBLIC_BACKEND_URL = https://graphai-backend.onrender.com
  7. Deploy

Lock down CORS
  8. Back in Render → graphai-backend → Environment:
       FRONTEND_ORIGIN = https://graphai.vercel.app
     (use your actual Vercel URL; comma-separated list is allowed)
  9. Trigger a manual redeploy (env vars only take effect on restart)
```

That's it. Open the Vercel URL, click **Start Generator** on the Overview
page, and the live stream should populate.

---

## What you need to choose up front

| Decision | Recommended | Cheaper | Why |
|---|---|---|---|
| Render instance | **Standard (2 GB RAM)** | Starter (512 MB) | Torch + multigraph + TGN need ~1.5 GB at steady state. Starter will OOM on first request. |
| Artifact source | **Tarball URL** | Commit via Git LFS, or "rules-only" mode | The trained models are ~72 MB. See *Artifact bootstrap* below for the three options. |
| Alerts DB | **Postgres (Render free plan)** | SQLite on Render Disk | SQLite at `logs/alerts.db` is wiped every deploy unless you attach a paid Render Disk. Postgres is in `render.yaml` already. |

---

## Backend on Render

### Render Blueprint (recommended)

The repo ships with `render.yaml`. On Render:

1. **New → Blueprint**
2. Select this repository, branch `main`
3. Render reads `render.yaml`, creates `graphai-backend` (web service) +
   `graphai-alerts-db` (Postgres) in one click
4. First build runs `pip install -r requirements.txt` (15–20 min — torch is
   ~600 MB, torch-geometric extensions another ~200 MB)
5. Health check: `GET /health` should return 200 once the build is up

### Resource sizing

```
                      RAM    cost    enough?
Render Free           512 MB free    NO  — torch alone won't fit
Render Starter        512 MB $7      NO  — boots, then OOMs on first request
Render Standard      2 GB   $25      YES — full 5-layer pipeline
Render Pro           4 GB   $85      YES — also runs the Scenario Studio + bigger contexts
```

The blueprint pins `plan: standard`. Bump to `pro` in `render.yaml` if you want
headroom for the Studio's bulk-injection paths.

### Required env vars (already in render.yaml)

| Var | Default | Notes |
|---|---|---|
| `PYTHONPATH` | `.` | so `src.*` resolves |
| `PYTHONIOENCODING` | `utf-8` | safe for the Indian-numeric labels |
| `PYTHONUNBUFFERED` | `1` | flush logs to Render dashboard in real time |
| `FRONTEND_ORIGIN` | `*` | set to your Vercel URL post-deploy |
| `DATABASE_URL` | from `graphai-alerts-db` | wired automatically by the Blueprint |
| `BOOTSTRAP_ARTIFACTS_URL` | *(unset)* | optional — see below |

### Health check

`GET /health` returns:
```json
{
  "status": "ok",
  "generator_running": false,
  "events_emitted": 0,
  "alerts_emitted": 0
}
```

Render polls this. If the service boots in degraded "rules-only" mode (no
artifacts) the health check still passes — only the L3/L4/L5 scores will be
null.

---

## Artifact bootstrap — three paths

The detector needs **~72 MB of trained artifacts** in `artifacts/`. They're
`.gitignore`d so they don't bloat the repo. Three deployable options:

### (A) Tarball URL — recommended

Train once locally, upload the tarball to a CDN (S3 / R2 / GitHub Release /
Cloudflare Pages), and let Render fetch it during the build.

**Locally:**
```bash
# Train the full pipeline (or run scripts/run_all.sh --clean)
make install
bash scripts/run_all.sh

# Package the artifacts/
tar -czf graphai-artifacts.tar.gz artifacts/

# Upload it somewhere public-readable. Examples:
#   - GitHub Release asset:  gh release create v1.0 graphai-artifacts.tar.gz
#   - Cloudflare R2:         wrangler r2 object put public/graphai-artifacts.tar.gz --file=...
#   - S3:                    aws s3 cp graphai-artifacts.tar.gz s3://bucket/ --acl public-read
```

**On Render:**
1. Service → Environment → add `BOOTSTRAP_ARTIFACTS_URL=https://your-cdn/graphai-artifacts.tar.gz`
2. Manual deploy → the build runs `scripts/bootstrap_artifacts.sh`, which
   `curl`s + extracts the tarball before `uvicorn` starts.

### (B) Git LFS

If you don't want a separate CDN: `git lfs install`, then `git lfs track "artifacts/*.pkl" "artifacts/*.pt" "artifacts/*.parquet" "artifacts/*.npy"`,
commit everything, and Render will clone the LFS objects automatically.

Trade-off: GitHub free LFS quota is 1 GB storage + 1 GB/month bandwidth.

### (C) Rules-only degraded mode (no artifacts needed)

Just deploy without setting `BOOTSTRAP_ARTIFACTS_URL`. The backend boots
fine; L1 rules and L2 live graph features still work because they're
computed from scratch on every event. The L3/L4/L5 inferencers report
`null` scores (the existing `_safe_load` graceful-fallback path), and the
dashboard shows them as 0.

This is the right mode for a "look how the rules layer works" demo or a
public read-only preview, since the supervised model is the layer that
most clearly carries IP.

---

## Frontend on Vercel

### Setup

1. **Vercel → Add New → Project**
2. Import this repo from GitHub
3. **Framework preset:** Next.js (auto-detected)
4. **Root directory:** `frontend/` (important — the repo has a Python
   project at the root)
5. **Build & dev settings:** all defaults
6. **Environment Variables:**
   ```
   NEXT_PUBLIC_BACKEND_URL=https://graphai-backend.onrender.com
   ```
   (use your actual Render URL; both Preview and Production environments)
7. Deploy

Vercel auto-deploys on every push to `main`. Preview deploys ship on PRs.

### The SSE buffering trap

Vercel buffers HTTP responses by default. The `/stream/events` route is
**Server-Sent Events** and would freeze unless we explicitly disable
buffering. The repo already handles this:

- `frontend/vercel.json` sets `Cache-Control: no-store` on `/api/*`
- `frontend/next.config.js` adds `X-Accel-Buffering: no` to
  `/api/stream/events`

If the dashboard shows "stream offline" or events stop arriving, that
header path is what to check first.

### Why we proxy through Next.js

The frontend talks to `/api/*` (relative path), which the Next.js rewrite
forwards to the Render backend. This means:

* **No CORS dance** for the browser — every request is same-origin.
* **No mixed-content issues** if you ever host frontend HTTP and backend
  HTTPS or vice versa.
* You can swap the backend URL without redeploying the frontend
  bundle — just edit the env var and redeploy.

---

## Lock down CORS post-deploy

`render.yaml` ships with `FRONTEND_ORIGIN=*`. That's fine for the first
boot. As soon as you know your Vercel URL, change it:

```
Render → graphai-backend → Environment → FRONTEND_ORIGIN
    https://graphai.vercel.app

(or multiple, comma-separated:)
    https://graphai.vercel.app,https://graphai-*.vercel.app
```

When `FRONTEND_ORIGIN` is anything other than `*`, the backend also
enables `Access-Control-Allow-Credentials: true` automatically. With `*`,
credentials are disabled (CORS spec forbids both at once).

---

## Verification checklist

After both deploys are live:

```bash
# 1. Backend health
curl https://graphai-backend.onrender.com/health
# expect: {"status":"ok","generator_running":false,...}

# 2. Backend layer availability
curl https://graphai-backend.onrender.com/generator/status | jq .layers_available
# Should report supervised/anomaly/tgn = true when artifacts are present.

# 3. Frontend reachable
curl -I https://graphai.vercel.app
# expect: HTTP/2 200

# 4. Frontend → Backend proxy
curl https://graphai.vercel.app/api/health
# expect: same JSON as step 1 (proves the rewrite works)

# 5. Live stream
curl -N https://graphai.vercel.app/api/stream/events
# expect: stays open, periodic comments / events as the generator runs
```

If step 5 hangs without output for >30 s after clicking Start Generator on
the dashboard, check the SSE buffering headers (frontend/next.config.js)
and that the Render instance has enough RAM (look for OOM kills in the
Render service logs).

---

## Cost summary

| Service | Plan | Monthly |
|---|---|---|
| Render web (backend, Standard, 2 GB) | Standard | $25 |
| Render Postgres (alerts DB) | Free | $0 |
| Vercel (frontend) | Hobby | $0 |
| **Total** | | **$25** |

Cheaper paths:
- **Rules-only mode + Render Starter ($7/mo):** total $7/mo, but you lose L3/L4/L5.
- **No backend at all:** if you only want to demo the UI screens, build the
  frontend with mocked data — Vercel hobby + `NEXT_PUBLIC_BACKEND_URL=` left
  unset gives you a no-cost static-style deployment that just won't have a
  live stream.
