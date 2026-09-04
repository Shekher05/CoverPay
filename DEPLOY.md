# Deploying CoverPay to Render

Three resources: a **PostgreSQL** database, a **Web Service** (FastAPI, native
Python), a **Static Site** (the Vite build). `render.yaml` in the repo root
defines all three; you can also create them by hand in the dashboard with the
settings below.

Native Python/Node runtimes - no Docker.

---

## 1. One-time: prepare the repo

The two model artifacts the API loads at runtime are now committed
(`models/fraud_xgb.json`, `models/feature_spec.json`, ~9 MB). Nothing else to do.
The 1.35 GB IEEE-CIS dataset is **not** deployed and is not needed at runtime -
only for retraining and for seeding (step 5).

Push to the branch Render will build (`main`).

---

## 2. Create the services

### Option A - Blueprint (recommended)

Render Dashboard -> **New** -> **Blueprint** -> pick this repo. Render reads
`render.yaml` and creates `coverpay-db`, `coverpay-api`, `coverpay-web`.

If a service name is already taken globally, Render appends a suffix. If that
happens, update the two cross-referenced URLs (`CORS_ORIGINS` on the API,
`VITE_API_BASE` on the web service) to the real hostnames and redeploy both.

### Option B - manual

| Resource | Setting | Value |
|---|---|---|
| **PostgreSQL** | Name | `coverpay-db` (free plan) |
| **Web Service** | Runtime | Python 3 |
| | Root directory | `backend` |
| | Build command | `pip install -e .` |
| | Start command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1` |
| | Health check path | `/health` |
| **Static Site** | Root directory | `frontend` |
| | Build command | `npm ci && npm run build` |
| | Publish directory | `dist` |
| | Redirect/Rewrite | `/*` -> `/index.html` (rewrite) |

`pip install -e .` (editable) is required: `config.py` finds `models/` and the
repo root from its own file location, so it must stay in `backend/`, not be
copied into `site-packages`.

`--workers 1` is required: the rate limiter and the model cache are per-process.

---

## 3. Environment variables (set in Render, never in code)

### Web Service (`coverpay-api`)

| Key | Value | Notes |
|---|---|---|
| `ENVIRONMENT` | `production` | turns off `/docs`, `/redoc`, `/openapi.json` |
| `DATABASE_URL` | *from `coverpay-db`* | Blueprint wires this; manual: paste the **Internal** connection string |
| `CORS_ORIGINS` | `https://coverpay-web.onrender.com` | your Static Site URL |
| `API_KEY` | *(generate a strong random value)* | gates `/transactions/score` and `/transactions/upload-csv` only |
| `OPENROUTER_API_KEY` | *(your key)* | **secret - set it here, never commit.** Get one at <https://openrouter.ai/keys> |
| `RATE_LIMIT_PER_MINUTE` | `30` | per-visitor cap on the assistant + uploads; raise if demos feel tight |

### Static Site (`coverpay-web`)

| Key | Value |
|---|---|
| `VITE_API_BASE` | `https://coverpay-api.onrender.com` (your Web Service URL) |

`VITE_API_BASE` is baked into the JS bundle at build time and is also added to
the page's `connect-src` CSP automatically. It is a URL, not a secret.

---

## 4. What is protected

| Endpoint | Public? | Guard |
|---|---|---|
| `GET /health`, `GET /dashboard/*`, `GET /transactions/{id}` | yes | none (read-only) |
| `POST /assistant/ask` | **yes** | per-visitor rate limit + the OpenRouter daily cap |
| `POST /transactions/score` | no | `X-API-Key` + rate limit |
| `POST /transactions/upload-csv` | no | `X-API-Key` + rate limit |

To use the scoring / CSV features from the deployed UI: open the app's **System**
tab and paste the `API_KEY` value (from the Render dashboard) into the X-API-Key
field. It is stored only in your browser's localStorage.

---

## 5. One-time: seed the database

A fresh Postgres is empty, so the dashboard and assistant have nothing to show.
Seeding runs the simulator, which needs the 1.35 GB dataset and the model - so
run it **from your machine**, pointed at Render's database:

```bash
cd backend
# "External Database URL" from the coverpay-db page in the Render dashboard
DATABASE_URL="postgresql://coverpay:...@...oregon-postgres.render.com/coverpay" \
  .venv/Scripts/python -m simulator.feed --rows 4000 --reset      # Windows
# DATABASE_URL="..." .venv/bin/python -m simulator.feed --rows 4000 --reset   # macOS/Linux
```

`--reset` drops and recreates the tables. Re-run any time you want fresh demo
data. Takes ~1-2 minutes for 4,000 rows.

---

## 6. Verify

1. `https://coverpay-api.onrender.com/health` -> `{"status":"ok","model_loaded":true,...}`
2. `https://coverpay-api.onrender.com/docs` -> **404** (docs off in production)
3. Open the Static Site URL -> dashboard shows transactions, timeline, incidents
4. Ask the assistant a question -> a grounded answer (needs `OPENROUTER_API_KEY`)
5. Browser devtools console -> no CSP violations

---

## Known limitations of the free tier

- **Cold starts.** A free Web Service sleeps after 15 min idle; the next request
  waits ~40-60s while it wakes and reloads the model. Upgrade the service to
  remove this.
- **Free Postgres expires after ~90 days.** Export or upgrade before then; if it
  lapses, re-run step 5 against the new database.
- **Keeping the assistant up under real traffic.** `minimax/minimax-m3:free` has
  an OpenRouter daily cap - 50 requests/day, or 1000/day once you have topped up
  $10 on OpenRouter. Past that the assistant returns 503. The per-visitor rate
  limit here bounds how fast any one visitor spends it; for a real launch, top up
  or point `OPENROUTER_MODEL` at a cheap paid model (the context is tiny, ~$0.002
  per answer).
- **Single worker.** Fine for a demo. Horizontal scaling needs a shared rate-limit
  store (Redis) and would benefit from a managed model cache.
