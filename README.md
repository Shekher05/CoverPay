# CoverPay

Defensive fraud intelligence for an Indian payment gateway. Scores a transaction,
explains why, spots behaviour that is unusual for a customer, detects coordinated
attacks on a merchant, and answers merchant questions in plain language.

Every recommendation is **advisory**. The platform never blocks a payment.

---

## What you need to provide

Short version: **one API key**. Everything else is already in place or generated.

| # | Item | Needed for | Status |
|---|---|---|---|
| 1 | IEEE-CIS dataset | training, simulator templates | **already present** in `ieee-fraud-detection/` |
| 2 | `OPENROUTER_API_KEY` | the AI merchant helper only | **you need to supply this** |
| 3 | Python 3.11+ / Node 18+ | everything | already installed here |

Nothing else. No cloud account, no Razorpay credentials, no database server, no
payment gateway access. Razorpay integration is explicitly future work and is not
required for any feature.

### 1. The dataset (already done)

`ieee-fraud-detection/` must contain these two files. The other Kaggle files are
unused: `test_*.csv` is unlabelled, so it cannot support honest evaluation.

```
ieee-fraud-detection/
  train_transaction.csv    652 MB, 590,540 rows x 394 columns
  train_identity.csv        25 MB, 144,233 rows x  41 columns
```

Source: <https://www.kaggle.com/c/ieee-fraud-detection/data> (free Kaggle account).
It is git-ignored and never committed.

If you move it, set `DATA_DIR` in `.env` to the new location.

### 2. The AI API key (the only missing piece)

Only the AI merchant helper needs this. Without it **every other feature works**;
the assistant returns HTTP 503 with a clear message rather than inventing an answer.

The assistant calls [OpenRouter](https://openrouter.ai/keys) (OpenAI-compatible).
Generate a key and put it in `.env` at the repository root:

```bash
cp .env.example .env
# then edit .env and set:
OPENROUTER_API_KEY=sk-or-v1-...
```


**Do not commit `.env`.** It is git-ignored; `.env.example` is the committed
template and holds no secrets.

### Optional, not required

| Setting | Default | When to change it |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./coverpay.db` | Point at PostgreSQL for concurrent writes. The ORM code is unchanged. |
| `RISK_REVIEW_THRESHOLD` | `0.30` | Merchant risk appetite. Lower catches more fraud and more false positives. |
| `RISK_BLOCK_THRESHOLD` | `0.70` | As above, for the BLOCK band. |
| `CHARGEBACK_FEE` | `750` | Admin cost of a chargeback, on top of the lost amount. |
| `REVIEW_COST` | `40` | Analyst cost to clear one flagged transaction. |
| `BLOCK_FRICTION_RATE` | `0.25` | Share of a wrongly BLOCKed legitimate amount that walks away. |
| `RANDOM_SEED` | `42` | Reproducibility of training and simulation. |
| `MODEL_VERSION` | `v1` | Stamped onto every stored prediction. |

Thresholds are **policy, not model**. Changing them re-tunes the precision and
recall trade-off with no retraining.

The three cost inputs are **assumptions, not measurements**. They exist so a
threshold can be argued about in rupees rather than in F1: `python -m ml.train`
sweeps every threshold pair against them on the validation set and reports the
cost-optimal pair in `docs/metrics.md`, next to what the configured pair costs.
Change them to match a real merchant and the recommendation changes with them.

---

## Layout

```
CoverPay/
├── backend/                  Python: everything that thinks
│   ├── api/                  FastAPI - routes, DB models, schemas, assistant
│   ├── ml/                   audit, feature pipeline, training, inference
│   ├── fraud_engine/         advisory policy, behaviour engine, incident engine
│   ├── simulator/            synthetic transactions, entity world, DB feed
│   ├── tests/                90 tests
│   ├── config.py             settings, reads ../.env
│   └── pyproject.toml
├── frontend/                 React + Vite monitoring dashboard
├── data/synthetic/           generated streams (git-ignored)
├── models/                   trained artifacts (git-ignored)
├── docs/                     audit, metrics, engine evaluations, PRD, design
├── ieee-fraud-detection/     the dataset (git-ignored)
└── .env                      your secrets (git-ignored)
```

Why `api/` sits inside `backend/`: the ML, fraud-engine and simulator packages are
not web code and are used from the command line without the API running, so the
HTTP layer is one package among four rather than the whole backend.

---

## Setup

```bash
# 1. Backend
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"          # Windows
# source .venv/bin/activate && pip install -e ".[dev]"   # macOS/Linux

# 2. Frontend
cd ../frontend
npm install

# 3. Secrets
cd ..
cp .env.example .env        # then add OPENROUTER_API_KEY if you want the assistant
```

---

## Running it

The model artifacts are already trained and present in `models/`. To rebuild them
from scratch run steps 1 and 2; otherwise skip to step 3.

```bash
cd backend

# 1. Audit the dataset         -> docs/audit.md, ml/feature_columns.json   (~3 min)
.venv/Scripts/python -m ml.audit

# 2. Train and evaluate        -> models/, docs/metrics.md                 (~10 min)
.venv/Scripts/python -m ml.train

# 3. Fill the database with a simulated stream through every engine        (~2 min)
.venv/Scripts/python -m simulator.feed --rows 4000 --reset

# 4. Serve the API on :8000
.venv/Scripts/python -m uvicorn api.main:app --port 8000

# 5. In a second terminal, the dashboard on :5173
cd ../frontend && npm run dev
```

Open <http://localhost:5173>.

### Everything else

```bash
cd backend

.venv/Scripts/python -m pytest tests/ -q            # 90 tests
.venv/Scripts/python -m ml.inference                # score one real fraud row
.venv/Scripts/python -m fraud_engine.behavioural    # -> docs/behaviour.md
.venv/Scripts/python -m fraud_engine.incidents      # -> docs/incidents.md
.venv/Scripts/python -m simulator.world --rows 5000 --mode evaluation --score
.venv/Scripts/python -m api.assistant "why did my fraud risk increase?"
.venv/Scripts/python -m api.assistant "..." --context-only   # no API key needed
```

`--context-only` prints the exact evidence the assistant would be given, without
calling the model. Useful for checking grounding, and free.

---

## The API

| Endpoint | Purpose |
|---|---|
| `GET /health` | model loaded, version |
| `POST /transactions/score` | score one transaction, store it, return evidence |
| `GET /transactions/{id}` | the stored assessment |
| `GET /dashboard/summary` | volume, advisory mix, value at risk, open incidents |
| `GET /dashboard/timeline` | bucketed volume and flagged traffic |
| `GET /dashboard/incidents` | detected spikes, most severe first |
| `GET /dashboard/transactions` | live feed with all three engines' verdicts |
| `POST /assistant/ask` | plain-language explanation, grounded in stored data |

Interactive docs at <http://localhost:8000/docs>.

---

## How it fits together

```
IEEE-CIS -> audit -> one feature pipeline -> XGBoost -> advisory policy
                                                 |
simulator -> model -> behaviour engine -> incident engine -> database
                                                 |
                            FastAPI -> React dashboard + AI merchant helper
```

Four components, four different questions:

| Component | Question it answers |
|---|---|
| Fraud model | How suspicious is this transaction on its own? |
| Behaviour engine | Is this unusual **for this customer**? |
| Incident engine | Is a coordinated attack hitting **this merchant**? |
| AI helper | What does all of that mean, in words? |

They are not interchangeable, and the measurements show it: on the synthetic
evaluation the incident engine catches 100% of the coordinated spikes that the
behaviour engine sees 8% of, and misses 100% of the account takeovers that only
the behaviour engine sees.

---

## Honest limitations

Read these before showing the project to anyone.

- **Model performance is PR-AUC 0.47 / ROC-AUC 0.87** on a held-out test set
  (`docs/metrics.md`). Two things make that number smaller than it used to be and
  both are deliberate. The split is chronological, which is harder and more honest
  than the random splits most published IEEE-CIS numbers use. And it is now
  three-way, 60/20/20: validation pays for early stopping and for threshold
  selection, test is untouched until both are frozen, and every headline figure
  comes from test. The same pipeline with a two-way split reported PR-AUC 0.58 on
  exactly these rows, because the rows that stopped training were also the rows
  that graded it. Part of the 0.58 to 0.47 drop is that bias being deleted, and
  part is the real price of holding a third set back: 354,324 training rows
  instead of 472,432, ending 40 days further from the test window.
- **The configured thresholds are no longer cost-optimal for this model.** The
  retrained scores sit lower, so at `RISK_REVIEW_THRESHOLD=0.30` precision is
  0.21 and recall is 0.64: roughly four in five reviewed transactions are
  legitimate. The cost sweep prefers `0.25` / `0.95`. They are left at `0.30` /
  `0.70` because thresholds are a merchant decision, and `docs/metrics.md` prices
  both so the decision can be made on numbers.
- **Sparse API payloads score confidently wrong.** The model reads ~270 features.
  Send 13 fields and the rest become missing, and the same transaction that scores
  1.0 complete scores 0.03. A production deployment needs a required-field floor
  or a completeness score on the response.
- **Synthetic data cannot support a detection claim.** The simulator resamples
  real rows for the 380 wide columns it cannot invent, so the model partly reacts
  to the borrowed template rather than the scenario. Its purpose is to exercise
  the pipeline and to feed the behaviour and incident engines. `docs/metrics.md`
  is the only real performance number.
- **No authentication on the API.** Fine for local use, not for exposure.
- **`TransactionDT` is a seconds offset, not a timestamp.** The dataset has no
  calendar origin, so the UI shows elapsed time and never a date. Do not "fix"
  this by inventing an epoch.
- **IEEE-CIS is IEEE-CIS.** It is not Razorpay data and must never be presented
  as data from any real payment gateway.
