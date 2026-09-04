"""Shared settings. Reads .env; every phase imports from here."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The Python project lives in backend/, but the dataset, model artifacts, docs
# and generated data are shared with the frontend and belong to the repository,
# so they sit one level up. Two roots, each with one job.
PACKAGE_ROOT = Path(__file__).parent  # backend/
ROOT = PACKAGE_ROOT.parent  # repository root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    data_dir: Path = Path("ieee-fraud-detection")
    database_url: str = "sqlite:///./coverpay.db"
    model_path: Path = Path("models/fraud_xgb.json")
    model_version: str = "v1"

    # "development" locally and in tests; set ENVIRONMENT=production on Render so
    # the interactive API docs (/docs, /redoc, /openapi.json) are turned off.
    environment: str = "development"

    # Browser origins allowed to call the API, comma-separated. The default is
    # the Vite dev server. On Render set CORS_ORIGINS to the Static Site URL
    # (e.g. https://coverpay.onrender.com). Only matters when the frontend calls
    # the API cross-origin - i.e. when VITE_API_BASE points straight at the
    # Web Service rather than a same-origin rewrite.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    risk_review_threshold: float = 0.30
    risk_block_threshold: float = 0.70
    random_seed: int = 42

    # --- Cost model ---
    # Assumptions, not measurements. They exist so a threshold can be argued
    # about in rupees instead of in F1, and so the argument is explicit rather
    # than buried in someone's head. Every one is a merchant policy input:
    # change them in .env and `python -m ml.train` re-derives the cost-optimal
    # thresholds against the new numbers.
    chargeback_fee: float = 750.0  # fixed admin cost of a chargeback, on top of the lost amount
    review_cost: float = 40.0  # analyst cost to clear one flagged transaction
    block_friction_rate: float = 0.25  # share of a wrongly BLOCKed legit amount that walks away

    # The AI merchant helper talks to OpenRouter (OpenAI-compatible). Only this
    # feature needs a key; everything else runs without one. On Render set
    # OPENROUTER_API_KEY in the dashboard - never commit it.
    openrouter_api_key: str | None = None
    openrouter_model: str = "minimax/minimax-m3:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- API access control ---
    # Optional. When unset (the default) the API is open, which keeps local
    # development and the test suite friction-free. Set it in .env for any
    # deployment and every mutating / model-spending endpoint then requires the
    # `X-API-Key` header. This is the minimum gate, not an identity system:
    # it does not map a caller to a merchant, so it does not by itself make
    # `merchant_id` filtering trustworthy.
    api_key: str | None = None

    # Fixed-window rate limit for the expensive endpoints (assistant, CSV
    # upload), counted per API key or, failing that, per client IP (the
    # left-most X-Forwarded-For entry behind a proxy, else the socket peer).
    # ponytail: in-process fixed window; resets on restart and is not shared
    # across workers. Move to a shared store (Redis) or a gateway limiter if
    # the service is ever run multi-process.
    rate_limit_per_minute: int = 30

    @property
    def raw(self) -> Path:
        """Absolute path to the IEEE-CIS CSV directory."""
        if self.data_dir.is_absolute():
            return self.data_dir
        return (ROOT / self.data_dir).resolve()

    @property
    def cors_origins_list(self) -> list[str]:
        """`cors_origins` split into a clean list for CORSMiddleware."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def database(self) -> str:
        """Usable SQLAlchemy URL.

        - `sqlite:///./coverpay.db` is resolved against the working directory, so
          running the API from backend/ and a script from the repo root would
          quietly open two different databases. Anchoring to ROOT removes the trap.
        - Render (like Heroku) hands out `postgres://...`, a scheme SQLAlchemy 2.x
          dropped. Rewrite it to `postgresql://` so `DATABASE_URL` from a Render
          PostgreSQL add-on works with no manual editing.
        """
        url = self.database_url
        if url.startswith("postgres://"):
            return "postgresql://" + url[len("postgres://") :]

        prefix = "sqlite:///./"
        if not url.startswith(prefix):
            return url
        return f"sqlite:///{(ROOT / url[len(prefix):]).as_posix()}"


settings = Settings()
