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

    # Only the AI merchant helper needs this. Read from .env rather than the
    # process environment, so the key never has to be exported by hand.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-latest"

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
    def database(self) -> str:
        """Connection URL with a relative SQLite file anchored to the repo root.

        `sqlite:///./coverpay.db` is resolved against the working directory, so
        running the API from backend/ and a script from the repo root would
        quietly open two different databases. Anchoring removes the trap.
        """
        prefix = "sqlite:///./"
        if not self.database_url.startswith(prefix):
            return self.database_url
        return f"sqlite:///{(ROOT / self.database_url[len(prefix):]).as_posix()}"


settings = Settings()
