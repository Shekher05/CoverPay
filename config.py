"""Shared settings. Reads .env; every phase imports from here."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("ieee-fraud-detection")
    database_url: str = "sqlite:///./coverpay.db"
    model_path: Path = Path("models/fraud_xgb.json")
    model_version: str = "v1"
    risk_review_threshold: float = 0.30
    risk_block_threshold: float = 0.70
    random_seed: int = 42

    @property
    def raw(self) -> Path:
        """Absolute path to the IEEE-CIS CSV directory."""
        return (ROOT / self.data_dir).resolve() if not self.data_dir.is_absolute() else self.data_dir


settings = Settings()
