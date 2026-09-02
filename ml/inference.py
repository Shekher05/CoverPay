"""Phase 4 - load the trained model once and score transactions.

Evidence comes from XGBoost's own `pred_contribs` (exact TreeSHAP), so a score
never arrives without the reasons behind it and no extra dependency is needed.

    python -m ml.inference        # scores one sample row end to end
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb

from config import ROOT, settings
from fraud_engine.advisory import Recommendation, RiskLevel, classify
from ml.features import FeatureSpec, transform
from ml.train import SPEC_PATH

MODEL_PATH = ROOT / settings.model_path
TOP_REASONS = 5


class ModelUnavailable(RuntimeError):
    """Raised when the artifacts are missing. The API turns this into a 503
    rather than scoring with a silently untrained model."""


@dataclass
class Reason:
    feature: str
    value: float | None
    contribution: float

    @property
    def direction(self) -> str:
        return "increases" if self.contribution > 0 else "decreases"

    @property
    def text(self) -> str:
        missing = self.value is None or np.isnan(self.value)
        shown = "missing" if missing else f"{self.value:,.4g}"
        return f"{self.feature} = {shown} {self.direction} risk"


@dataclass
class RiskAssessment:
    transaction_id: str
    risk_score: float
    risk_level: RiskLevel
    recommendation: Recommendation
    amount: float
    model_version: str
    reasons: list[Reason] = field(default_factory=list)


@lru_cache(maxsize=1)
def get_model() -> tuple[xgb.XGBClassifier, FeatureSpec]:
    """Load booster + spec once per process. Reloading per request would
    dominate latency, so the result is cached."""
    if not MODEL_PATH.exists() or not SPEC_PATH.exists():
        raise ModelUnavailable(
            f"Missing {MODEL_PATH.name} or {SPEC_PATH.name} in {MODEL_PATH.parent}. "
            "Run `python -m ml.train` first."
        )
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model, FeatureSpec.load(SPEC_PATH)


def _reasons(contribs: np.ndarray, row: pd.Series, columns: list[str]) -> list[Reason]:
    """Top contributors for one row. The final element is the model bias, which
    is constant across transactions and therefore not evidence."""
    per_feature = contribs[:-1]
    top = np.argsort(-np.abs(per_feature))[:TOP_REASONS]
    return [
        Reason(
            feature=columns[i],
            value=float(row.iloc[i]),
            contribution=float(per_feature[i]),
        )
        for i in top
    ]


def score(transaction: dict | pd.DataFrame) -> RiskAssessment:
    """Score one transaction. Same feature path as training, by construction."""
    model, spec = get_model()
    df = pd.DataFrame([transaction]) if isinstance(transaction, dict) else transaction
    if len(df) != 1:
        raise ValueError(f"score() handles one transaction, got {len(df)}")

    X = transform(df, spec)
    probability = float(model.predict_proba(X)[0, 1])
    level, recommendation = classify(probability)

    contribs = model.get_booster().predict(
        xgb.DMatrix(X, feature_names=spec.columns), pred_contribs=True
    )

    raw = df.iloc[0]
    return RiskAssessment(
        transaction_id=str(raw.get("TransactionID", "unknown")),
        risk_score=round(probability, 6),
        risk_level=level,
        recommendation=recommendation,
        amount=float(raw.get("TransactionAmt", float("nan"))),
        model_version=settings.model_version,
        reasons=_reasons(contribs[0], X.iloc[0], spec.columns),
    )


def main() -> None:
    """Smoke test: score the fraud the model is most confident about.

    Also checks that the single-row path agrees with the batch path on real
    data, which is the one way this pipeline can fail silently.
    """
    from ml.features import load_raw, split

    model, spec = get_model()
    df = load_raw()
    _, val = split(df, spec.split_dt)
    frauds = val[val["isFraud"] == 1]

    batch_scores = model.predict_proba(transform(frauds, spec))[:, 1]
    worst = frauds.iloc[[int(np.argmax(batch_scores))]]

    result = score(worst)
    assert abs(result.risk_score - batch_scores.max()) < 1e-6, "batch/single mismatch"
    print(f"transaction    {result.transaction_id}")
    print(f"risk_score     {result.risk_score}")
    print(f"risk_level     {result.risk_level}")
    print(f"recommendation {result.recommendation}")
    print(f"amount         {result.amount:,.2f}")
    print(f"model_version  {result.model_version}")
    print("reasons:")
    for reason in result.reasons:
        print(f"  - {reason.text} ({reason.contribution:+.3f})")


if __name__ == "__main__":
    main()
