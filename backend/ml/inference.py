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


def format_value(value: float | None) -> str:
    """Render a feature value for a human reading an investigation queue.

    `%g` turns a six-digit card number into "9.062e+05", which tells an analyst
    nothing. Whole numbers print in full with separators; only genuine decimals
    get decimal places.
    """
    if value is None or np.isnan(value):
        return "missing"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


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
        return f"{self.feature} = {format_value(self.value)} {self.direction} risk"


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


def reasons_for_row(
    contribs: np.ndarray, row: pd.Series, columns: list[str]
) -> list[Reason]:
    """Top contributors for one row. The final element is the model bias, which
    is constant across transactions and therefore not evidence.

    The single definition of "which features are the reasons" - `score`,
    `score_batch` and the simulator feed all call this, so a transaction's
    stored evidence and its live evidence can never be selected differently.
    """
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
        reasons=reasons_for_row(contribs[0], X.iloc[0], spec.columns),
    )


def score_batch(
    df: pd.DataFrame, max_rows: int = 2_000_000, shap_top_n: int = 50
) -> list[RiskAssessment]:
    """Score every row of a DataFrame of transactions.

    The probability pass is vectorised and cheap, so every row is scored - the
    caller gets counts for the whole file, not a silent prefix. Exact TreeSHAP
    reasons are computed only for the `shap_top_n` riskiest rows, because a full
    contributions matrix is O(rows x trees x features) and does not scale; the
    remaining rows carry a single summary reason.

    `max_rows` is a memory rail, not a UI trim: above it the batch is rejected
    rather than quietly cut down to a length the totals would then misreport.
    """
    if len(df) == 0:
        return []
    if len(df) > max_rows:
        raise ValueError(
            f"{len(df):,} rows exceeds the {max_rows:,}-row limit for one batch"
        )

    model, spec = get_model()
    X = transform(df, spec)
    probs = model.predict_proba(X)[:, 1]

    # A list, not a set: this order lines each contributions row up with the
    # DataFrame row it explains, and a set only happens to preserve that.
    top = list(np.argsort(-probs)[: min(shap_top_n, len(df))])
    contribs_top = model.get_booster().predict(
        xgb.DMatrix(X.iloc[top], feature_names=spec.columns), pred_contribs=True
    )
    contrib_by_row = {row_pos: contribs_top[k] for k, row_pos in enumerate(top)}

    results = []
    for i in range(len(df)):
        probability = float(probs[i])
        level, recommendation = classify(probability)
        raw = df.iloc[i]

        tx_id = str(raw.get("TransactionID", raw.get("transaction_id", f"CSV_{i + 1}")))
        amt_val = raw.get("TransactionAmt", raw.get("amount", float("nan")))
        amt = float(amt_val) if pd.notna(amt_val) else 0.0

        if i in contrib_by_row:
            reasons = reasons_for_row(contrib_by_row[i], X.iloc[i], spec.columns)
        else:
            reasons = [
                Reason(
                    feature="risk_score",
                    value=probability,
                    contribution=probability,
                )
            ]

        results.append(
            RiskAssessment(
                transaction_id=tx_id,
                risk_score=round(probability, 6),
                risk_level=level,
                recommendation=recommendation,
                amount=amt,
                model_version=settings.model_version,
                reasons=reasons,
            )
        )
    return results


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
