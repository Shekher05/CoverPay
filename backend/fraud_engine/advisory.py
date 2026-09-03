"""Phase 4 - advisory policy.

Deliberately separate from the model: the model answers "how suspicious is
this?", policy answers "what should the merchant do about it?". Thresholds are
config (`.env`), so they can be retuned for a merchant's risk appetite without
retraining anything.

Every recommendation is advisory. Nothing here blocks a payment.
"""
from __future__ import annotations

from enum import StrEnum

from config import settings


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Recommendation(StrEnum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"  # advisory only - the platform never stops a real payment


def classify(score: float) -> tuple[RiskLevel, Recommendation]:
    """Map a fraud probability onto a risk band and an advisory action."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be a probability in [0, 1], got {score}")

    if score >= settings.risk_block_threshold:
        return RiskLevel.HIGH, Recommendation.BLOCK
    if score >= settings.risk_review_threshold:
        return RiskLevel.MEDIUM, Recommendation.REVIEW
    return RiskLevel.LOW, Recommendation.ALLOW
