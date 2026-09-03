"""Phase 5 - the API contract.

A caller sends raw IEEE-CIS-shaped fields. Only the handful the pipeline cannot
work without are required and validated here; the remaining ~430 optional
columns ride along as extras and are picked up by name in `ml.features`.
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A real transaction has ~434 fields. Anything far beyond that is a malformed or
# hostile payload, not a payment.
MAX_FIELDS = 600


class TransactionIn(BaseModel):
    """One transaction submitted for scoring."""

    model_config = ConfigDict(extra="allow")

    TransactionAmt: float = Field(gt=0, description="Transaction value")
    TransactionDT: int = Field(
        ge=0, description="Seconds offset from the dataset epoch, not a unix timestamp"
    )
    TransactionID: int | str | None = Field(default=None)
    ProductCD: str | None = None

    @field_validator("TransactionAmt")
    @classmethod
    def finite_amount(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("TransactionAmt must be a finite number")
        return v

    @model_validator(mode="after")
    def bounded_payload(self) -> TransactionIn:
        extras = self.__pydantic_extra__ or {}
        if len(extras) > MAX_FIELDS:
            raise ValueError(f"payload has {len(extras)} fields, limit is {MAX_FIELDS}")
        return self

    def to_features(self) -> dict[str, Any]:
        """Flatten back to the raw column names the feature pipeline expects.

        `model_dump` on this `extra="allow"` model already returns the declared
        fields and the extra IEEE-CIS columns together; `exclude_none` drops the
        optional fields the caller omitted, which the pipeline treats as missing
        anyway.
        """
        return self.model_dump(exclude_none=True)


class ReasonOut(BaseModel):
    feature: str
    value: float | None
    contribution: float
    direction: str
    text: str


class AssessmentOut(BaseModel):
    """The response shape promised in the PRD."""

    transaction_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    amount_inr: float
    reasons: list[ReasonOut]
    model_version: str


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    detail: str | None = None


class AskIn(BaseModel):
    """A merchant question for the assistant."""

    question: str = Field(min_length=1, max_length=1000)
    merchant_id: str | None = Field(default=None, max_length=64)
    hours: float | None = Field(default=None, gt=0, le=24 * 90)


class AskOut(BaseModel):
    """The answer, plus the evidence it was grounded on.

    `context` is returned deliberately: every figure in `answer` should be
    checkable against it, so a merchant is never asked to trust prose alone.
    """

    question: str
    answer: str
    context: dict
    model: str


class BatchAnalysisOut(BaseModel):
    filename: str
    total_transactions: int
    possible_attacks_flagged: int
    review_recommended: int
    block_recommended: int
    high_risk_percentage: float
    total_value_flagged: float
    riskiest_transactions: list[AssessmentOut]


# --- Dashboard read models -------------------------------------------------
# The dashboard endpoints are read-only projections of the database. Typing
# their responses keeps the OpenAPI schema honest and makes a drift between
# the API and the frontend a test failure rather than a runtime surprise.


class _Breakdown(BaseModel):
    count: int
    value: float


class _EngineFlags(BaseModel):
    model: int
    behaviour: int
    incident: int


class _Window(BaseModel):
    first_dt: int | None
    last_dt: int | None
    hours: float


class SummaryOut(BaseModel):
    transactions: int
    total_value: float
    flagged: int
    flagged_value: float
    by_recommendation: dict[str, _Breakdown]
    incidents: dict[str, int]
    incidents_open: int
    engine_flags: _EngineFlags
    window: _Window


class TimelineBucket(BaseModel):
    bucket: int
    start_dt: int
    hours: float
    transactions: int
    value: float
    flagged_value: float
    flagged: int


class TimelineOut(BaseModel):
    bucket_seconds: int
    first_dt: int | None
    buckets: list[TimelineBucket]


class IncidentOut(BaseModel):
    incident_id: str
    merchant_id: str
    status: str
    severity: str
    opened_dt: int
    last_dt: int
    duration_seconds: int
    transactions: int
    customers: int
    devices: int
    total_value: float
    risky_value: float


class FeedTransactionOut(BaseModel):
    transaction_id: str
    amount: float
    transaction_dt: int
    product_cd: str | None
    customer_id: str | None
    merchant_id: str | None
    device_id: str | None
    risk_score: float
    risk_level: str
    recommendation: str
    behaviour_score: float | None
    behaviour_detail: str | None
    incident_id: str | None
    model_version: str
    reasons: list[ReasonOut]

