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
    def bounded_payload(self) -> "TransactionIn":
        extras = self.__pydantic_extra__ or {}
        if len(extras) > MAX_FIELDS:
            raise ValueError(f"payload has {len(extras)} fields, limit is {MAX_FIELDS}")
        return self

    def to_features(self) -> dict[str, Any]:
        """Flatten back to the raw column names the feature pipeline expects."""
        return {**(self.__pydantic_extra__ or {}), **self.model_dump(exclude_none=True)}


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
