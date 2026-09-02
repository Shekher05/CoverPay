"""Phase 5 - the transaction scoring API.

    uvicorn backend.main:app --reload

Every recommendation is advisory. This service never blocks a payment.
"""
from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import dashboard
from backend.database import Prediction, Transaction, get_session, init_db
from backend.schemas import AssessmentOut, HealthOut, ReasonOut, TransactionIn
from config import settings
from ml.inference import ModelUnavailable, get_model, score

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables and warm the model once, so the first request is not the
    one that pays the load cost - or discovers the artifacts are missing."""
    init_db()
    try:
        get_model()
        log.info("Model %s loaded", settings.model_version)
    except ModelUnavailable as exc:
        # Start anyway: /health must be able to report the problem.
        log.error("Model unavailable: %s", exc)
    yield


app = FastAPI(
    title="CoverPay Fraud Intelligence",
    description="Advisory transaction fraud scoring. Does not block payments.",
    version=settings.model_version,
    lifespan=lifespan,
)

# The Vite dev server runs on its own origin. Localhost only, and explicitly
# listed rather than "*", so this cannot quietly become a production hole.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(dashboard.router)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Log the detail, return none of it - internals are not the caller's."""
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error"})


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    try:
        get_model()
    except ModelUnavailable as exc:
        return HealthOut(
            status="degraded",
            model_loaded=False,
            model_version=settings.model_version,
            detail=str(exc),
        )
    return HealthOut(
        status="ok", model_loaded=True, model_version=settings.model_version
    )


def _to_response(assessment, transaction_id: str) -> AssessmentOut:
    return AssessmentOut(
        transaction_id=transaction_id,
        risk_score=assessment.risk_score,
        risk_level=str(assessment.risk_level),
        recommendation=str(assessment.recommendation),
        amount_inr=assessment.amount,
        model_version=assessment.model_version,
        reasons=[
            ReasonOut(
                feature=r.feature,
                # NaN is not valid JSON; a missing feature reports as null.
                value=None if r.value is None or math.isnan(r.value) else r.value,
                contribution=r.contribution,
                direction=r.direction,
                text=r.text,
            )
            for r in assessment.reasons
        ],
    )


@app.post("/transactions/score", response_model=AssessmentOut)
def score_transaction(
    payload: TransactionIn, session: Session = Depends(get_session)
) -> AssessmentOut:
    """Score one transaction, store the result, return the assessment."""
    try:
        assessment = score(payload.to_features())
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    transaction_id = (
        str(payload.TransactionID) if payload.TransactionID is not None else "unknown"
    )

    transaction = Transaction(
        transaction_id=transaction_id,
        amount=payload.TransactionAmt,
        transaction_dt=payload.TransactionDT,
        product_cd=payload.ProductCD,
    )
    session.add(transaction)
    session.flush()  # assigns transaction.id without ending the transaction

    response = _to_response(assessment, transaction_id)
    session.add(
        Prediction(
            transaction_pk=transaction.id,
            risk_score=response.risk_score,
            risk_level=response.risk_level,
            recommendation=response.recommendation,
            model_version=response.model_version,
            reasons=[r.model_dump() for r in response.reasons],
        )
    )
    session.commit()
    return response


@app.get("/transactions/{transaction_id}", response_model=AssessmentOut)
def get_assessment(
    transaction_id: str, session: Session = Depends(get_session)
) -> AssessmentOut:
    """Most recent stored assessment for a transaction."""
    row = session.execute(
        select(Transaction, Prediction)
        .join(Prediction, Prediction.transaction_pk == Transaction.id)
        .where(Transaction.transaction_id == transaction_id)
        .order_by(Prediction.created_at.desc())
        .limit(1)
    ).first()

    if row is None:
        raise HTTPException(
            status_code=404, detail=f"No assessment for {transaction_id}"
        )

    transaction, prediction = row
    return AssessmentOut(
        transaction_id=transaction.transaction_id,
        risk_score=prediction.risk_score,
        risk_level=prediction.risk_level,
        recommendation=prediction.recommendation,
        amount_inr=transaction.amount,
        model_version=prediction.model_version,
        reasons=[ReasonOut(**r) for r in prediction.reasons],
    )
