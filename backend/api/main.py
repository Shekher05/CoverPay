"""Phase 5 - the transaction scoring API.

    uvicorn api.main:app --reload

Every recommendation is advisory. This service never blocks a payment.
"""
from __future__ import annotations

import io
import logging
import math
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from api import dashboard
from api.assistant import AssistantUnavailable, ask
from api.auth import rate_limit, require_api_key
from api.database import Prediction, Transaction, get_session, init_db
from api.schemas import (
    AskIn,
    AskOut,
    AssessmentOut,
    BatchAnalysisOut,
    HealthOut,
    ReasonOut,
    TransactionIn,
)
from config import settings
from ml.inference import ModelUnavailable, RiskAssessment, get_model, score, score_batch

log = logging.getLogger(__name__)

# A CSV upload is read fully into memory before parsing, so it needs a hard
# ceiling. 10 MB is well above any demo batch and far below what would pressure
# a small container. The row cap is a second rail: a very narrow CSV can pack a
# lot of rows into 10 MB, and score_batch's own limit would surface as a
# confusing 422 rather than "too large".
MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_CSV_ROWS = 200_000


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
    # X-API-Key so a browser client can authenticate once API_KEY is set; the
    # preflight would otherwise strip it.
    allow_headers=["Content-Type", "X-API-Key"],
)

app.include_router(dashboard.router)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Baseline hardening headers on every response. The API returns only JSON,
    so it is never a framing or script-execution surface itself - these matter
    for the error pages, /docs, and defence in depth. A strict CSP is set on the
    frontend document (frontend/index.html), not here, since /docs needs its own.
    """
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    return response


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


def _to_response(assessment: RiskAssessment, transaction_id: str) -> AssessmentOut:
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


@app.post(
    "/transactions/score",
    response_model=AssessmentOut,
    dependencies=[Depends(require_api_key)],
)
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


@app.post(
    "/assistant/ask",
    response_model=AskOut,
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
def assistant_ask(payload: AskIn, session: Session = Depends(get_session)) -> AskOut:
    """Explain stored fraud evidence in plain language.

    The assistant never scores anything. It reads what the three engines already
    decided and writes prose about it, and the evidence it used comes back in
    `context` so the answer can be checked rather than trusted.
    """
    try:
        result = ask(payload.question, session, payload.merchant_id, payload.hours)
    except AssistantUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskOut(
        question=result.question,
        answer=result.answer,
        context=result.context,
        model=result.model,
    )


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


@app.post(
    "/transactions/upload-csv",
    response_model=BatchAnalysisOut,
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
def upload_csv_batch(file: UploadFile = File(...)) -> BatchAnalysisOut:
    """Score every row of an uploaded transaction CSV and summarise the risk.

    Sync `def` on purpose: CSV parsing and batch inference are CPU-bound and
    blocking, so FastAPI runs this in its worker threadpool rather than on the
    event loop - the same reason `/transactions/score` is sync.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv file")

    # Reject an oversized upload before pulling it all into memory. `file.size`
    # comes from the multipart part; the capped read is the real guard when it
    # is absent or understated.
    if file.size is not None and file.size > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="CSV exceeds the 10 MB limit")
    content = file.file.read(MAX_CSV_BYTES + 1)
    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="CSV exceeds the 10 MB limit")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ValueError,
        UnicodeDecodeError,
    ) as exc:
        # The real parser error names the caller's file contents; log it, do not
        # return it.
        log.info("Rejected CSV upload %r: %s", file.filename, exc)
        raise HTTPException(
            status_code=400, detail="Could not parse the file as CSV"
        ) from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file has no rows")
    if len(df) > MAX_CSV_ROWS:
        raise HTTPException(
            status_code=413, detail=f"CSV has more than {MAX_CSV_ROWS:,} rows"
        )

    try:
        assessments = score_batch(df)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        # A CSV missing the columns the feature pipeline needs is the caller's
        # to fix, but the exception text can echo their data - keep it in the log.
        log.info("CSV upload %r could not be scored: %s", file.filename, exc)
        raise HTTPException(
            status_code=422, detail="CSV is missing columns the model needs"
        ) from exc

    total = len(assessments)
    flagged = [a for a in assessments if a.recommendation != "ALLOW"]
    reviews = [a for a in assessments if a.recommendation == "REVIEW"]
    blocks = [a for a in assessments if a.recommendation == "BLOCK"]

    total_flagged_val = sum(a.amount for a in flagged if not math.isnan(a.amount))
    high_risk_pct = round((len(flagged) / total) * 100, 1) if total > 0 else 0.0

    sorted_riskiest = sorted(assessments, key=lambda a: a.risk_score, reverse=True)[:10]

    return BatchAnalysisOut(
        filename=filename,
        total_transactions=total,
        possible_attacks_flagged=len(flagged),
        review_recommended=len(reviews),
        block_recommended=len(blocks),
        high_risk_percentage=high_risk_pct,
        total_value_flagged=round(total_flagged_val, 2),
        riskiest_transactions=[_to_response(a, a.transaction_id) for a in sorted_riskiest],
    )

