"""Phase 13 - read-only endpoints behind the monitoring dashboard.

Reads the database and nothing else: no model, no simulator. The dashboard
therefore behaves identically whether the rows came from the simulator or from
real scored traffic.

On time: `transaction_dt` is a seconds offset with no real calendar origin, so
nothing here pretends to know a wall-clock date. Buckets are integer-second
arithmetic and the API returns offsets plus hours elapsed, leaving presentation
to the client.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Integer, case, func, select
from sqlalchemy.orm import Session

from api.database import IncidentRecord, Prediction, Transaction, get_session
from api.schemas import FeedTransactionOut, IncidentOut, SummaryOut, TimelineOut

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

SECONDS_PER_HOUR = 3600
BEHAVIOUR_FLAG_THRESHOLD = 0.3

# Severity order for SQL sorting. Applied before LIMIT so a high-severity
# incident just outside a recent-time page is not dropped from the list.
_SEVERITY_RANK = case(
    (IncidentRecord.severity == "CRITICAL", 0),
    (IncidentRecord.severity == "HIGH", 1),
    (IncidentRecord.severity == "MEDIUM", 2),
    (IncidentRecord.severity == "LOW", 3),
    else_=9,
)


@router.get("/summary", response_model=SummaryOut)
def summary(session: Session = Depends(get_session)) -> dict:
    """Headline numbers: volume, advisory mix, money at risk, open incidents."""
    count, total_value, first_dt, last_dt = session.execute(
        select(
            func.count(Transaction.id),
            func.coalesce(func.sum(Transaction.amount), 0.0),
            func.min(Transaction.transaction_dt),
            func.max(Transaction.transaction_dt),
        )
    ).one()

    by_recommendation = {
        name: {"count": n, "value": round(value or 0.0, 2)}
        for name, n, value in session.execute(
            select(
                Prediction.recommendation,
                func.count(Prediction.id),
                func.sum(Transaction.amount),
            )
            .join(Transaction, Transaction.id == Prediction.transaction_pk)
            .group_by(Prediction.recommendation)
        )
    }
    flagged = {k: v for k, v in by_recommendation.items() if k != "ALLOW"}
    flagged_count = sum(v["count"] for v in flagged.values())
    flagged_value = sum(v["value"] for v in flagged.values())

    incidents = {
        status: n
        for status, n in session.execute(
            select(
                IncidentRecord.status, func.count(IncidentRecord.incident_id)
            ).group_by(IncidentRecord.status)
        )
    }

    behaviour_flagged = session.execute(
        select(func.count(Prediction.id)).where(
            Prediction.behaviour_score >= BEHAVIOUR_FLAG_THRESHOLD
        )
    ).scalar_one()
    in_incident = session.execute(
        select(func.count(Prediction.id)).where(Prediction.incident_id.is_not(None))
    ).scalar_one()

    return {
        "transactions": count,
        "total_value": round(total_value or 0.0, 2),
        "flagged": flagged_count,
        "flagged_value": round(flagged_value, 2),
        "by_recommendation": by_recommendation,
        "incidents": incidents,
        "incidents_open": incidents.get("ACTIVE", 0) + incidents.get("SUSPICIOUS", 0),
        # How many transactions each engine flags. The point of the three-engine
        # design is that these sets overlap far less than people expect.
        "engine_flags": {
            "model": flagged_count,
            "behaviour": behaviour_flagged,
            "incident": in_incident,
        },
        "window": {
            "first_dt": first_dt,
            "last_dt": last_dt,
            "hours": round(((last_dt or 0) - (first_dt or 0)) / SECONDS_PER_HOUR, 2),
        },
    }


@router.get("/timeline", response_model=TimelineOut)
def timeline(
    buckets: int = Query(48, ge=4, le=240), session: Session = Depends(get_session)
) -> dict:
    """Volume and flagged value over equal-width time buckets."""
    first_dt, last_dt = session.execute(
        select(
            func.min(Transaction.transaction_dt), func.max(Transaction.transaction_dt)
        )
    ).one()
    if first_dt is None:
        return {"buckets": [], "bucket_seconds": 0, "first_dt": None}

    width = max(max(last_dt - first_dt, 1) // buckets, 1)

    # Bucketing in SQL keeps the series server-side instead of shipping every
    # row to the client to be counted there.
    bucket = ((Transaction.transaction_dt - first_dt) / width).cast(Integer)
    rows = session.execute(
        select(
            bucket.label("bucket"),
            func.count(Transaction.id),
            func.sum(Transaction.amount),
            func.sum(
                case((Prediction.recommendation != "ALLOW", Transaction.amount), else_=0.0)
            ),
            func.sum(case((Prediction.recommendation != "ALLOW", 1), else_=0)),
        )
        .join(Prediction, Prediction.transaction_pk == Transaction.id)
        .group_by("bucket")
        .order_by("bucket")
    ).all()

    return {
        "bucket_seconds": width,
        "first_dt": first_dt,
        "buckets": [
            {
                "bucket": int(b),
                "start_dt": first_dt + int(b) * width,
                "hours": round(int(b) * width / SECONDS_PER_HOUR, 3),
                "transactions": int(n),
                "value": round(value or 0.0, 2),
                "flagged_value": round(flagged_value or 0.0, 2),
                "flagged": int(flagged or 0),
            }
            for b, n, value, flagged_value, flagged in rows
        ],
    }


@router.get("/incidents", response_model=list[IncidentOut])
def incidents(
    limit: int = Query(50, ge=1, le=500), session: Session = Depends(get_session)
) -> list[dict]:
    """Incidents, most severe first, then most recent.

    Ordered in SQL so the `limit` keeps the most severe incidents, not just the
    most recent ones that happen to also be severe.
    """
    rows = (
        session.execute(
            select(IncidentRecord)
            .order_by(_SEVERITY_RANK, IncidentRecord.opened_dt.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )

    return [
        {
            "incident_id": r.incident_id,
            "merchant_id": r.merchant_id,
            "status": r.status,
            "severity": r.severity,
            "opened_dt": r.opened_dt,
            "last_dt": r.last_dt,
            "duration_seconds": r.last_dt - r.opened_dt,
            "transactions": r.transactions,
            "customers": r.customers,
            "devices": r.devices,
            "total_value": r.total_value,
            "risky_value": r.risky_value,
        }
        for r in rows
    ]


@router.get("/transactions", response_model=list[FeedTransactionOut])
def transactions(
    limit: int = Query(60, ge=1, le=500),
    recommendation: str | None = Query(None),
    incident_id: str | None = Query(None),
    session: Session = Depends(get_session),
) -> list[dict]:
    """The live feed: recent transactions with all three engines' verdicts."""
    query = (
        select(Transaction, Prediction)
        .join(Prediction, Prediction.transaction_pk == Transaction.id)
        .order_by(Transaction.transaction_dt.desc())
        .limit(limit)
    )
    if recommendation:
        query = query.where(Prediction.recommendation == recommendation.upper())
    if incident_id:
        query = query.where(Prediction.incident_id == incident_id)

    return [
        {
            "transaction_id": t.transaction_id,
            "amount": t.amount,
            "transaction_dt": t.transaction_dt,
            "product_cd": t.product_cd,
            "customer_id": t.customer_id,
            "merchant_id": t.merchant_id,
            "device_id": t.device_id,
            "risk_score": p.risk_score,
            "risk_level": p.risk_level,
            "recommendation": p.recommendation,
            "behaviour_score": p.behaviour_score,
            "behaviour_detail": p.behaviour_detail,
            "incident_id": p.incident_id,
            "model_version": p.model_version,
            "reasons": p.reasons,
        }
        for t, p in session.execute(query).all()
    ]
