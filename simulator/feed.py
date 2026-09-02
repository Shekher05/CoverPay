"""Phase 13 - run a simulated stream through every engine and into the database.

    python -m simulator.feed --rows 4000 --reset

This is the pipeline the design doc draws end to end:

    simulator -> model -> behaviour engine -> incident engine -> database

The dashboard reads only the database, so it never imports a model or a
simulator. That keeps the read path fast, and means the dashboard would work
identically against real scored traffic.

Ground truth stays out. `synthetic_is_fraud` and friends are evaluation
metadata; writing them to the database would let a dashboard quietly display
answers no real deployment could know.
"""
from __future__ import annotations

import argparse
import math

import xgboost as xgb

from backend.database import (
    IncidentRecord,
    Prediction,
    SessionLocal,
    Transaction,
    init_db,
    reset_db,
)
from config import settings
from fraud_engine.advisory import classify
from fraud_engine.behavioural import analyse_stream
from fraud_engine.incidents import detect_incidents
from ml.features import transform
from ml.inference import TOP_REASONS, get_model

COMMIT_EVERY = 500


def _reasons(contribs, features, position: int, columns: list[str]) -> list[dict]:
    """Top contributions for one row, as stored evidence."""
    row = contribs[position][:-1]  # last element is the model bias
    top = sorted(range(len(columns)), key=lambda i: -abs(row[i]))[:TOP_REASONS]
    values = features.iloc[position]

    out = []
    for i in top:
        value = float(values.iloc[i])
        shown = "missing" if math.isnan(value) else f"{value:,.4g}"
        direction = "increases" if row[i] > 0 else "decreases"
        out.append(
            {
                "feature": columns[i],
                "value": None if math.isnan(value) else value,
                "contribution": float(row[i]),
                "direction": direction,
                "text": f"{columns[i]} = {shown} {direction} risk",
            }
        )
    return out


def feed_database(
    rows: int = 4_000, seed: int | None = None, reset: bool = False
) -> dict:
    """Generate, analyse and persist a stream. Returns what was written."""
    from simulator.world import generate_stream

    model, spec = get_model()

    print(f"Generating {rows:,} transactions...", flush=True)
    stream = generate_stream(rows, "evaluation", seed, scenario_rate=0.05)

    print("Scoring with the model...", flush=True)
    features = transform(stream, spec)
    scores = model.predict_proba(features)[:, 1]
    # One TreeSHAP pass for the whole batch rather than per row.
    contribs = model.get_booster().predict(
        xgb.DMatrix(features, feature_names=spec.columns), pred_contribs=True
    )

    print("Analysing behaviour...", flush=True)
    stream = stream.join(analyse_stream(stream))

    print("Detecting spikes...", flush=True)
    per_row, incidents = detect_incidents(stream, risk_column="behaviour_score")
    stream = stream.join(per_row)

    reset_db() if reset else init_db()

    print("Writing to the database...", flush=True)
    written = 0
    with SessionLocal() as session:
        for position, (_, row) in enumerate(stream.iterrows()):
            score = float(scores[position])
            level, recommendation = classify(score)

            transaction = Transaction(
                transaction_id=str(row["TransactionID"]),
                amount=float(row["TransactionAmt"]),
                transaction_dt=int(row["TransactionDT"]),
                product_cd=row.get("ProductCD"),
                customer_id=row.get("customer_id"),
                merchant_id=row.get("merchant_id"),
                device_id=row.get("device_id"),
            )
            session.add(transaction)
            session.flush()

            session.add(
                Prediction(
                    transaction_pk=transaction.id,
                    risk_score=round(score, 6),
                    risk_level=str(level),
                    recommendation=str(recommendation),
                    model_version=settings.model_version,
                    reasons=_reasons(contribs, features, position, spec.columns),
                    behaviour_score=float(row["behaviour_score"]),
                    behaviour_detail=str(row["behaviour_detail"])[:512],
                    incident_id=row["incident_id"] or None,
                )
            )
            written += 1
            if written % COMMIT_EVERY == 0:
                session.commit()
                print(f"  {written:,}/{len(stream):,}", flush=True)

        for incident in incidents:
            summary = incident.summary()
            session.merge(
                IncidentRecord(
                    incident_id=summary["incident_id"],
                    merchant_id=summary["merchant_id"],
                    status=summary["status"],
                    severity=summary["severity"],
                    opened_dt=summary["opened_dt"],
                    last_dt=summary["last_dt"],
                    transactions=summary["transactions"],
                    customers=summary["customers"],
                    devices=summary["devices"],
                    total_value=summary["total_value"],
                    risky_value=summary["risky_value"],
                )
            )
        session.commit()

    return {"transactions": written, "incidents": len(incidents)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feed the database from the simulator."
    )
    parser.add_argument("--rows", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--reset", action="store_true", help="drop and recreate the tables first"
    )
    args = parser.parse_args()

    result = feed_database(args.rows, args.seed, args.reset)
    print(
        f"\nWrote {result['transactions']:,} transactions and "
        f"{result['incidents']:,} incidents."
    )
    print("Start the API, then open the dashboard.")


if __name__ == "__main__":
    main()
