"""Dashboard read endpoints.

These have no model and no simulator - pure projections of the database - so the
fixture just writes rows directly.

    pytest tests/test_dashboard.py
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, IncidentRecord, Prediction, Transaction, get_session
from api.main import app

START = 20_000_000


@pytest.fixture
def client(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'dash.db'}")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = override
    with TestClient(app) as c:
        yield c, TestSession
    app.dependency_overrides.clear()


def _tx(db, *, tid, dt, amount, recommendation, risk, behaviour=0.0, incident=None):
    t = Transaction(
        transaction_id=tid, amount=amount, transaction_dt=dt,
        product_cd="C", customer_id=f"C{tid}", merchant_id="M1", device_id=f"D{tid}",
    )
    db.add(t)
    db.flush()
    db.add(Prediction(
        transaction_pk=t.id, risk_score=risk,
        risk_level="HIGH" if risk >= 0.7 else "LOW",
        recommendation=recommendation, model_version="v1",
        reasons=[{"feature": "V1", "value": 1.0, "contribution": 0.2,
                  "direction": "increases", "text": "V1 = 1 increases risk"}],
        behaviour_score=behaviour, behaviour_detail="nothing unusual",
        incident_id=incident,
    ))


def _incident(db, *, iid, severity, opened):
    db.add(IncidentRecord(
        incident_id=iid, merchant_id="M1", status="ACTIVE", severity=severity,
        opened_dt=opened, last_dt=opened + 300, transactions=10, customers=8,
        devices=8, total_value=5000.0, risky_value=3000.0,
    ))


@pytest.fixture
def populated(client):
    c, TestSession = client
    with TestSession() as db:
        _tx(db, tid="1", dt=START, amount=100.0, recommendation="ALLOW", risk=0.01)
        _tx(db, tid="2", dt=START + 60, amount=200.0, recommendation="REVIEW", risk=0.4,
            behaviour=0.5)
        _tx(db, tid="3", dt=START + 120, amount=300.0, recommendation="BLOCK", risk=0.95,
            incident="INC_9999")
        db.commit()
    return c


def test_summary_shape_and_counts(populated):
    body = populated.get("/dashboard/summary").json()

    assert body["transactions"] == 3
    assert body["total_value"] == 600.0
    assert body["flagged"] == 2  # REVIEW + BLOCK
    assert body["flagged_value"] == 500.0
    assert body["engine_flags"] == {"model": 2, "behaviour": 1, "incident": 1}
    assert set(body["window"]) == {"first_dt", "last_dt", "hours"}


def test_timeline_buckets_sum_to_the_total(populated):
    body = populated.get("/dashboard/timeline?buckets=4").json()
    assert sum(b["transactions"] for b in body["buckets"]) == 3
    assert sum(b["flagged"] for b in body["buckets"]) == 2


def test_timeline_on_empty_db_is_well_formed(client):
    c, _ = client
    body = c.get("/dashboard/timeline").json()
    assert body == {"buckets": [], "bucket_seconds": 0, "first_dt": None}


def test_transactions_feed_filters_by_recommendation(populated):
    rows = populated.get("/dashboard/transactions?recommendation=BLOCK").json()
    assert [r["transaction_id"] for r in rows] == ["3"]
    assert rows[0]["reasons"][0]["feature"] == "V1"


def test_incidents_keep_the_most_severe_under_a_small_limit(client):
    """The severe incident is the oldest, so a limit that ordered by recency
    first would drop it. It must survive."""
    c, TestSession = client
    with TestSession() as db:
        _incident(db, iid="INC_OLD_CRIT", severity="CRITICAL", opened=START)
        for i in range(5):
            _incident(db, iid=f"INC_NEW_{i}", severity="LOW", opened=START + 10_000 + i)
        db.commit()

    rows = c.get("/dashboard/incidents?limit=3").json()
    assert len(rows) == 3
    assert rows[0]["incident_id"] == "INC_OLD_CRIT"
    assert rows[0]["severity"] == "CRITICAL"
