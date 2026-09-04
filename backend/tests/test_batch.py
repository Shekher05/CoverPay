"""Batch scoring: the `ml.inference.score_batch` helper and the
`/transactions/upload-csv` endpoint that wraps it.

Requires trained artifacts in models/ - run `python -m ml.train`.

    pytest tests/test_batch.py
"""
import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, get_session
from api.main import app
from ml.features import transform
from ml.inference import ModelUnavailable, get_model, reasons_for_row, score_batch

ROW = {
    "TransactionID": 3545324,
    "TransactionDT": 12500000,
    "TransactionAmt": 300.0,
    "ProductCD": "C",
    "card1": 13926,
    "card4": "visa",
    "card6": "credit",
    "P_emaildomain": "gmail.com",
    "C1": 1.0,
    "C13": 0.0,
}


def _frame(n: int) -> pd.DataFrame:
    """`n` transactions that vary enough for the model to rank them."""
    rows = []
    for i in range(n):
        row = dict(ROW)
        row["TransactionID"] = 1000 + i
        row["TransactionAmt"] = 10.0 + i * 137.0
        row["C1"] = float(i % 7)
        row["card1"] = 13926 + (i % 3)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def client(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'batch.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# --- score_batch ---------------------------------------------------------


def test_score_batch_scores_every_row_not_a_prefix():
    """The old implementation silently truncated to `max_rows` while the caller
    reported counts as if the whole file was scored."""
    df = _frame(120)
    assert len(score_batch(df, shap_top_n=10)) == len(df)


def test_score_batch_lines_shap_reasons_up_with_the_right_rows():
    """Only the top-N riskiest rows get real TreeSHAP reasons; the rest carry
    the summary reason. The contribution-row to DataFrame-row mapping must not
    rely on set iteration order."""
    results = score_batch(_frame(40), shap_top_n=3)

    ranked = sorted(range(len(results)), key=lambda i: results[i].risk_score, reverse=True)
    top3 = set(ranked[:3])

    for i, r in enumerate(results):
        primary = r.reasons[0].feature
        if i in top3:
            assert primary != "risk_score", f"row {i} is top-3 but has the fallback reason"
        else:
            assert primary == "risk_score", f"row {i} outside top-3 but got a SHAP reason"


def test_score_batch_rejects_an_oversized_batch_instead_of_truncating():
    with pytest.raises(ValueError, match="exceeds"):
        score_batch(_frame(5), max_rows=2)


def test_score_batch_on_an_empty_frame_is_empty():
    assert score_batch(pd.DataFrame()) == []


def test_stored_and_live_reason_selection_share_one_definition():
    """`simulator.feed._reasons` and the API both go through `reasons_for_row`,
    so stored evidence can never be selected differently from live evidence."""
    from simulator import feed

    df = _frame(6)
    model, spec = get_model()
    X = transform(df, spec)
    contribs = model.get_booster().predict(
        xgb.DMatrix(X, feature_names=spec.columns), pred_contribs=True
    )

    direct = reasons_for_row(contribs[2], X.iloc[2], spec.columns)
    via_feed = feed._reasons(contribs, X, 2, spec.columns)
    assert [r.feature for r in direct] == [d["feature"] for d in via_feed]


# --- /transactions/upload-csv -----------------------------------------------


def _upload(client, content: bytes, name: str = "batch.csv"):
    return client.post(
        "/transactions/upload-csv", files={"file": (name, content, "text/csv")}
    )


def test_upload_csv_happy_path(client):
    csv = _frame(30).to_csv(index=False).encode()
    body = _upload(client, csv).json()

    assert body["total_transactions"] == 30
    assert body["possible_attacks_flagged"] == (
        body["review_recommended"] + body["block_recommended"]
    )
    assert 0.0 <= body["high_risk_percentage"] <= 100.0
    assert len(body["riskiest_transactions"]) <= 10


def test_upload_csv_actually_streams_in_chunks(client, monkeypatch):
    """A file larger than one CSV_CHUNK_SIZE must be scored across multiple
    chunks, not read/scored in a single pass - pin this down by spying on
    score_batch rather than trusting the streaming rewrite by inspection."""
    from api import main as main_module

    n = main_module.CSV_CHUNK_SIZE * 2 + main_module.CSV_CHUNK_SIZE // 4
    df = _frame(n)
    csv = df.to_csv(index=False).encode()

    calls = []
    real_score_batch = main_module.score_batch

    def spy(chunk, **kwargs):
        calls.append(len(chunk))
        return real_score_batch(chunk, **kwargs)

    monkeypatch.setattr(main_module, "score_batch", spy)

    body = _upload(client, csv).json()

    assert len(calls) == 3, f"expected 3 chunks, scored {len(calls)}: {calls}"
    assert calls[0] == calls[1] == main_module.CSV_CHUNK_SIZE
    assert calls[2] == n - 2 * main_module.CSV_CHUNK_SIZE
    assert sum(calls) == n
    assert body["total_transactions"] == n

    # Cross-chunk aggregation must match a single unchunked pass.
    direct = real_score_batch(df, max_rows=n)
    direct_flagged = sum(1 for a in direct if a.recommendation != "ALLOW")
    assert body["possible_attacks_flagged"] == direct_flagged

    direct_top10 = sorted((a.risk_score for a in direct), reverse=True)[:10]
    reported_top10 = sorted(
        (a["risk_score"] for a in body["riskiest_transactions"]), reverse=True
    )
    assert reported_top10 == pytest.approx(direct_top10)


def test_upload_rejects_a_non_csv_name(client):
    assert _upload(client, b"x,y\n1,2\n", name="notes.txt").status_code == 400


def test_upload_rejects_an_empty_file(client):
    assert _upload(client, b"").status_code == 400


def test_upload_rejects_an_oversized_file(client, monkeypatch):
    """MAX_CSV_BYTES is 200 MB in production; shrink it here so the test
    rejects on file.size without actually generating and scoring a
    multi-hundred-MB payload."""
    monkeypatch.setattr("api.main.MAX_CSV_BYTES", 100)
    big = b"TransactionID,TransactionAmt,TransactionDT\n" + b"1,2,3\n" * 20
    assert len(big) > 100
    assert _upload(client, big).status_code == 413


def test_upload_rejects_too_many_rows_as_413_not_422(client, monkeypatch):
    """A narrow CSV can pack many rows under the byte cap; the row cap must
    surface as 'too large', not as a misleading 'missing columns' 422."""
    monkeypatch.setattr("api.main.MAX_CSV_ROWS", 3)
    csv = _frame(5).to_csv(index=False).encode()
    r = _upload(client, csv)
    assert r.status_code == 413
    assert "rows" in r.json()["detail"]


def test_upload_does_not_leak_internal_error_text(client):
    """A CSV the model cannot score comes back generic, not as the raw
    exception (which quotes the caller's columns)."""
    r = _upload(client, b"colA,colB\n1,2\n3,4\n")
    assert r.status_code == 422
    assert r.json()["detail"] == "CSV is missing columns the model needs"


def test_upload_reports_503_when_the_model_is_missing(client, monkeypatch):
    def unavailable(_df, **_kwargs):
        raise ModelUnavailable("artifacts missing")

    monkeypatch.setattr("api.main.score_batch", unavailable)
    csv = _frame(3).to_csv(index=False).encode()
    assert _upload(client, csv).status_code == 503


# --- optional API key -----------------------------------------------------


def test_api_key_is_enforced_only_when_configured(client, monkeypatch):
    csv = _frame(3).to_csv(index=False).encode()
    assert _upload(client, csv).status_code == 200  # no key configured

    monkeypatch.setattr("config.settings.api_key", "s3cret")
    assert _upload(client, csv).status_code == 401

    r = client.post(
        "/transactions/upload-csv",
        files={"file": ("b.csv", csv, "text/csv")},
        headers={"X-API-Key": "s3cret"},
    )
    assert r.status_code == 200


def test_non_ascii_api_key_raises_401_not_500(monkeypatch):
    """A non-ASCII header value must be a clean 401, not a TypeError from
    `compare_digest` surfacing as a 500."""
    from fastapi import HTTPException

    from api.auth import require_api_key

    monkeypatch.setattr("config.settings.api_key", "s3cret")
    with pytest.raises(HTTPException) as exc:
        require_api_key("cl\xe9-secr\xe8te")
    assert exc.value.status_code == 401
