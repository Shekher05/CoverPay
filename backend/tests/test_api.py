"""API tests for the scoring endpoint.

Uses a throwaway SQLite file per test, so it never touches the real
coverpay.db. Requires trained artifacts in models/ - run `python -m ml.train`.

    pytest tests/test_api.py
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, get_session
from api.main import app
from ml.inference import ModelUnavailable

VALID = {
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


@pytest.fixture
def client(tmp_path):
    """App wired to a temporary database."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
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


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True, "run `python -m ml.train` first"
    assert body["status"] == "ok"


def test_scoring_returns_the_prd_contract(client):
    response = client.post("/transactions/score", json=VALID)
    assert response.status_code == 200

    body = response.json()
    assert body["transaction_id"] == "3545324"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["recommendation"] in {"ALLOW", "REVIEW", "BLOCK"}
    assert body["amount_inr"] == 300.0
    assert body["model_version"]
    assert body["reasons"], "a score without evidence is not an assessment"
    assert all(r["direction"] in {"increases", "decreases"} for r in body["reasons"])


def test_stored_assessment_can_be_retrieved(client):
    posted = client.post("/transactions/score", json=VALID).json()
    fetched = client.get("/transactions/3545324").json()

    assert fetched["risk_score"] == posted["risk_score"]
    assert fetched["recommendation"] == posted["recommendation"]
    assert fetched["reasons"] == posted["reasons"]


def test_unknown_transaction_is_404(client):
    assert client.get("/transactions/does-not-exist").status_code == 404


def test_responses_carry_security_headers(client):
    h = client.get("/health").headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert h["referrer-policy"] == "no-referrer"


def test_assistant_is_public_even_with_api_key(client, monkeypatch):
    """The assistant is the interactive part of the public demo: setting API_KEY
    must not lock visitors out of it (it stays rate-limited instead)."""
    from api.assistant import AssistantUnavailable

    monkeypatch.setattr("config.settings.api_key", "s3cret")

    def unavailable(*_a, **_k):
        raise AssistantUnavailable("no creds")

    monkeypatch.setattr("api.main.ask", unavailable)
    # 503 means the request reached the handler - it was not turned away at 401.
    assert client.post("/assistant/ask", json={"question": "hi"}).status_code == 503


def test_score_requires_key_when_configured(client, monkeypatch):
    """Writing endpoints stay gated once API_KEY is set."""
    monkeypatch.setattr("config.settings.api_key", "s3cret")
    assert client.post("/transactions/score", json=VALID).status_code == 401


def test_client_ip_keys_on_forwarded_for():
    """Rate limiting must bucket per real visitor, not per shared proxy IP."""
    from api.auth import _client_ip

    class Req:
        def __init__(self, headers, host):
            self.headers = headers
            self.client = type("C", (), {"host": host})()

    # Proxy present: the left-most X-Forwarded-For entry wins over the peer.
    assert _client_ip(Req({"x-forwarded-for": "203.0.113.7, 10.0.0.1"}, "10.0.0.1")) == "203.0.113.7"
    # No proxy: fall back to the socket peer.
    assert _client_ip(Req({}, "198.51.100.2")) == "198.51.100.2"


@pytest.mark.parametrize(
    "payload,reason",
    [
        ({**VALID, "TransactionAmt": -5.0}, "negative amount"),
        ({**VALID, "TransactionAmt": 0}, "zero amount"),
        ({k: v for k, v in VALID.items() if k != "TransactionAmt"}, "missing amount"),
        ({**VALID, "TransactionDT": -1}, "negative offset"),
        ({**VALID, "TransactionAmt": "not a number"}, "non-numeric amount"),
    ],
)
def test_invalid_input_is_rejected_before_inference(client, payload, reason):
    assert client.post("/transactions/score", json=payload).status_code == 422, reason


def test_oversized_payload_is_rejected(client):
    flooded = {**VALID, **{f"junk_{i}": 1 for i in range(700)}}
    assert client.post("/transactions/score", json=flooded).status_code == 422


def test_missing_model_reports_503_not_500(client, monkeypatch):
    """A missing artifact is an operational problem the caller can act on, so it
    must not surface as an opaque 500."""

    def unavailable(_):
        raise ModelUnavailable("artifacts missing")

    monkeypatch.setattr("api.main.score", unavailable)
    response = client.post("/transactions/score", json=VALID)
    assert response.status_code == 503


def test_assistant_route_maps_unavailable_to_503(client, monkeypatch):
    """The assistant helper is covered directly in test_assistant.py; this pins
    that the HTTP layer turns AssistantUnavailable into a 503, not a 500 or a
    fabricated answer."""
    from api.assistant import AssistantUnavailable

    def unavailable(*_a, **_k):
        raise AssistantUnavailable("No OpenRouter credentials found.")

    monkeypatch.setattr("api.main.ask", unavailable)
    response = client.post("/assistant/ask", json={"question": "what happened?"})
    assert response.status_code == 503
