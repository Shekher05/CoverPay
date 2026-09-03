"""Tests for the AI merchant helper.

These never call the Gemini API. What matters here is the grounding layer,
and that is pure SQL: if a figure is not in the context, the model has no
legitimate way to produce it. Testing the prose would cost money and prove
little; testing the evidence proves the boundary holds.

    pytest tests/test_assistant.py
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.assistant import AssistantUnavailable, ask, build_context
from api.database import Base, IncidentRecord, Prediction, Transaction

START = 16_000_000


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'assistant.db'}")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        yield db


def add(
    db,
    *,
    tid,
    amount,
    merchant,
    dt,
    recommendation,
    risk,
    behaviour=0.0,
    incident=None,
):
    transaction = Transaction(
        transaction_id=tid,
        amount=amount,
        transaction_dt=dt,
        merchant_id=merchant,
        customer_id=f"CUST_{tid}",
        device_id=f"DEV_{tid}",
    )
    db.add(transaction)
    db.flush()
    db.add(
        Prediction(
            transaction_pk=transaction.id,
            risk_score=risk,
            risk_level="HIGH" if risk >= 0.7 else "LOW",
            recommendation=recommendation,
            model_version="v1",
            reasons=[{"text": "V258 = 6 increases risk", "feature": "V258"}],
            behaviour_score=behaviour,
            behaviour_detail="nothing unusual",
            incident_id=incident,
        )
    )


@pytest.fixture
def populated(session):
    add(session, tid="1", amount=100.0, merchant="MERCH_A", dt=START,
        recommendation="ALLOW", risk=0.01)
    add(session, tid="2", amount=200.0, merchant="MERCH_A", dt=START + 60,
        recommendation="BLOCK", risk=0.98, behaviour=0.5, incident="INC_0001")
    add(session, tid="3", amount=300.0, merchant="MERCH_B", dt=START + 120,
        recommendation="REVIEW", risk=0.45)
    session.add(
        IncidentRecord(
            incident_id="INC_0001",
            merchant_id="MERCH_A",
            status="ACTIVE",
            severity="HIGH",
            opened_dt=START,
            last_dt=START + 300,
            transactions=12,
            customers=9,
            devices=9,
            total_value=1500.0,
            risky_value=900.0,
        )
    )
    session.commit()
    return session


def test_empty_database_reports_no_data(session):
    context = build_context(session)
    assert context["transactions"] == 0
    assert context["incidents"] == []


def test_context_totals_match_the_database(populated):
    context = build_context(populated)

    assert context["transactions"] == 3
    assert context["total_value"] == 600.0
    # Flagged means anything that is not ALLOW.
    assert context["flagged_count"] == 2
    assert context["flagged_value"] == 500.0
    assert context["advisory_breakdown"]["ALLOW"]["count"] == 1


def test_context_reports_each_engine_separately(populated):
    """The three engines disagree by design, so their counts must not be merged."""
    flags = build_context(populated)["engine_flags"]
    assert flags["model_flagged"] == 2
    assert flags["behaviour_flagged"] == 1
    assert flags["in_an_incident"] == 1


def test_merchant_scope_filters_everything(populated):
    context = build_context(populated, merchant_id="MERCH_B")

    assert context["transactions"] == 1
    assert context["total_value"] == 300.0
    assert {t["merchant_id"] for t in context["riskiest_transactions"]} == {"MERCH_B"}
    # MERCH_A's incident must not leak into MERCH_B's view.
    assert context["incidents"] == []


def test_riskiest_transactions_are_ordered_and_carry_evidence(populated):
    riskiest = build_context(populated)["riskiest_transactions"]

    scores = [t["risk_score"] for t in riskiest]
    assert scores == sorted(scores, reverse=True)
    assert riskiest[0]["transaction_id"] == "2"
    assert riskiest[0]["top_reason"], "a risky transaction must arrive with its reason"


def test_context_never_carries_ground_truth(populated):
    """The simulator knows which transactions were injected. A real deployment
    would not, so that must never reach the assistant."""
    flat = repr(build_context(populated))
    assert "synthetic_is_fraud" not in flat
    assert "synthetic_scenario" not in flat


def test_time_window_is_reported_as_a_span_not_a_date(populated):
    """TransactionDT is a seconds offset with no calendar origin."""
    scope = build_context(populated)["scope"]
    # Reported to 2dp, so the tolerance matches the rounding rather than fighting it.
    assert scope["window_hours"] == pytest.approx(120 / 3600, abs=0.005)
    assert "date" not in scope and "timestamp" not in scope


def test_hours_filter_narrows_the_window(populated):
    """Only the most recent slice, measured back from the latest transaction."""
    context = build_context(populated, limit_hours=0.02)  # 72 seconds
    assert context["transactions"] == 2


def test_empty_question_is_rejected_before_any_api_call(populated):
    with pytest.raises(ValueError, match="empty"):
        ask("   ", populated)


def test_no_data_answers_without_calling_the_model(session):
    """An empty database has nothing to explain, so it must not spend an API
    call to say so."""
    result = ask("why did my risk increase?", session)

    assert "no transaction data" in result.answer.lower()
    assert result.context["transactions"] == 0


def test_missing_credentials_raise_assistant_unavailable(populated, monkeypatch):
    """A missing key is an operational problem, not a wrong answer. It must
    surface rather than degrade into invented prose."""
    import api.assistant as assistant

    def no_client():
        raise AssistantUnavailable("No Gemini credentials found.")

    monkeypatch.setattr(assistant, "_client", no_client)

    with pytest.raises(AssistantUnavailable, match="credentials"):
        ask("what happened?", populated)


def _fake_client(raises=None, text=None):
    """A stand-in shaped like google-genai's client, so no request is made."""

    class Models:
        @staticmethod
        def generate_content(**_):
            if raises is not None:
                raise raises
            return type("Response", (), {"text": text})()

    return type("Client", (), {"models": Models()})()


def test_api_rejection_reads_as_unavailable_not_as_an_answer(populated, monkeypatch):
    """A bad or exhausted key is an operational problem the operator can fix.
    It must never degrade into invented prose."""
    import api.assistant as assistant
    from google.genai import errors

    rejection = errors.ClientError(
        400, {"error": {"message": "API key not valid", "status": "INVALID_ARGUMENT"}}
    )
    monkeypatch.setattr(assistant, "_client", lambda: _fake_client(raises=rejection))

    with pytest.raises(AssistantUnavailable, match="rejected"):
        ask("what happened?", populated)


def test_upstream_outage_reads_as_unavailable(populated, monkeypatch):
    import api.assistant as assistant
    from google.genai import errors

    outage = errors.ServerError(503, {"error": {"message": "overloaded"}})
    monkeypatch.setattr(assistant, "_client", lambda: _fake_client(raises=outage))

    with pytest.raises(AssistantUnavailable, match="unavailable"):
        ask("what happened?", populated)


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_blank_answer_is_reported_not_returned(populated, monkeypatch, empty):
    """A safety block or truncation returns no text. A merchant must be told
    that, rather than shown a blank answer that reads as 'nothing happened'."""
    import api.assistant as assistant

    monkeypatch.setattr(assistant, "_client", lambda: _fake_client(text=empty))

    with pytest.raises(AssistantUnavailable, match="no answer"):
        ask("what happened?", populated)


def test_a_real_answer_comes_back_with_its_evidence(populated, monkeypatch):
    """The context travels with the answer so every figure can be checked."""
    import api.assistant as assistant

    monkeypatch.setattr(
        assistant,
        "_client",
        lambda: _fake_client(text="  Two payments were flagged.  "),
    )

    result = ask("what happened?", populated)

    assert result.answer == "Two payments were flagged."
    assert result.context["transactions"] == 3
    assert result.model == assistant.MODEL
