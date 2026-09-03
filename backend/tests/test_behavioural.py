"""Tests for the behaviour engine.

The property worth guarding above all others is causality: a transaction must be
judged only against transactions that preceded it. A leak here is invisible in
the output and inflates every reported number, exactly like a random train/test
split would.

Runs on hand-built frames - no dataset, no model.

    pytest tests/test_behavioural.py
"""
import pandas as pd
import pytest

from fraud_engine.behavioural import (
    MIN_HISTORY,
    VELOCITY_THRESHOLD,
    BehaviourSignal,
    analyse_stream,
    combine,
)


def frame(rows: list[dict]) -> pd.DataFrame:
    """Build a stream frame from partial rows, filling the required columns."""
    base = {"customer_id": "CUST_00001", "device_id": "DEV_00001"}
    return pd.DataFrame([{**base, **row} for row in rows])


def steady(n: int, amount: float = 100.0, start: int = 16_000_000, gap: int = 86_400):
    """`n` unremarkable transactions, a day apart, so no velocity signal fires."""
    return [
        {"TransactionDT": start + i * gap, "TransactionAmt": amount + i}
        for i in range(n)
    ]


def test_missing_entity_columns_is_an_error():
    with pytest.raises(ValueError, match="customer_id"):
        analyse_stream(pd.DataFrame({"TransactionDT": [1], "TransactionAmt": [1.0]}))


def test_a_transaction_is_never_judged_against_itself():
    """With no prior history there is nothing to be unusual against, so the
    first transaction must score zero however extreme it looks."""
    out = analyse_stream(
        frame([{"TransactionDT": 16_000_000, "TransactionAmt": 999_999.0}])
    )
    assert out["behaviour_score"].iloc[0] == 0.0
    assert out["behaviour_signals"].iloc[0] == []


def test_future_transactions_do_not_influence_the_present():
    """Appending later activity must not change earlier verdicts. This is the
    causality guarantee; if it fails, the engine is peeking at the future."""
    early = frame(steady(6))
    late = frame(steady(6) + [{"TransactionDT": 16_900_000, "TransactionAmt": 50_000.0}])

    first = analyse_stream(early)["behaviour_score"].tolist()
    second = analyse_stream(late)["behaviour_score"].tolist()[: len(first)]
    assert first == second


def test_amount_anomaly_needs_history_before_it_fires():
    """A customer with almost no history has no 'normal' to deviate from."""
    rows = steady(MIN_HISTORY - 1) + [
        {"TransactionDT": 16_500_000, "TransactionAmt": 90_000.0}
    ]
    out = analyse_stream(frame(rows))
    assert "amount_anomaly" not in out["behaviour_signals"].iloc[-1]


def test_amount_anomaly_fires_on_a_spend_far_above_normal():
    rows = steady(8) + [{"TransactionDT": 16_800_000, "TransactionAmt": 90_000.0}]
    out = analyse_stream(frame(rows))

    assert "amount_anomaly" in out["behaviour_signals"].iloc[-1]
    assert out["behaviour_score"].iloc[-1] > 0
    assert "above" in out["behaviour_detail"].iloc[-1]


def test_ordinary_spending_stays_quiet():
    """The engine has to be silent on normal traffic or it is unusable."""
    out = analyse_stream(frame(steady(20)))
    assert (out["behaviour_score"] == 0.0).all()


def test_velocity_fires_on_a_burst_from_one_customer():
    """Card testing: many payments from one card within minutes."""
    burst = [
        {"TransactionDT": 16_000_000 + i * 3, "TransactionAmt": 5.0}
        for i in range(VELOCITY_THRESHOLD + 3)
    ]
    out = analyse_stream(frame(burst))
    assert "velocity" in out["behaviour_signals"].iloc[-1]


def test_velocity_ignores_the_same_count_spread_over_days():
    """Rate is the signal, not volume. Spacing the burst out must silence it."""
    out = analyse_stream(frame(steady(VELOCITY_THRESHOLD + 3)))
    assert all("velocity" not in s for s in out["behaviour_signals"])


def test_new_device_fires_for_an_established_customer():
    rows = steady(5) + [
        {"TransactionDT": 16_600_000, "TransactionAmt": 105.0, "device_id": "DEV_NEW_1"}
    ]
    out = analyse_stream(frame(rows))
    assert "new_device" in out["behaviour_signals"].iloc[-1]


def test_returning_to_a_known_device_is_not_flagged():
    rows = steady(5) + [
        {"TransactionDT": 16_600_000, "TransactionAmt": 105.0, "device_id": "DEV_NEW_1"},
        {"TransactionDT": 16_700_000, "TransactionAmt": 106.0, "device_id": "DEV_00001"},
    ]
    out = analyse_stream(frame(rows))
    assert "new_device" not in out["behaviour_signals"].iloc[-1]


def test_customers_are_scored_independently():
    """One customer's burst must not raise another customer's score."""
    rows = [
        {
            "customer_id": "CUST_A",
            "device_id": "DEV_A",
            "TransactionDT": 16_000_000 + i,
            "TransactionAmt": 5.0,
        }
        for i in range(VELOCITY_THRESHOLD + 3)
    ]
    rows.append(
        {
            "customer_id": "CUST_B",
            "device_id": "DEV_B",
            "TransactionDT": 16_000_100,
            "TransactionAmt": 5.0,
        }
    )
    out = analyse_stream(pd.DataFrame(rows))
    assert out["behaviour_score"].iloc[-1] == 0.0


def test_output_row_order_matches_the_input():
    """Analysis sorts by time internally; the caller must get its own order back."""
    rows = frame(
        [
            {"TransactionDT": 16_000_300, "TransactionAmt": 30.0},
            {"TransactionDT": 16_000_100, "TransactionAmt": 10.0},
            {"TransactionDT": 16_000_200, "TransactionAmt": 20.0},
        ]
    )
    out = analyse_stream(rows)
    assert list(out.index) == list(rows.index)
    assert len(out) == len(rows)


def test_combine_is_bounded_and_monotonic():
    assert combine([]) == 0.0
    one = combine([BehaviourSignal("velocity", 1.0, "")])
    two = combine(
        [BehaviourSignal("velocity", 1.0, ""), BehaviourSignal("new_device", 1.0, "")]
    )
    assert 0.0 < one < two <= 1.0


def test_unknown_signal_still_contributes_without_crashing():
    """A future signal type must not silently score zero or raise."""
    assert combine([BehaviourSignal("brand_new_signal", 1.0, "")]) > 0.0
