"""Tests for the spike and incident engine.

Two things matter: a quiet merchant must never raise an incident, and a burst of
many customers at one merchant must always raise one. Everything else is detail.

Runs on hand-built frames - no dataset, no model.

    pytest tests/test_incidents.py
"""
import pandas as pd
import pytest

from fraud_engine.incidents import (
    ACTIVE_COUNT,
    COOLDOWN_SECONDS,
    MIN_CUSTOMER_FANOUT,
    MIN_WINDOW_COUNT,
    IncidentStatus,
    Severity,
    detect_incidents,
)

START = 16_000_000


def rows(specs: list[dict]) -> pd.DataFrame:
    base = {
        "merchant_id": "MERCH_001",
        "customer_id": "CUST_0001",
        "device_id": "DEV_0001",
        "TransactionAmt": 100.0,
    }
    return pd.DataFrame([{**base, **spec} for spec in specs])


def spike(n: int, gap: int = 5, merchant: str = "MERCH_001", start: int = START):
    """`n` transactions from `n` different customers, seconds apart."""
    return [
        {
            "merchant_id": merchant,
            "customer_id": f"CUST_{i:04d}",
            "device_id": f"DEV_{i:04d}",
            "TransactionDT": start + i * gap,
        }
        for i in range(n)
    ]


def quiet(n: int, gap: int = 1_800, merchant: str = "MERCH_001", start: int = START):
    """`n` transactions half an hour apart - an ordinary trickle."""
    return [
        {
            "merchant_id": merchant,
            "customer_id": f"CUST_{i:04d}",
            "device_id": f"DEV_{i:04d}",
            "TransactionDT": start + i * gap,
        }
        for i in range(n)
    ]


def test_missing_entity_columns_is_an_error():
    with pytest.raises(ValueError, match="merchant_id"):
        detect_incidents(pd.DataFrame({"TransactionDT": [1], "TransactionAmt": [1.0]}))


def test_quiet_trading_raises_no_incident():
    """A rate detector that fires on normal traffic is worse than nothing."""
    per_row, incidents = detect_incidents(rows(quiet(30)))
    assert incidents == []
    assert (per_row["incident_id"] == "").all()


def test_a_burst_from_many_customers_raises_an_incident():
    per_row, incidents = detect_incidents(rows(spike(MIN_CUSTOMER_FANOUT * 3)))

    assert len(incidents) == 1
    incident = incidents[0]
    assert incident.merchant_id == "MERCH_001"
    assert len(incident.customers) >= MIN_CUSTOMER_FANOUT
    assert (per_row["incident_id"] != "").any()


def test_a_burst_below_the_minimum_count_is_ignored():
    """Ratios are meaningless on tiny volumes: 3 transactions where there are
    usually none is noise, not an attack."""
    _, incidents = detect_incidents(rows(spike(MIN_WINDOW_COUNT - 1)))
    assert incidents == []


def test_sustained_activity_is_promoted_to_active():
    _, incidents = detect_incidents(rows(spike(ACTIVE_COUNT + 15)))
    assert incidents[0].status == IncidentStatus.ACTIVE
    assert incidents[0].transactions >= ACTIVE_COUNT


def test_a_short_burst_stays_suspicious():
    _, incidents = detect_incidents(rows(spike(MIN_CUSTOMER_FANOUT * 2)))
    assert incidents[0].status == IncidentStatus.SUSPICIOUS


def test_an_incident_resolves_after_the_cooldown():
    """Quiet must eventually close an incident, or everything stays open."""
    burst = spike(MIN_CUSTOMER_FANOUT * 3)
    later = burst[-1]["TransactionDT"] + COOLDOWN_SECONDS + 1
    trailing = quiet(30, gap=1_800, start=later)

    _, incidents = detect_incidents(rows(burst + trailing))
    assert incidents[0].status == IncidentStatus.RESOLVED


def test_separate_bursts_become_separate_incidents():
    first = spike(MIN_CUSTOMER_FANOUT * 3, start=START)
    second_start = first[-1]["TransactionDT"] + COOLDOWN_SECONDS * 3
    second = spike(MIN_CUSTOMER_FANOUT * 3, start=second_start)

    _, incidents = detect_incidents(rows(first + second))
    assert len(incidents) == 2
    assert incidents[0].incident_id != incidents[1].incident_id


def test_merchants_are_tracked_independently():
    """A spike at one merchant must not implicate another."""
    busy = spike(MIN_CUSTOMER_FANOUT * 3, merchant="MERCH_001")
    calm = quiet(20, merchant="MERCH_002")

    _, incidents = detect_incidents(rows(busy + calm))
    assert {i.merchant_id for i in incidents} == {"MERCH_001"}


def test_incident_records_the_entities_involved():
    _, incidents = detect_incidents(rows(spike(MIN_CUSTOMER_FANOUT * 3)))
    summary = incidents[0].summary()

    assert summary["customers"] >= MIN_CUSTOMER_FANOUT
    assert summary["devices"] >= MIN_CUSTOMER_FANOUT
    assert summary["total_value"] > 0
    assert summary["duration_seconds"] >= 0
    assert summary["status"] in {"SUSPICIOUS", "ACTIVE", "RESOLVED"}
    assert summary["severity"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_risky_value_only_counts_transactions_above_the_threshold():
    frame = rows(spike(MIN_CUSTOMER_FANOUT * 3))
    # The incident only opens once MIN_WINDOW_COUNT transactions are in the
    # window, so the risky rows have to be at the end to fall inside it.
    frame["behaviour_score"] = [0.0] * (len(frame) - 5) + [0.9] * 5

    _, incidents = detect_incidents(
        frame, risk_column="behaviour_score", risk_threshold=0.5
    )
    incident = incidents[0]
    assert 0 < incident.risky_value < incident.total_value


def test_severity_rises_with_scale():
    _, small = detect_incidents(rows(spike(MIN_CUSTOMER_FANOUT * 2)))
    _, large = detect_incidents(rows(spike(ACTIVE_COUNT + 25)))
    order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    assert order.index(large[0].severity) >= order.index(small[0].severity)


def test_output_row_order_matches_the_input():
    frame = rows(
        [
            {"TransactionDT": START + 300, "customer_id": "CUST_C"},
            {"TransactionDT": START + 100, "customer_id": "CUST_A"},
            {"TransactionDT": START + 200, "customer_id": "CUST_B"},
        ]
    )
    per_row, _ = detect_incidents(frame)
    assert list(per_row.index) == list(frame.index)
    assert len(per_row) == len(frame)
