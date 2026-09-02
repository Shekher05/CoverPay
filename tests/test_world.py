"""Tests for the synthetic payment environment.

Two properties matter more than the rest: entities must actually recur (no
history, no behaviour engine), and ground-truth columns must never become model
features.

    pytest tests/test_world.py
"""
import numpy as np
import pandas as pd
import pytest

from ml.inference import get_model
from simulator.world import SCENARIOS, build_world, generate_stream

ENTITY_COLUMNS = ["customer_id", "merchant_id", "device_id"]
GROUND_TRUTH = ["synthetic_is_fraud", "synthetic_scenario", "synthetic_scenario_id"]


@pytest.fixture(scope="module")
def stream():
    """One seeded stream, reused - each build reads 120k template rows."""
    return generate_stream(rows=1_200, mode="evaluation", seed=11, scenario_rate=0.05)


def test_ground_truth_never_becomes_a_feature(stream):
    _, spec = get_model()
    for column in GROUND_TRUTH:
        assert column in stream.columns
        assert column not in spec.columns, f"{column} leaked into model input"
    assert "isFraud" not in stream.columns


def test_entities_recur_so_history_exists(stream):
    """The whole reason this module exists. If every customer appeared once,
    there would be no behavioural baseline to compare against."""
    per_customer = stream["customer_id"].value_counts()
    assert per_customer.median() > 1, "customers must transact more than once"
    assert stream["merchant_id"].nunique() <= 12
    assert stream["customer_id"].nunique() < len(stream)


def test_entity_columns_are_populated(stream):
    for column in ENTITY_COLUMNS:
        assert column in stream.columns
        assert stream[column].notna().all()
    assert stream["customer_id"].str.startswith("CUST_").all()
    assert stream["merchant_id"].str.startswith("MERCH_").all()


def test_transaction_clock_is_an_increasing_integer_offset(stream):
    dt = stream["TransactionDT"]
    assert dt.dtype.kind in "iu", "TransactionDT must stay an integer offset"
    assert dt.is_monotonic_increasing
    assert dt.min() >= 16_000_000, "must not overlap the real data's timeline"


def test_transaction_ids_cannot_collide_with_real_data(stream):
    assert stream["TransactionID"].is_unique
    assert stream["TransactionID"].min() >= 9_500_000


def test_every_scenario_id_is_one_coherent_incident(stream):
    """An incident is a group of related transactions, so a scenario id must not
    span two different attack types."""
    tagged = stream[stream["synthetic_scenario_id"] != ""]
    types_per_incident = tagged.groupby("synthetic_scenario_id")[
        "synthetic_scenario"
    ].nunique()
    assert (types_per_incident == 1).all()
    assert set(tagged["synthetic_scenario"]) <= set(SCENARIOS)
    # Everything inside an injected scenario is labelled fraud.
    assert (tagged["synthetic_is_fraud"] == 1).all()


def test_normal_traffic_is_unlabelled_and_untagged(stream):
    normal = stream[stream["synthetic_scenario"] == "normal"]
    assert (normal["synthetic_is_fraud"] == 0).all()
    assert (normal["synthetic_scenario_id"] == "").all()


def test_card_testing_is_a_burst_of_small_payments(stream):
    """Scenario shape must match its description, or the behaviour engine will
    be tuned against a fiction."""
    bursts = stream[stream["synthetic_scenario"] == "card_testing"]
    if bursts.empty:
        pytest.skip("no card_testing scenario in this seeded stream")

    for _, group in bursts.groupby("synthetic_scenario_id"):
        assert group["card1"].nunique() == 1, "one card per testing burst"
        assert group["TransactionAmt"].max() <= 25, "card testing uses small amounts"
        span = group["TransactionDT"].max() - group["TransactionDT"].min()
        assert span < 3600, "a burst happens within the hour"


def test_account_takeover_changes_device_and_value(stream):
    takeovers = stream[stream["synthetic_scenario"] == "account_takeover"]
    if takeovers.empty:
        pytest.skip("no account_takeover scenario in this seeded stream")

    for _, group in takeovers.groupby("synthetic_scenario_id"):
        customer = group["customer_id"].iloc[0]
        assert group["customer_id"].nunique() == 1
        assert group["device_id"].str.startswith("DEV_NEW_").all(), "new device"
        history = stream[
            (stream["customer_id"] == customer)
            & (stream["synthetic_scenario"] == "normal")
        ]
        if not history.empty:
            assert group["TransactionAmt"].mean() > history["TransactionAmt"].mean()


def test_coordinated_spike_hits_one_merchant_from_many_customers(stream):
    spikes = stream[stream["synthetic_scenario"] == "coordinated_spike"]
    if spikes.empty:
        pytest.skip("no coordinated_spike scenario in this seeded stream")

    for _, group in spikes.groupby("synthetic_scenario_id"):
        assert group["merchant_id"].nunique() == 1, "one merchant under attack"
        assert group["customer_id"].nunique() >= 10, "many customers involved"


def test_evaluation_mode_is_reproducible():
    left = generate_stream(rows=200, mode="evaluation", seed=5)
    right = generate_stream(rows=200, mode="evaluation", seed=5)
    pd.testing.assert_frame_equal(left, right)


def test_live_mode_differs_between_runs():
    left = generate_stream(rows=200, mode="live")
    right = generate_stream(rows=200, mode="live")
    assert not left["TransactionAmt"].equals(right["TransactionAmt"])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="evaluation"):
        generate_stream(rows=10, mode="production")


def test_world_population_is_stable_for_a_seed():
    left = build_world(np.random.default_rng(3))
    right = build_world(np.random.default_rng(3))
    assert left == right
