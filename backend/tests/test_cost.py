"""The cost model is the piece that turns a threshold into a business argument,
so it gets a test that pins the arithmetic rather than trusting the report.

Runs on eight hand-built rows. No dataset, no model, no trained artifacts.

    pytest tests/test_cost.py
"""
import numpy as np
import pandas as pd
import pytest

from config import settings
from ml.train import GRID, cost, do_nothing_cost, sweep


@pytest.fixture
def scored():
    """Eight rows spanning both labels and the whole score range.

    Scores sit either side of 0.30 and 0.70 on purpose, so a shift in either
    threshold moves a known row between bands.
    """
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.05, 0.35, 0.75, 0.95, 0.10, 0.40, 0.80, 0.99])
    amounts = pd.Series([100.0, 200.0, 400.0, 800.0, 1000.0, 2000.0, 4000.0, 8000.0])
    return y, scores, amounts


def test_do_nothing_is_every_fraud_plus_a_fee_each(scored):
    y, _, amounts = scored
    expected = (1000 + 2000 + 4000 + 8000) + settings.chargeback_fee * 4
    assert do_nothing_cost(y, amounts) == pytest.approx(expected)


def test_three_bands_are_priced_separately(scored):
    y, scores, amounts = scored
    c = cost(y, scores, amounts, review_t=0.30, block_t=0.70)

    # Below 0.30: one fraud (1000) missed, plus one legit row nobody looks at.
    assert c["fraud_loss"] == pytest.approx(1000 + settings.chargeback_fee)
    # At or above 0.30: six rows in the queue.
    assert c["reviewed"] == 6
    assert c["review_effort"] == pytest.approx(settings.review_cost * 6)
    # At or above 0.70: four rows, of which two are legitimate (400 + 800).
    assert c["blocked"] == 4
    assert c["block_friction"] == pytest.approx(settings.block_friction_rate * 1200)
    assert c["total"] == pytest.approx(
        c["fraud_loss"] + c["review_effort"] + c["block_friction"]
    )


def test_flagging_everything_removes_fraud_loss_but_not_cost(scored):
    y, scores, amounts = scored
    c = cost(y, scores, amounts, review_t=0.0, block_t=1.01)
    assert c["fraud_loss"] == 0.0
    assert c["blocked"] == 0  # nothing reaches BLOCK, so no customer friction
    assert c["total"] == pytest.approx(settings.review_cost * len(y))


def test_raising_the_review_threshold_never_reduces_missed_fraud(scored):
    y, scores, amounts = scored
    losses = [cost(y, scores, amounts, t, 1.01)["fraud_loss"] for t in GRID]
    assert losses == sorted(losses), "missed fraud must be monotonic in the threshold"


def test_sweep_returns_the_cheapest_policy_first(scored):
    y, scores, amounts = scored
    ranked = sweep(y, scores, amounts)

    assert [c["total"] for c in ranked] == sorted(c["total"] for c in ranked)
    assert ranked[0]["block_threshold"] >= ranked[0]["review_threshold"]
    # The grid is triangular: every review threshold paired with every block at
    # or above it.
    assert len(ranked) == sum(int((GRID >= r).sum()) for r in GRID)


def test_sweep_prefers_catching_fraud_when_a_chargeback_dwarfs_a_review(scored):
    """With reviews at 40 and chargebacks at 750 plus the amount, the cheapest
    policy is a low review threshold. This is the sanity check that the sweep
    optimises cost and not, say, precision."""
    y, scores, amounts = scored
    best = sweep(y, scores, amounts)[0]
    assert best["review_threshold"] <= 0.10
    assert best["total"] < cost(y, scores, amounts, 0.30, 0.70)["total"]
