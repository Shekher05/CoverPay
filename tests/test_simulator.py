"""Tests for the synthetic transaction generator.

The rule these guard is the one the project plan calls out explicitly:
synthetic ground-truth labels are evaluation metadata and must never become
model input features.

Needs the dataset (for templates) and trained artifacts.

    pytest tests/test_simulator.py
"""
import pandas as pd
import pytest

from ml.inference import get_model
from simulator.generate import SYNTHETIC_LABEL_COLUMNS, generate_batch, score_batch


@pytest.fixture(scope="module")
def batch():
    """One small seeded batch, reused - each build reads 120k template rows."""
    return generate_batch(rows=300, fraud_rate=0.1, seed=7)


def test_ground_truth_labels_never_become_features(batch):
    """The whole point of keeping them in separate columns."""
    _, spec = get_model()
    for column in SYNTHETIC_LABEL_COLUMNS:
        assert column in batch.columns, "ground truth must be recorded"
        assert column not in spec.columns, f"{column} leaked into the model input"
    assert "isFraud" not in batch.columns, "real label must be consumed, not copied"


def test_rows_are_full_width_so_they_actually_score(batch):
    """A sparse payload scores near zero regardless of its risk. Synthetic rows
    must carry the columns the model was trained on."""
    _, spec = get_model()
    present = [c for c in spec.columns if c in batch.columns]
    assert len(present) > 200, (
        f"only {len(present)} of {len(spec.columns)} features present"
    )


def test_seeded_generation_is_reproducible():
    """Evaluation mode needs repeatable batches."""
    left = generate_batch(rows=120, fraud_rate=0.1, seed=99)
    right = generate_batch(rows=120, fraud_rate=0.1, seed=99)
    pd.testing.assert_frame_equal(left, right)


def test_different_seeds_differ():
    left = generate_batch(rows=120, fraud_rate=0.1, seed=1)
    right = generate_batch(rows=120, fraud_rate=0.1, seed=2)
    assert not left["TransactionAmt"].equals(right["TransactionAmt"])


def test_identifiers_and_clock_are_synthetic(batch):
    """IDs must not collide with real transactions, and TransactionDT stays a
    strictly increasing integer offset."""
    assert batch["TransactionID"].min() >= 9_000_000
    assert batch["TransactionID"].is_unique
    dt = batch["TransactionDT"]
    assert dt.is_monotonic_increasing and dt.is_unique
    assert dt.dtype.kind in "iu", "TransactionDT must stay an integer offset"


def test_requested_fraud_rate_is_honoured(batch):
    assert batch["synthetic_is_fraud"].sum() == 30  # 10% of 300


def test_scoring_produces_a_valid_advisory_for_every_row(batch):
    scored = score_batch(batch)

    assert len(scored) == len(batch)
    assert scored["risk_score"].between(0.0, 1.0).all()
    assert set(scored["recommendation"]) <= {"ALLOW", "REVIEW", "BLOCK"}
    assert set(scored["risk_level"]) <= {"LOW", "MEDIUM", "HIGH"}
    # Level and recommendation are one policy mapping, so they cannot disagree.
    pairs = set(zip(scored["risk_level"], scored["recommendation"]))
    assert pairs <= {("LOW", "ALLOW"), ("MEDIUM", "REVIEW"), ("HIGH", "BLOCK")}


def test_injected_fraud_scores_higher_than_normal_traffic(batch):
    """Not a detection benchmark - a wiring check. If the model cannot separate
    fraud templates from legit ones at all, the pipeline is misassembled."""
    scored = score_batch(batch)
    fraud = scored.loc[scored["synthetic_is_fraud"] == 1, "risk_score"].mean()
    legit = scored.loc[scored["synthetic_is_fraud"] == 0, "risk_score"].mean()
    assert fraud > legit, f"fraud mean {fraud:.3f} not above legit mean {legit:.3f}"
