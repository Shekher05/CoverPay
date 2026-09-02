"""The invariant the whole project rests on: training and inference must build
identical features.

Training transforms a 470k-row frame at once; the API transforms one row. If
those two paths ever disagree, offline metrics stay beautiful while production
scores are nonsense - the failure is silent, which is why it gets a test.

Runs on a synthetic frame with its own column decisions, so it needs neither
the 1.35GB dataset nor a prior audit run.

    pytest tests/test_features.py
"""
import json

import pandas as pd
import pytest

from ml import features


@pytest.fixture
def spec_env(tmp_path, monkeypatch):
    """Point the feature module at a synthetic set of column decisions."""
    decisions = {
        "rows_audited": 6,
        "split_dt": 300.0,
        "keep": ["TransactionAmt", "ProductCD", "card1", "card4", "addr1", "C1", "D1"],
        "categorical": ["ProductCD", "card4"],
    }
    path = tmp_path / "feature_columns.json"
    path.write_text(json.dumps(decisions), encoding="utf-8")
    monkeypatch.setattr(features, "COLUMNS_PATH", path)
    return decisions


@pytest.fixture
def raw():
    """Six rows shaped like IEEE-CIS, including the nulls that break naive code."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [86400, 90000, 180000, 264000, 350000, 440000],
            "TransactionAmt": [68.5, 29.0, 1200.75, 15.25, 68.5, 499.99],
            "ProductCD": ["W", "C", "W", "H", "C", "W"],
            "card1": [13926, 2755, 13926, 4497, 2755, 13926],
            "card4": ["discover", "visa", "visa", None, "visa", "mastercard"],
            "addr1": [315.0, 325.0, None, 315.0, 315.0, 441.0],
            "C1": [1.0, 2.0, 1.0, None, 3.0, 1.0],
            "D1": [14.0, 0.0, 30.0, 1.0, None, 7.0],
            "P_emaildomain": [
                "gmail.com",
                "gmail",
                "yahoo.fr",
                None,
                "hotmail.com",
                "gmail.com",
            ],
            "isFraud": [0, 1, 0, 0, 1, 0],
        }
    )


def test_single_row_matches_batch(spec_env, raw):
    """Score one row at a time; every value must match the batch transform."""
    spec = features.fit(raw)
    batch = features.transform(raw, spec)

    for i in range(len(raw)):
        one = features.transform(raw.iloc[[i]], spec)
        pd.testing.assert_frame_equal(
            one.reset_index(drop=True),
            batch.iloc[[i]].reset_index(drop=True),
            check_dtype=True,
        )


def test_unseen_values_score_without_raising(spec_env, raw):
    """A card and a product never seen in training must map to 0, not explode."""
    spec = features.fit(raw)
    novel = raw.iloc[[0]].copy()
    novel["card1"] = 999999
    novel["ProductCD"] = "Z"
    novel["card4"] = "amex-not-in-training"

    out = features.transform(novel, spec)

    assert out["card1_freq"].iloc[0] == 0
    assert out["ProductCD_freq"].iloc[0] == 0
    assert out["card4_freq"].iloc[0] == 0
    assert list(out.columns) == spec.columns


def test_missing_optional_field_is_nan_not_error(spec_env, raw):
    """The API request model omits most of the 434 columns; those become NaN."""
    spec = features.fit(raw)
    partial = raw.iloc[[0]].drop(columns=["C1", "D1"])

    out = features.transform(partial, spec)

    assert out["C1"].isna().all()
    assert list(out.columns) == spec.columns


def test_leaky_columns_never_reach_the_model(spec_env, raw):
    """isFraud is the label; TransactionDT would leak the train/val boundary."""
    spec = features.fit(raw)
    assert "isFraud" not in spec.columns
    assert "TransactionDT" not in spec.columns
    assert "TransactionID" not in spec.columns


def test_email_provider_collapses_regional_variants(spec_env, raw):
    """'gmail.com' and 'gmail' are one provider, so they share a frequency."""
    spec = features.fit(raw)
    freq = spec.freq_maps["P_emaildomain_provider"]
    assert freq["gmail"] == 3  # gmail.com x2 + gmail x1


def test_split_is_chronological(spec_env, raw):
    """Later rows must never appear in training."""
    train, val = features.split(raw, 300000)
    assert train["TransactionDT"].max() < val["TransactionDT"].min()
    assert len(train) + len(val) == len(raw)


def test_miscategorised_column_raises_instead_of_silently_nanning(spec_env, raw, tmp_path, monkeypatch):
    """A string column left out of `categorical` used to become an all-NaN
    feature. The audit's dtype check regressed exactly this way once."""
    broken = dict(spec_env, categorical=[])  # ProductCD/card4 now look numeric
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    monkeypatch.setattr(features, "COLUMNS_PATH", path)

    with pytest.raises(ValueError, match="ProductCD"):
        features.fit(raw)
