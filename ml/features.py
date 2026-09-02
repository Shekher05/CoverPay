"""Phase 2 - the one feature pipeline.

Training and inference both call `transform()`. If the two ever diverge, the
model looks fine offline and scores garbage in production, so this module is
the single definition of what a feature is. Nothing else may build features.

Column selection is not decided here - it is read from `feature_columns.json`,
written by `ml/audit.py`. Run the audit first.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT, settings

COLUMNS_PATH = ROOT / "ml" / "feature_columns.json"
SECONDS_PER_DAY = 86400

# High-cardinality numeric identifiers. Their integer values are arbitrary
# labels, but *how often a value appears* is strong signal: a card seen once in
# 590k rows behaves very differently from one seen 5,000 times.
COUNT_ENCODE = ["card1", "addr1"]

# Missing is signal on this dataset (only 24% of rows carry identity data), so
# it gets its own bucket rather than being imputed away.
MISSING = "__missing__"


def load_raw(train: bool = True, nrows: int | None = None) -> pd.DataFrame:
    """Load and join the transaction + identity CSVs.

    The V/C/D blocks are ~380 of the 434 columns; as float64 they cost ~1.9GB,
    as float32 about half. Categoricals stay object. `nrows` caps the read for
    callers that only need a sample.
    """
    prefix = "train" if train else "test"
    tx_path = settings.raw / f"{prefix}_transaction.csv"
    id_path = settings.raw / f"{prefix}_identity.csv"
    if not tx_path.exists():
        sys.exit(f"Not found: {tx_path}\nSet DATA_DIR in .env to the IEEE-CIS folder.")

    header = pd.read_csv(tx_path, nrows=0).columns
    wide = [c for c in header if c[0] in "VCD" and c[1:].isdigit()]
    dtypes = {c: "float32" for c in wide}
    dtypes["TransactionAmt"] = "float32"
    if "isFraud" in header:
        dtypes["isFraud"] = "int8"

    tx = pd.read_csv(tx_path, dtype=dtypes, nrows=nrows)
    ident = pd.read_csv(id_path)
    return tx.merge(ident, on="TransactionID", how="left")


@dataclass
class FeatureSpec:
    """Everything fitted on training data that inference must reproduce exactly.

    Saved beside the model artifact. A model without its spec cannot be served.
    """

    columns: list[str]  # exact model input order
    freq_maps: dict[str, dict[str, int]]
    split_dt: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FeatureSpec:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def read_column_decisions() -> dict:
    if not COLUMNS_PATH.exists():
        sys.exit(f"Not found: {COLUMNS_PATH}\nRun `python -m ml.audit` first.")
    return json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))


def _keys(s: pd.Series) -> pd.Series:
    """Stable string keys for frequency lookup.

    JSON keys are strings, and a value must produce the same key when fitting on
    a full column and when scoring a single API row. Nullable Int64 pins whole
    numbers so 13926 never arrives as "13926.0" on one side and "13926" on the
    other.
    """
    if pd.api.types.is_bool_dtype(s):
        s = s.astype("object")
    elif pd.api.types.is_float_dtype(s):
        finite = s.dropna()
        if not finite.empty and (finite % 1 == 0).all():
            s = s.astype("Int64")
    elif pd.api.types.is_integer_dtype(s):
        s = s.astype("Int64")
    return s.astype("string").fillna(MISSING)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns. Pure - no fitted state, safe on a single row."""
    out = df.copy()

    # Amounts span 0.25 to 31,937 and are heavily right-skewed.
    out["log_amount"] = np.log1p(out["TransactionAmt"])

    # TransactionDT is a seconds offset, not a timestamp. The offset itself is
    # excluded as a feature (it would let the model memorise the train/val
    # boundary), but time-of-day and weekday within it are legitimate.
    dt = out["TransactionDT"]
    out["hour_of_day"] = (dt // 3600) % 24
    out["day_of_week"] = (dt // SECONDS_PER_DAY) % 7

    # "gmail" and "gmail.com" are the same provider; the first token collapses
    # regional variants without a hand-maintained lookup table.
    for col in ("P_emaildomain", "R_emaildomain"):
        if col in out.columns:
            out[f"{col}_provider"] = out[col].astype("string").str.split(".").str[0]

    return out


def fit(df: pd.DataFrame) -> FeatureSpec:
    """Fit on TRAINING ROWS ONLY. Passing the full dataset leaks the future."""
    decisions = read_column_decisions()
    derived = derive(df)

    freq_cols = decisions["categorical"] + COUNT_ENCODE
    freq_cols += [c for c in derived.columns if c.endswith("_provider")]
    freq_cols = [c for c in dict.fromkeys(freq_cols) if c in derived.columns]

    freq_maps = {
        col: {k: int(v) for k, v in _keys(derived[col]).value_counts().items()}
        for col in freq_cols
    }

    # Final input order: audited survivors that are not raw categoricals, plus
    # one `_freq` column per frequency-encoded field, plus the derived numerics.
    numeric = [
        c
        for c in decisions["keep"]
        if c not in decisions["categorical"] and c in derived.columns
    ]
    # `transform` coerces these with errors="coerce", so a categorical that slips
    # through here becomes an all-NaN column and the model quietly loses a real
    # signal. Fail loudly instead - this has bitten once already.
    leaked = [c for c in numeric if not pd.api.types.is_numeric_dtype(derived[c])]
    if leaked:
        raise ValueError(
            f"Non-numeric columns not listed as categorical: {leaked}. "
            "Re-run `python -m ml.audit` to regenerate feature_columns.json."
        )
    columns = (
        numeric
        + [f"{c}_freq" for c in freq_cols]
        + ["log_amount", "hour_of_day", "day_of_week"]
    )
    return FeatureSpec(
        columns=columns, freq_maps=freq_maps, split_dt=float(decisions["split_dt"])
    )


def transform(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Raw rows -> model matrix. Identical path for training and inference."""
    derived = derive(df)

    # Built as whole blocks and concatenated once. Assigning ~250 columns one at
    # a time fragments the frame, and this runs on every scoring request.
    encoded = {}
    for col, freq in spec.freq_maps.items():
        if col in derived.columns:
            # Unseen value -> 0. A card never seen in training is itself
            # informative, and it must not raise at inference time.
            encoded[f"{col}_freq"] = _keys(derived[col]).map(freq).fillna(0).astype("int32")
        else:
            encoded[f"{col}_freq"] = 0

    blocks = [derived, pd.DataFrame(encoded, index=derived.index)]
    absent = [c for c in spec.columns if c not in derived.columns and c not in encoded]
    if absent:  # caller omitted optional fields
        blocks.append(pd.DataFrame(np.nan, index=derived.index, columns=absent))

    out = pd.concat(blocks, axis=1)[spec.columns]
    # XGBoost needs numbers; every categorical became a _freq column above.
    return out.apply(pd.to_numeric, errors="coerce").astype("float32")


def split(df: pd.DataFrame, split_dt: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split. A random split leaks card history across the boundary."""
    return df[df["TransactionDT"] < split_dt], df[df["TransactionDT"] >= split_dt]
