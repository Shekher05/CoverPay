"""Synthetic transactions for exercising the scoring path end to end.

    python -m simulator.generate --rows 2000 --score

Method: real rows are used as *templates* and their entity identifiers are
replaced with synthetic cards, addresses and emails, amounts are jittered, and
timestamps are re-sequenced. Template sampling is what keeps a row scoreable -
the model reads ~270 features whose joint structure a per-column random
generator would destroy, which is exactly why a 13-field hand-written payload
scores near zero.

What this is NOT: it does not invent new fraud structure, so it cannot be used
to claim detection performance. Its job is to exercise serving, persistence and
throughput with full-width rows, and to give the Phase 7 simulator a baseline.
Real detection numbers come from `docs/metrics.md` only.

`synthetic_is_fraud` and `synthetic_scenario` are evaluation metadata. They are
never model inputs - `transform()` selects `spec.columns`, which excludes them.
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np
import pandas as pd

from config import ROOT, settings
from ml.features import load_raw, transform

OUT_DIR = ROOT / "data" / "synthetic"
OUT_PATH = OUT_DIR / "transactions.csv"

# Ground truth, kept beside the data and out of the feature matrix.
SYNTHETIC_LABEL_COLUMNS = ["synthetic_is_fraud", "synthetic_scenario"]

# Enough rows to hold a few thousand fraud templates without loading 1.8GB.
TEMPLATE_ROWS = 120_000

SECONDS_BETWEEN_TRANSACTIONS = 37  # ~2,300 transactions/day, a small merchant


def _synthetic_entities(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Fresh card/address/email identifiers, drawn from small pools so repeat
    customers exist - frequency encoding is meaningless if every card is seen
    exactly once."""
    cards = rng.integers(100_000, 999_999, size=max(n // 12, 1))
    addrs = rng.integers(100, 599, size=max(n // 40, 1)).astype(float)
    domains = np.array(
        ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com"]
    )
    return {
        "card1": rng.choice(cards, size=n),
        "addr1": rng.choice(addrs, size=n),
        "P_emaildomain": rng.choice(domains, size=n, p=[0.55, 0.2, 0.12, 0.09, 0.04]),
    }


@lru_cache(maxsize=1)
def template_pools() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(legit, fraud) template rows, loaded once per process.

    Every synthetic row borrows the ~380 V/C/D/M/id values it cannot plausibly
    invent from one of these. Cached because reading 120k rows takes seconds and
    both generators need it.
    """
    templates = load_raw(nrows=TEMPLATE_ROWS)
    legit = templates[templates["isFraud"] == 0]
    fraud = templates[templates["isFraud"] == 1]
    if fraud.empty:
        raise RuntimeError(f"No fraud rows in the first {TEMPLATE_ROWS:,} templates")
    return legit, fraud


def generate_batch(
    rows: int = 2_000, fraud_rate: float = 0.035, seed: int | None = None
) -> pd.DataFrame:
    """Build `rows` synthetic transactions. Seeded runs are reproducible."""
    rng = np.random.default_rng(settings.random_seed if seed is None else seed)

    legit_pool, fraud_pool = template_pools()
    n_fraud = int(round(rows * fraud_rate))
    picks = pd.concat(
        [
            legit_pool.sample(
                rows - n_fraud, replace=True, random_state=int(rng.integers(1e9))
            ),
            fraud_pool.sample(
                n_fraud, replace=True, random_state=int(rng.integers(1e9))
            ),
        ]
    )

    out = picks.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)

    # Appended in one concat; the remaining assignments overwrite columns that
    # already exist, so they do not fragment the 434-column frame.
    labels = (out.pop("isFraud") == 1).astype(int).to_numpy()
    out = pd.concat(
        [
            out,
            pd.DataFrame(
                {
                    "synthetic_is_fraud": labels,
                    "synthetic_scenario": np.where(
                        labels == 1, "fraud_template", "normal"
                    ),
                },
                index=out.index,
            ),
        ],
        axis=1,
    )

    out["TransactionID"] = np.arange(9_000_000, 9_000_000 + len(out))
    # A steady arrival sequence. Still a seconds offset, never a wall-clock time.
    out["TransactionDT"] = int(legit_pool["TransactionDT"].max()) + np.arange(
        1, len(out) + 1
    ) * SECONDS_BETWEEN_TRANSACTIONS
    # +/-30% jitter so amounts are not verbatim copies of real transactions.
    out["TransactionAmt"] = (
        out["TransactionAmt"].to_numpy() * rng.uniform(0.7, 1.3, size=len(out))
    ).round(2)

    for column, values in _synthetic_entities(rng, len(out)).items():
        out[column] = values

    return out


def score_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score every row and attach the advisory outcome."""
    from fraud_engine.advisory import classify
    from ml.inference import get_model

    model, spec = get_model()
    scores = model.predict_proba(transform(df, spec))[:, 1]

    scored = df[["TransactionID", "TransactionAmt", *SYNTHETIC_LABEL_COLUMNS]].copy()
    scored["risk_score"] = scores
    outcomes = [classify(float(s)) for s in scores]
    scored["risk_level"] = [str(level) for level, _ in outcomes]
    scored["recommendation"] = [str(rec) for _, rec in outcomes]
    return scored


def _summarise(scored: pd.DataFrame) -> None:
    total = len(scored)
    injected = scored["synthetic_is_fraud"] == 1
    flagged = scored["recommendation"] != "ALLOW"
    n_injected = int(injected.sum())

    print(f"\nScored {total:,} synthetic transactions")
    print(f"  injected fraud     {n_injected:,} ({injected.mean() * 100:.2f}%)")
    print("\n  recommendation breakdown")
    for name, count in scored["recommendation"].value_counts().items():
        print(f"    {name:<7} {count:>7,} ({count / total * 100:5.2f}%)")

    caught = int((injected & flagged).sum())
    print(
        f"\n  injected fraud flagged   {caught:,} of {n_injected:,} "
        f"({caught / max(n_injected, 1) * 100:.1f}%)"
    )
    print(
        f"  legit sent to review     {int((~injected & flagged).sum()):,} "
        f"of {int((~injected).sum()):,}"
    )
    print(f"  value at risk (flagged)  {scored.loc[flagged, 'TransactionAmt'].sum():,.0f}")
    print(
        "\nThese are template-resampled rows, not novel fraud. Treat this as a "
        "pipeline check, not a detection benchmark."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic transactions.")
    parser.add_argument("--rows", type=int, default=2_000)
    parser.add_argument("--fraud-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--score", action="store_true", help="score after generating")
    args = parser.parse_args()

    print(f"Generating {args.rows:,} synthetic transactions...", flush=True)
    batch = generate_batch(args.rows, args.fraud_rate, args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    batch.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({batch.shape[0]:,} rows x {batch.shape[1]} cols)")

    if args.score:
        _summarise(score_batch(batch))


if __name__ == "__main__":
    main()
