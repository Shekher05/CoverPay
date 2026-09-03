"""Phase 11 - the behaviour engine.

    python -m fraud_engine.behavioural --rows 4000

The transaction model asks "does this look like fraud?". This asks a different
question the model cannot: "is this unusual *for this customer*?" A 4,000 rupee
payment is unremarkable in general and alarming from someone who has never spent
above 300.

Strictly causal. Each transaction is judged against only the transactions that
came before it, and state is updated afterwards. Peeking at the whole history -
including the future - is the behavioural equivalent of a random train/test
split, and would make every number here a lie.

This complements the model, it does not replace it. It emits evidence; the
advisory layer decides what to do with it.

Deliberately entity-scoped: signals concern one customer or one device.
Merchant-wide and cross-entity patterns belong to the spike engine (Phase 12).
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

from config import ROOT

# A customer needs some history before "unusual for them" means anything.
MIN_HISTORY = 3

# Robust z-score threshold. Median/MAD rather than mean/std because a single
# huge fraud would inflate the standard deviation and hide itself.
Z_THRESHOLD = 3.5
MAD_TO_SIGMA = 1.4826  # makes MAD comparable to a standard deviation

VELOCITY_WINDOW_SECONDS = 300
VELOCITY_THRESHOLD = 5  # prior transactions by one customer inside the window

# Severity weights. Policy, not model: tune per merchant risk appetite.
WEIGHTS = {
    "amount_anomaly": 0.45,
    "velocity": 0.40,
    "new_device": 0.35,
}


@dataclass
class BehaviourSignal:
    name: str
    severity: float  # 0..1
    detail: str


@dataclass
class BehaviourResult:
    score: float  # 0..1, combined
    signals: list[BehaviourSignal] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "; ".join(s.detail for s in self.signals) or "nothing unusual"


@dataclass
class _CustomerHistory:
    """What we know about one customer from transactions already seen."""

    amounts: list[float] = field(default_factory=list)
    recent: deque = field(default_factory=deque)  # TransactionDT values
    devices: set = field(default_factory=set)

    def observe(self, dt: int, amount: float, device: str) -> None:
        self.amounts.append(amount)
        self.recent.append(dt)
        self.devices.add(device)

    def prune(self, dt: int) -> None:
        while self.recent and dt - self.recent[0] > VELOCITY_WINDOW_SECONDS:
            self.recent.popleft()


def _amount_anomaly(history: _CustomerHistory, amount: float) -> BehaviourSignal | None:
    """How far this amount sits from the customer's own normal."""
    if len(history.amounts) < MIN_HISTORY:
        return None

    # ponytail: recomputed per transaction. Fine at stream scale (a customer has
    # tens of payments); swap for a streaming quantile if histories grow long.
    median = statistics.median(history.amounts)
    mad = statistics.median([abs(a - median) for a in history.amounts])

    if mad == 0:
        # Every past payment was identical. Any change is notable, but without
        # spread there is no scale, so report a fixed moderate severity.
        if amount == median:
            return None
        z = Z_THRESHOLD
    else:
        z = abs(amount - median) / (mad * MAD_TO_SIGMA)

    if z < Z_THRESHOLD:
        return None

    direction = "above" if amount > median else "below"
    return BehaviourSignal(
        name="amount_anomaly",
        severity=min(z / (Z_THRESHOLD * 3), 1.0),
        detail=(
            f"amount {amount:,.2f} is {z:.1f} deviations {direction} this "
            f"customer's usual {median:,.2f}"
        ),
    )


def _velocity(history: _CustomerHistory) -> BehaviourSignal | None:
    """Transactions bunched into a few minutes look automated."""
    count = len(history.recent)
    if count < VELOCITY_THRESHOLD:
        return None
    return BehaviourSignal(
        name="velocity",
        severity=min(count / (VELOCITY_THRESHOLD * 4), 1.0),
        detail=(
            f"{count + 1} transactions in {VELOCITY_WINDOW_SECONDS // 60} minutes "
            "from one customer"
        ),
    )


def _new_device(history: _CustomerHistory, device: str) -> BehaviourSignal | None:
    """A device never used before, by a customer with an established pattern."""
    if len(history.amounts) < MIN_HISTORY or device in history.devices:
        return None
    return BehaviourSignal(
        name="new_device",
        severity=0.6,
        detail=f"device {device} never seen for this customer before",
    )


def combine(signals: list[BehaviourSignal]) -> float:
    """Fold signals into one 0..1 score.

    Noisy-OR rather than a sum: two independent weak signals should raise
    suspicion without three of them mechanically pinning the score at 1.0.
    """
    remaining = 1.0
    for signal in signals:
        remaining *= 1.0 - WEIGHTS.get(signal.name, 0.2) * signal.severity
    return round(1.0 - remaining, 6)


def analyse_stream(df: pd.DataFrame) -> pd.DataFrame:
    """Score every transaction against the history preceding it.

    Returns one row per input row, in the input's order, carrying the behaviour
    score and its supporting evidence.
    """
    required = {"customer_id", "device_id", "TransactionDT", "TransactionAmt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"stream is missing {sorted(missing)}")

    ordered = df.sort_values("TransactionDT", kind="stable")
    histories: dict[str, _CustomerHistory] = defaultdict(_CustomerHistory)
    rows = []

    for row in ordered.itertuples(index=True):
        customer, device = row.customer_id, row.device_id
        dt, amount = int(row.TransactionDT), float(row.TransactionAmt)

        history = histories[customer]
        history.prune(dt)

        # Judged before observing: a transaction is not part of its own history.
        signals = [
            signal
            for signal in (
                _amount_anomaly(history, amount),
                _velocity(history),
                _new_device(history, device),
            )
            if signal is not None
        ]
        history.observe(dt, amount, device)

        rows.append(
            {
                "index": row.Index,
                "behaviour_score": combine(signals),
                "behaviour_signals": [s.name for s in signals],
                "behaviour_detail": BehaviourResult(0.0, signals).text,
            }
        )

    out = pd.DataFrame(rows).set_index("index")
    return out.reindex(df.index)  # back into the caller's original order


def _evaluate(stream: pd.DataFrame, analysed: pd.DataFrame, threshold: float) -> str:
    """Measure against the simulator's ground truth, per scenario."""
    merged = stream.join(analysed)
    flagged = merged["behaviour_score"] >= threshold
    injected = merged["synthetic_is_fraud"] == 1

    tp = int((flagged & injected).sum())
    fp = int((flagged & ~injected).sum())
    fn = int((~flagged & injected).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    lines = [
        "# Behaviour Engine Evaluation",
        "",
        f"Generated by `python -m fraud_engine.behavioural`. Threshold {threshold}.",
        "",
        "Measured on the seeded synthetic stream, where every injected scenario is "
        "known. These are behavioural signals only - the ML model is not involved.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Transactions | {len(merged):,} |",
        f"| Injected fraud | {int(injected.sum()):,} |",
        f"| Flagged | {int(flagged.sum()):,} |",
        f"| Precision | {precision:.3f} |",
        f"| Recall | {recall:.3f} |",
        f"| False positives | {fp:,} of {int((~injected).sum()):,} legitimate |",
        "",
        "## Detection by scenario",
        "",
        "| Scenario | Transactions | Flagged | Rate |",
        "|---|---|---|---|",
    ]
    for name, group in merged.groupby("synthetic_scenario"):
        hit = int((group["behaviour_score"] >= threshold).sum())
        lines.append(f"| {name} | {len(group):,} | {hit:,} | {hit / len(group):.3f} |")

    lines += [
        "",
        "## Signal frequency",
        "",
        "| Signal | Fires on fraud | Fires on normal |",
        "|---|---|---|",
    ]
    for signal in WEIGHTS:
        fires = merged["behaviour_signals"].apply(lambda s, n=signal: n in s)
        lines.append(
            f"| {signal} | {int((fires & injected).sum()):,} | "
            f"{int((fires & ~injected).sum()):,} |"
        )

    lines += [
        "",
        "## Reading this honestly",
        "",
        "`coordinated_spike` is expected to score poorly here. Each of its "
        "transactions is unremarkable for the individual customer - the pattern "
        "only exists across many customers at one merchant, which is the spike "
        "engine's job, not this one's.",
        "",
        "`card_testing` should be caught by velocity, and `account_takeover` by the "
        "new-device and amount signals. If those two are missed, the engine is not "
        "working.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    from simulator.world import generate_stream

    parser = argparse.ArgumentParser(description="Evaluate the behaviour engine.")
    parser.add_argument("--rows", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    print(f"Generating {args.rows:,} transactions...", flush=True)
    stream = generate_stream(args.rows, "evaluation", args.seed, scenario_rate=0.05)

    print("Analysing behaviour...", flush=True)
    analysed = analyse_stream(stream)

    report = _evaluate(stream, analysed, args.threshold)
    out = ROOT / "docs" / "behaviour.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
