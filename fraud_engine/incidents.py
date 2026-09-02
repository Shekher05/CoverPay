"""Phase 12 - spike detection and incident management.

    python -m fraud_engine.incidents --rows 4000

The behaviour engine asks "is this unusual for this customer?" and by design
cannot see a coordinated attack: twenty customers each making one ordinary-
looking payment at the same merchant is unremarkable per customer and obvious in
aggregate. That gap is measured, not asserted - the behaviour engine flags 8% of
coordinated spikes.

This engine watches *merchants* over rolling windows and asks a different
question: is activity here suddenly unlike this merchant's own recent normal?

Detection is rate-based rather than suspicion-based. A spike is a spike whether
or not the individual transactions look fraudulent, which is precisely why this
catches what the other two miss.

Causal, like the behaviour engine: a window only ever contains transactions that
have already arrived.

Incidents follow the lifecycle the PRD describes:

    NORMAL -> SUSPICIOUS -> ACTIVE -> RESOLVED
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from config import ROOT

WINDOW_SECONDS = 300  # the "what is happening now" window
BASELINE_SECONDS = 3600  # what this merchant normally does
COOLDOWN_SECONDS = 900  # quiet time before an incident is resolved

# A merchant must be doing *something* before a ratio means anything: 3
# transactions where there is usually 0.5 is noise, not an attack.
MIN_WINDOW_COUNT = 8
RATE_FACTOR = 4.0  # window rate over baseline rate

# Many distinct customers at once is the coordinated-attack fingerprint.
MIN_CUSTOMER_FANOUT = 6

# Sustained activity promotes SUSPICIOUS to ACTIVE.
ACTIVE_COUNT = 20


class IncidentStatus(StrEnum):
    SUSPICIOUS = "SUSPICIOUS"
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Incident:
    """A window of coordinated activity at one merchant."""

    incident_id: str
    merchant_id: str
    opened_dt: int
    last_dt: int
    status: IncidentStatus = IncidentStatus.SUSPICIOUS
    transactions: int = 0
    customers: set = field(default_factory=set)
    devices: set = field(default_factory=set)
    total_value: float = 0.0
    risky_value: float = 0.0
    peak_rate_ratio: float = 0.0
    peak_fanout: int = 0

    @property
    def duration_seconds(self) -> int:
        return self.last_dt - self.opened_dt

    @property
    def severity(self) -> Severity:
        """Severity blends how abnormal, how broad, and how much money."""
        score = 0
        score += self.peak_rate_ratio >= RATE_FACTOR * 2
        score += self.peak_fanout >= MIN_CUSTOMER_FANOUT * 2
        score += self.transactions >= ACTIVE_COUNT
        score += self.risky_value > 10_000
        return [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL][
            min(score, 3)
        ]

    def summary(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "merchant_id": self.merchant_id,
            "status": str(self.status),
            "severity": str(self.severity),
            "opened_dt": self.opened_dt,
            "last_dt": self.last_dt,
            "duration_seconds": self.duration_seconds,
            "transactions": self.transactions,
            "customers": len(self.customers),
            "devices": len(self.devices),
            "total_value": round(self.total_value, 2),
            "risky_value": round(self.risky_value, 2),
            "peak_rate_ratio": round(self.peak_rate_ratio, 2),
            "peak_fanout": self.peak_fanout,
        }


@dataclass
class _MerchantWindow:
    """Rolling activity for one merchant. Two deques: the short window that
    detects, and the long one that defines normal."""

    window: deque = field(default_factory=deque)  # (dt, customer, device)
    baseline: deque = field(default_factory=deque)  # dt only
    incident: Incident | None = None

    def prune(self, dt: int) -> None:
        while self.window and dt - self.window[0][0] > WINDOW_SECONDS:
            self.window.popleft()
        while self.baseline and dt - self.baseline[0] > BASELINE_SECONDS:
            self.baseline.popleft()

    def rate_ratio(self) -> float:
        """Window transactions-per-second against the baseline's.

        The baseline includes the window, which is fine: it makes the ratio
        conservative rather than flattering.
        """
        window_rate = len(self.window) / WINDOW_SECONDS
        baseline_rate = len(self.baseline) / BASELINE_SECONDS
        if baseline_rate == 0:
            return float("inf") if window_rate else 0.0
        return window_rate / baseline_rate

    def fanout(self) -> int:
        return len({customer for _, customer, _ in self.window})


def detect_incidents(
    df: pd.DataFrame, risk_column: str | None = None, risk_threshold: float = 0.5
) -> tuple[pd.DataFrame, list[Incident]]:
    """Walk the stream in time order, opening and closing incidents.

    Returns per-row incident assignment (in the caller's row order) and the
    incidents themselves.
    """
    required = {
        "merchant_id",
        "customer_id",
        "device_id",
        "TransactionDT",
        "TransactionAmt",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"stream is missing {sorted(missing)}")

    ordered = df.sort_values("TransactionDT", kind="stable")
    merchants: dict[str, _MerchantWindow] = defaultdict(_MerchantWindow)
    incidents: list[Incident] = []
    rows = []
    counter = 0

    for row in ordered.itertuples(index=True):
        merchant = row.merchant_id
        dt = int(row.TransactionDT)
        amount = float(row.TransactionAmt)
        risky = (
            risk_column is not None
            and float(getattr(row, risk_column, 0.0) or 0.0) >= risk_threshold
        )

        state = merchants[merchant]
        state.prune(dt)

        # Observe first: a spike is a property of the window *including* this
        # transaction, unlike a per-customer anomaly which needs a prior baseline.
        state.window.append((dt, row.customer_id, row.device_id))
        state.baseline.append(dt)

        count = len(state.window)
        ratio = state.rate_ratio()
        fanout = state.fanout()
        spiking = count >= MIN_WINDOW_COUNT and (
            ratio >= RATE_FACTOR or fanout >= MIN_CUSTOMER_FANOUT
        )

        incident = state.incident
        if incident is not None and dt - incident.last_dt > COOLDOWN_SECONDS:
            incident.status = IncidentStatus.RESOLVED
            state.incident = incident = None

        if spiking:
            if incident is None:
                counter += 1
                incident = Incident(
                    incident_id=f"INC_{counter:04d}",
                    merchant_id=merchant,
                    opened_dt=dt,
                    last_dt=dt,
                )
                state.incident = incident
                incidents.append(incident)

            incident.last_dt = dt
            incident.transactions += 1
            incident.customers.add(row.customer_id)
            incident.devices.add(row.device_id)
            incident.total_value += amount
            if risky:
                incident.risky_value += amount
            incident.peak_rate_ratio = max(incident.peak_rate_ratio, ratio)
            incident.peak_fanout = max(incident.peak_fanout, fanout)
            if incident.transactions >= ACTIVE_COUNT:
                incident.status = IncidentStatus.ACTIVE

        rows.append(
            {
                "index": row.Index,
                "incident_id": incident.incident_id if spiking and incident else "",
                "window_count": count,
                "rate_ratio": round(ratio, 3) if ratio != float("inf") else -1.0,
                "customer_fanout": fanout,
            }
        )

    # Anything still open at the end of the stream simply has not resolved yet.
    per_row = pd.DataFrame(rows).set_index("index").reindex(df.index)
    return per_row, incidents


def _evaluate(
    stream: pd.DataFrame, per_row: pd.DataFrame, incidents: list[Incident]
) -> str:
    """Score detected incidents against the simulator's injected ones."""
    merged = stream.join(per_row)
    detected = merged["incident_id"] != ""
    injected = merged["synthetic_scenario_id"] != ""

    truth_ids = set(merged.loc[injected, "synthetic_scenario_id"])
    caught = {
        sid
        for sid, group in merged[injected].groupby("synthetic_scenario_id")
        if (group["incident_id"] != "").any()
    }

    tp = int((detected & injected).sum())
    fp = int((detected & ~injected).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0

    lines = [
        "# Spike and Incident Engine Evaluation",
        "",
        "Generated by `python -m fraud_engine.incidents`. Measured on the seeded "
        "synthetic stream, where every injected scenario is known.",
        "",
        "Detection here is **rate-based, not suspicion-based**: the engine reacts to "
        "a merchant's activity departing from its own recent normal, whether or not "
        "the individual transactions look fraudulent.",
        "",
        "## Incident-level detection",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Injected incidents | {len(truth_ids):,} |",
        f"| Injected incidents detected | {len(caught):,} |",
        f"| Incident recall | {len(caught) / max(len(truth_ids), 1):.3f} |",
        f"| Incidents raised | {len(incidents):,} |",
        "",
        "## Transaction-level",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Transactions in a detected incident | {int(detected.sum()):,} |",
        f"| Precision against injected scenarios | {precision:.3f} |",
        f"| Flagged but not part of an injected scenario | {fp:,} |",
        "",
        "## Detection by scenario",
        "",
        "| Scenario | Incidents injected | Detected | Recall |",
        "|---|---|---|---|",
    ]
    for name, group in merged[injected].groupby("synthetic_scenario"):
        ids = set(group["synthetic_scenario_id"])
        hit = len(ids & caught)
        lines.append(f"| {name} | {len(ids):,} | {hit:,} | {hit / len(ids):.3f} |")

    lines += [
        "",
        "## Incident severity mix",
        "",
        "| Severity | Incidents |",
        "|---|---|",
    ]
    for name, count in pd.Series(
        [str(i.severity) for i in incidents]
    ).value_counts().items():
        lines.append(f"| {name} | {count:,} |")

    lines += [
        "",
        "## Largest incidents",
        "",
        "| Incident | Merchant | Status | Severity | Txns | Customers | Value |",
        "|---|---|---|---|---|---|---|",
    ]
    for incident in sorted(incidents, key=lambda i: -i.transactions)[:10]:
        s = incident.summary()
        lines.append(
            f"| {s['incident_id']} | {s['merchant_id']} | {s['status']} | "
            f"{s['severity']} | {s['transactions']:,} | {s['customers']:,} | "
            f"{s['total_value']:,.0f} |"
        )

    lines += [
        "",
        "## What this engine is for",
        "",
        "Compare against `behaviour.md`: the behaviour engine detects roughly 8% of "
        "coordinated spikes because each transaction is ordinary for its own "
        "customer. Cross-entity concentration is only visible from the merchant's "
        "vantage point, which is what this engine watches.",
        "",
        "The cost is the reverse blind spot. A rate-based detector cannot tell an "
        "attack from a genuine surge - a flash sale looks identical. That is why "
        "incidents are advisory evidence for an analyst, not an automatic action.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    from fraud_engine.behavioural import analyse_stream
    from simulator.world import generate_stream

    parser = argparse.ArgumentParser(description="Evaluate the incident engine.")
    parser.add_argument("--rows", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"Generating {args.rows:,} transactions...", flush=True)
    stream = generate_stream(args.rows, "evaluation", args.seed, scenario_rate=0.05)

    print("Analysing behaviour...", flush=True)
    stream = stream.join(analyse_stream(stream))

    print("Detecting spikes...", flush=True)
    per_row, incidents = detect_incidents(stream, risk_column="behaviour_score")

    report = _evaluate(stream, per_row, incidents)
    out = ROOT / "docs" / "incidents.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
