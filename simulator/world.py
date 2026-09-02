"""Phase 7 - the synthetic payment environment.

    python -m simulator.world --rows 5000 --mode evaluation --score

Where `generate.py` produces independent rows, this produces a *stream*: a
fixed population of customers, merchants and devices whose transactions recur
over time, so entities accumulate history. That history is the point. Frequency
encoding is meaningless when every card appears once, and the behaviour engine
in the next phase can only ask "is this unusual for this customer?" if the
customer existed before now.

Two modes, as the project plan requires:

    evaluation  seeded, reproducible, known ground truth
    live        randomly timed scenarios of varying severity

Three scenario families are injected, matching the plan's fraud classes:

    card_testing        one card, a burst of small payments (automated attack)
    account_takeover    an established customer, sudden new device and value
    coordinated_spike   many customers hitting one merchant at once

Honest limit: the ~380 V/C/D/M/id columns are borrowed from real template rows,
because no generator can invent their joint structure. The transaction model
therefore reacts mostly to the template, not to the scenario. The scenarios are
built for the behaviour and spike engines, which read entity history rather than
V columns - they are not evidence that the ML model detects these attacks.

Every `synthetic_*` column is evaluation metadata and never a model input.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import ROOT, settings
from simulator.generate import score_batch, template_pools

OUT_PATH = ROOT / "data" / "synthetic" / "stream.csv"

SCENARIOS = ("card_testing", "account_takeover", "coordinated_spike")

# IEEE-CIS product codes, so frequency encoding sees values it was trained on.
PRODUCT_CODES = ("W", "C", "H", "R", "S")
EMAIL_DOMAINS = ("gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com")
DEVICE_TYPES = ("desktop", "mobile")
DEVICE_INFO = ("Windows", "MacOS", "iOS Device", "Trident/7.0", "SAMSUNG SM-G930V")

MEAN_GAP_SECONDS = 40  # normal inter-arrival time
BURST_GAP_SECONDS = 3  # inside an automated attack
START_DT = 16_000_000  # past the real data's 15.8M, so streams never overlap it


@dataclass(frozen=True)
class Customer:
    customer_id: str
    card1: int
    addr1: float
    email: str
    device_id: str
    device_type: str
    device_info: str
    base_amount: float
    activity: float  # relative transaction frequency


@dataclass(frozen=True)
class Merchant:
    merchant_id: str
    product_cd: str
    amount_multiplier: float


@dataclass(frozen=True)
class World:
    customers: list[Customer]
    merchants: list[Merchant]

    @property
    def weights(self) -> np.ndarray:
        activity = np.array([c.activity for c in self.customers])
        return activity / activity.sum()


def build_world(
    rng: np.random.Generator, customers: int = 400, merchants: int = 12
) -> World:
    """A population small enough that entities genuinely repeat.

    400 customers over a few thousand transactions gives each one a handful of
    payments - enough history for a behavioural baseline to exist.
    """
    people = [
        Customer(
            customer_id=f"CUST_{i:05d}",
            card1=int(rng.integers(100_000, 999_999)),
            addr1=float(rng.integers(100, 599)),
            email=str(rng.choice(EMAIL_DOMAINS, p=[0.55, 0.2, 0.12, 0.09, 0.04])),
            device_id=f"DEV_{i:05d}",
            device_type=str(rng.choice(DEVICE_TYPES, p=[0.6, 0.4])),
            device_info=str(rng.choice(DEVICE_INFO)),
            # Everyday payments: mostly small, with a long right tail.
            base_amount=float(np.clip(rng.lognormal(4.2, 0.7), 20, 15_000)),
            activity=float(rng.gamma(2.0, 1.0) + 0.1),
        )
        for i in range(customers)
    ]
    shops = [
        Merchant(
            merchant_id=f"MERCH_{i:03d}",
            product_cd=str(rng.choice(PRODUCT_CODES, p=[0.74, 0.12, 0.06, 0.06, 0.02])),
            amount_multiplier=float(rng.uniform(0.6, 2.2)),
        )
        for i in range(merchants)
    ]
    return World(customers=people, merchants=shops)


def _event(
    customer: Customer,
    merchant: Merchant,
    amount: float,
    dt: int,
    is_fraud: int,
    scenario: str,
    scenario_id: str,
    device_id: str | None = None,
    device_info: str | None = None,
) -> dict:
    return {
        "customer_id": customer.customer_id,
        "merchant_id": merchant.merchant_id,
        "device_id": device_id or customer.device_id,
        "card1": customer.card1,
        "addr1": customer.addr1,
        "P_emaildomain": customer.email,
        "DeviceType": customer.device_type,
        "DeviceInfo": device_info or customer.device_info,
        "ProductCD": merchant.product_cd,
        "TransactionAmt": round(max(amount, 0.5), 2),
        "TransactionDT": dt,
        "synthetic_is_fraud": is_fraud,
        "synthetic_scenario": scenario,
        "synthetic_scenario_id": scenario_id,
    }


def _normal(rng: np.random.Generator, world: World, dt: int) -> list[dict]:
    customer = world.customers[rng.choice(len(world.customers), p=world.weights)]
    merchant = world.merchants[rng.integers(len(world.merchants))]
    amount = customer.base_amount * merchant.amount_multiplier * rng.lognormal(0, 0.35)
    return [_event(customer, merchant, amount, dt, 0, "normal", "")]


def _card_testing(
    rng: np.random.Generator, world: World, dt: int, sid: str
) -> list[dict]:
    """One stolen card probed with many tiny payments in minutes."""
    customer = world.customers[rng.integers(len(world.customers))]
    merchant = world.merchants[rng.integers(len(world.merchants))]
    events = []
    for _ in range(int(rng.integers(15, 45))):
        dt += int(rng.integers(1, BURST_GAP_SECONDS * 2))
        events.append(
            _event(customer, merchant, rng.uniform(1, 25), dt, 1, "card_testing", sid)
        )
    return events


def _account_takeover(
    rng: np.random.Generator, world: World, dt: int, sid: str
) -> list[dict]:
    """An established customer, suddenly on a new device spending far more."""
    customer = world.customers[rng.integers(len(world.customers))]
    stolen_device = f"DEV_NEW_{rng.integers(10_000, 99_999)}"
    events = []
    for _ in range(int(rng.integers(2, 6))):
        dt += int(rng.integers(20, 180))
        merchant = world.merchants[rng.integers(len(world.merchants))]
        events.append(
            _event(
                customer,
                merchant,
                customer.base_amount * rng.uniform(6, 15),
                dt,
                1,
                "account_takeover",
                sid,
                device_id=stolen_device,
                device_info=str(rng.choice(DEVICE_INFO)),
            )
        )
    return events


def _coordinated_spike(
    rng: np.random.Generator, world: World, dt: int, sid: str
) -> list[dict]:
    """Many unrelated customers hitting one merchant in a short window."""
    merchant = world.merchants[rng.integers(len(world.merchants))]
    victims = rng.choice(
        len(world.customers), size=int(rng.integers(10, 30)), replace=False
    )
    events = []
    for index in victims:
        customer = world.customers[index]
        dt += int(rng.integers(1, 20))
        events.append(
            _event(
                customer,
                merchant,
                customer.base_amount * rng.uniform(1.5, 4.0),
                dt,
                1,
                "coordinated_spike",
                sid,
            )
        )
    return events


def _attach_templates(events: list[dict], rng: np.random.Generator) -> pd.DataFrame:
    """Borrow the wide feature columns a generator cannot invent."""
    legit_pool, fraud_pool = template_pools()
    frame = pd.DataFrame(events)
    is_fraud = frame["synthetic_is_fraud"].to_numpy() == 1
    n_fraud = int(is_fraud.sum())

    picks = pd.concat(
        [
            fraud_pool.sample(
                n_fraud, replace=True, random_state=int(rng.integers(1e9))
            ),
            legit_pool.sample(
                len(frame) - n_fraud, replace=True, random_state=int(rng.integers(1e9))
            ),
        ]
    ).reset_index(drop=True)

    # Line the templates up so fraud events receive fraud templates.
    order = np.empty(len(frame), dtype=int)
    order[np.flatnonzero(is_fraud)] = np.arange(n_fraud)
    order[np.flatnonzero(~is_fraud)] = np.arange(n_fraud, len(frame))
    picks = picks.iloc[order].reset_index(drop=True).drop(columns=["isFraud"])

    # Entity-driven columns win over whatever the template carried.
    picks = picks.drop(columns=[c for c in frame.columns if c in picks.columns])
    out = pd.concat([picks, frame], axis=1)
    out["TransactionID"] = np.arange(9_500_000, 9_500_000 + len(out))
    return out


def generate_stream(
    rows: int = 5_000,
    mode: str = "evaluation",
    seed: int | None = None,
    scenario_rate: float = 0.02,
) -> pd.DataFrame:
    """Emit `rows` transactions in time order with scenarios mixed in.

    `evaluation` seeds from settings for reproducibility; `live` draws a fresh
    seed each call so demo runs differ.
    """
    if mode not in {"evaluation", "live"}:
        raise ValueError(f"mode must be 'evaluation' or 'live', got {mode!r}")

    rng = np.random.default_rng(
        (settings.random_seed if seed is None else seed)
        if mode == "evaluation"
        else seed
    )

    world = build_world(rng)
    events: list[dict] = []
    dt = START_DT
    scenario_count = 0

    while len(events) < rows:
        if rng.random() < scenario_rate:
            scenario_count += 1
            sid = f"SCN_{scenario_count:04d}"
            name = str(rng.choice(SCENARIOS))
            burst = {
                "card_testing": _card_testing,
                "account_takeover": _account_takeover,
                "coordinated_spike": _coordinated_spike,
            }[name](rng, world, dt, sid)
            events.extend(burst)
            dt = burst[-1]["TransactionDT"]
        else:
            events.extend(_normal(rng, world, dt))

        # Exponential gaps: real arrivals are bursty, not evenly spaced.
        dt += max(1, int(rng.exponential(MEAN_GAP_SECONDS)))

    return _attach_templates(events[:rows], rng)


def _summarise(stream: pd.DataFrame, scored: pd.DataFrame | None) -> None:
    span_days = (
        stream["TransactionDT"].max() - stream["TransactionDT"].min()
    ) / 86_400
    print(f"\nStream of {len(stream):,} transactions")
    print(f"  customers active       {stream['customer_id'].nunique():,}")
    print(f"  merchants              {stream['merchant_id'].nunique():,}")
    print(f"  devices                {stream['device_id'].nunique():,}")
    print(
        "  transactions/customer  median "
        f"{stream['customer_id'].value_counts().median():.0f}"
    )
    print(f"  span                   {span_days:.1f} days")

    print("\n  scenario mix")
    for name, count in stream["synthetic_scenario"].value_counts().items():
        print(f"    {name:<20} {count:>6,}")
    incidents = stream.loc[
        stream["synthetic_scenario_id"] != "", "synthetic_scenario_id"
    ]
    print(f"    distinct incidents   {incidents.nunique():>6,}")

    if scored is None:
        return
    flagged = scored["recommendation"] != "ALLOW"
    injected = scored["synthetic_is_fraud"] == 1
    print(f"\n  flagged by the model   {int(flagged.sum()):,} of {len(scored):,}")
    print(
        f"  injected fraud flagged {int((injected & flagged).sum()):,} "
        f"of {int(injected.sum()):,}"
    )
    print(
        "\nThe model reacts to borrowed template columns, not to scenario shape. "
        "These streams exist for the behaviour and spike engines."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic transaction stream."
    )
    parser.add_argument("--rows", type=int, default=5_000)
    parser.add_argument("--mode", choices=("evaluation", "live"), default="evaluation")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario-rate", type=float, default=0.02)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    print(f"Generating {args.rows:,} transactions in {args.mode} mode...", flush=True)
    stream = generate_stream(args.rows, args.mode, args.seed, args.scenario_rate)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stream.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({stream.shape[0]:,} rows x {stream.shape[1]} cols)")

    _summarise(stream, score_batch(stream) if args.score else None)


if __name__ == "__main__":
    main()
