"""Phase 3 - train the fraud model and report honestly on it.

Training and evaluation live together so the 1.8GB load happens once. The run
produces three artifacts:

    models/fraud_xgb.json     the booster
    models/feature_spec.json  the fitted feature spec - a model without it cannot be served
    docs/metrics.md           ML metrics and rupee impact

Three chronological sets, and the division of labour between them is the point
of this module:

    train       fits the booster
    validation  pays for early stopping and for choosing the thresholds
    test        never looked at until both are frozen, then measured once

A two-way split lets the same rows choose the stopping iteration, choose the
thresholds, and then report the score, which flatters the model by an amount
nobody can estimate afterwards. Splitting the job three ways costs 20% of the
rows and buys a number that means something.

    python -m ml.train
"""
import gc

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import ROOT, settings
from ml.features import fit, load_raw, read_column_decisions, split, transform

MODEL_DIR = ROOT / "models"
SPEC_PATH = MODEL_DIR / "feature_spec.json"

PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 4,
    "tree_method": "hist",  # keeps the 268-column matrix training in minutes
    "eval_metric": "aucpr",  # PR-AUC, not accuracy - the classes are 1:28
    "early_stopping_rounds": 100,
    "n_jobs": -1,
}

# Candidate thresholds for the cost sweep. 0.05 steps: finer resolution is
# false precision when the cost inputs are estimates to begin with.
GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)


def money(mask: np.ndarray, amounts: pd.Series) -> float:
    return float(amounts.to_numpy()[mask].sum())


def evaluate(
    y: np.ndarray, scores: np.ndarray, amounts: pd.Series, threshold: float
) -> dict:
    """Metrics at one advisory threshold, in both counts and rupees."""
    pred = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    is_fraud = y == 1
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        # The numbers a merchant actually feels.
        "legit_value_flagged": money(pred & ~is_fraud, amounts),
        "fraud_value_missed": money(~pred & is_fraud, amounts),
        "fraud_value_caught": money(pred & is_fraud, amounts),
    }


def cost(
    y: np.ndarray, scores: np.ndarray, amounts: pd.Series, review_t: float, block_t: float
) -> dict:
    """Net rupee cost of one advisory policy over a scored set.

    Three bands, three different things happening to the merchant:

        ALLOW   score < review          nobody looks. Fraud here is a full loss:
                                        the amount plus the chargeback admin fee.
        REVIEW  review <= score < block an analyst looks. That costs analyst time
                                        whether or not the transaction was fraud.
        BLOCK   score >= block          an analyst looks AND the merchant acts on
                                        the advisory, so a legitimate customer is
                                        turned away and some do not come back.

    Fraud that reaches REVIEW or BLOCK is counted as stopped. That is the
    optimistic half of the model and it is deliberate: it isolates the cost of
    *ranking*, which this project controls, from the cost of the manual process
    downstream, which it does not. Every rate below is an assumption living in
    `.env`, not a measurement.
    """
    amt = amounts.to_numpy()
    is_fraud = y == 1
    flagged = scores >= review_t
    blocked = scores >= block_t

    missed = is_fraud & ~flagged
    fraud_loss = float(amt[missed].sum()) + settings.chargeback_fee * int(missed.sum())
    review_effort = settings.review_cost * int(flagged.sum())
    friction = settings.block_friction_rate * float(amt[blocked & ~is_fraud].sum())

    return {
        "review_threshold": float(review_t),
        "block_threshold": float(block_t),
        "fraud_loss": fraud_loss,
        "review_effort": review_effort,
        "block_friction": friction,
        "total": fraud_loss + review_effort + friction,
        "reviewed": int(flagged.sum()),
        "blocked": int(blocked.sum()),
    }


def do_nothing_cost(y: np.ndarray, amounts: pd.Series) -> float:
    """What the window costs with no model at all: every fraud lands."""
    is_fraud = y == 1
    return float(amounts.to_numpy()[is_fraud].sum()) + settings.chargeback_fee * int(
        is_fraud.sum()
    )


def sweep(y: np.ndarray, scores: np.ndarray, amounts: pd.Series) -> list[dict]:
    """Cost of every (review, block) pair on the grid, cheapest first.

    Run on VALIDATION only. Picking thresholds on the set you then report from
    is the same mistake as tuning on test, one level down.
    """
    out = [cost(y, scores, amounts, r, b) for r in GRID for b in GRID if b >= r]
    return sorted(out, key=lambda c: c["total"])


def report(
    *,
    model,
    spec,
    test_y,
    test_scores,
    test_amounts,
    val_pr_auc,
    best,
    sizes,
    boundaries,
    best_iter,
) -> str:
    """Keyword-only: ten positional args of similar type are one transposition
    away from a silently wrong report."""
    n_train, n_val, n_test = sizes
    val_dt, test_dt = boundaries

    pr_auc = average_precision_score(test_y, test_scores)
    roc_auc = roc_auc_score(test_y, test_scores)
    base_rate = test_y.mean()
    total_fraud_value = money(test_y == 1, test_amounts)

    policy_review = settings.risk_review_threshold
    policy_block = settings.risk_block_threshold
    review = evaluate(test_y, test_scores, test_amounts, policy_review)
    block = evaluate(test_y, test_scores, test_amounts, policy_block)

    baseline = do_nothing_cost(test_y, test_amounts)
    policy_cost = cost(test_y, test_scores, test_amounts, policy_review, policy_block)
    best_cost = cost(
        test_y, test_scores, test_amounts, best["review_threshold"], best["block_threshold"]
    )

    importance = model.get_booster().get_score(importance_type="gain")
    top = sorted(importance.items(), key=lambda kv: -kv[1])[:20]

    def band(m: dict) -> list[str]:
        return [
            f"| Precision | {m['precision']:.4f} |",
            f"| Recall | {m['recall']:.4f} |",
            f"| F1 | {m['f1']:.4f} |",
            f"| True positives | {m['tp']:,} |",
            f"| False positives | {m['fp']:,} |",
            f"| False negatives | {m['fn']:,} |",
            f"| True negatives | {m['tn']:,} |",
            f"| False-positive rate | {m['false_positive_rate'] * 100:.2f}% |",
            f"| False-negative rate | {m['false_negative_rate'] * 100:.2f}% |",
            f"| Legit value flagged | {m['legit_value_flagged']:,.0f} |",
            f"| Fraud value caught | {m['fraud_value_caught']:,.0f} |",
            f"| Fraud value missed | {m['fraud_value_missed']:,.0f} |",
        ]

    def cost_row(label: str, c: dict) -> str:
        saved = baseline - c["total"]
        return (
            f"| {label} | {c['review_threshold']:.2f} | {c['block_threshold']:.2f} | "
            f"{c['fraud_loss']:,.0f} | {c['review_effort']:,.0f} | "
            f"{c['block_friction']:,.0f} | **{c['total']:,.0f}** | "
            f"{saved:,.0f} ({saved / baseline * 100:.1f}%) |"
        )

    lines = [
        "# Model Evaluation",
        "",
        f"Generated by `python -m ml.train`. Model version `{settings.model_version}`, "
        f"seed {settings.random_seed}.",
        "",
        "## Which set each number comes from",
        "",
        "This report separates the set that *chose* things from the set that *measures* "
        "them, because mixing the two is the most common way a fraud model ends up "
        "looking better on a slide than in production.",
        "",
        "| Set | Rows | Used for |",
        "|---|---|---|",
        f"| Train | {n_train:,} | fitting the booster and the feature spec |",
        f"| Validation | {n_val:,} | early stopping, threshold selection |",
        f"| Test | {n_test:,} | **every number below, measured once** |",
        "",
        "The split is chronological, at "
        f"`TransactionDT = {val_dt:,.0f}` and `{test_dt:,.0f}`. A random split lets card "
        "and address history bleed across the boundary and inflates everything.",
        "",
        f"Validation PR-AUC was **{val_pr_auc:.4f}** and test PR-AUC is **{pr_auc:.4f}**. "
        "The validation figure appears here once, for comparison only. It is the number "
        "early stopping optimised against, so it is not a performance claim.",
        "",
        "Expect this to sit below what a two-way split reports for the same model "
        "family, for two reasons worth keeping separate. One is bias this design "
        "removes: with two sets, early stopping and threshold selection both run "
        "against the rows that are then reported, and the reported figure inherits "
        "that. The other is a real cost: holding back a third set leaves "
        f"{n_train:,} training rows instead of {n_train + n_val:,}, and moves the end "
        "of the training window further from the test window, so the model is both "
        "smaller and staler. The first is bias worth deleting. The second is the "
        "price of deleting it.",
        "",
        "## Ranking quality (test)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **PR-AUC** | **{pr_auc:.4f}** |",
        f"| ROC-AUC | {roc_auc:.4f} |",
        f"| Baseline PR-AUC (random) | {base_rate:.4f} |",
        f"| Lift over baseline | {pr_auc / base_rate:.1f}x |",
        f"| Best iteration | {best_iter} of {PARAMS['n_estimators']} |",
        f"| Features | {len(spec.columns)} |",
        "",
        "PR-AUC leads because the test set is "
        f"{base_rate * 100:.2f}% fraud: predicting 'never fraud' scores "
        f"{(1 - base_rate) * 100:.1f}% accuracy while catching nothing, so accuracy is "
        "not reported.",
        "",
        "## Cost model",
        "",
        "A threshold is a business decision, and precision and recall are the wrong "
        "units for arguing about one. These three inputs convert the confusion matrix "
        "into rupees. They are **assumptions, set in `.env`, not measurements**:",
        "",
        "| Input | Value | Meaning |",
        "|---|---|---|",
        f"| `CHARGEBACK_FEE` | {settings.chargeback_fee:,.0f} | admin cost of a chargeback, "
        "on top of the lost amount |",
        f"| `REVIEW_COST` | {settings.review_cost:,.0f} | analyst cost to clear one "
        "flagged transaction |",
        f"| `BLOCK_FRICTION_RATE` | {settings.block_friction_rate:.2f} | share of a wrongly "
        "BLOCKed legitimate amount that walks away |",
        "",
        "Cost of a policy = missed fraud (amount + fee) + analyst time on everything "
        "flagged + friction on legitimate transactions that reached BLOCK. Fraud that "
        "reaches REVIEW or BLOCK counts as stopped, which isolates the cost of ranking "
        "from the cost of the manual process this project does not own.",
        "",
        f"Doing nothing costs **{baseline:,.0f}** over the test window: "
        f"{int((test_y == 1).sum()):,} frauds worth {total_fraud_value:,.0f} plus fees.",
        "",
        "| Policy | Review | Block | Missed fraud | Analyst time | Block friction | "
        "Net cost | Saved vs doing nothing |",
        "|---|---|---|---|---|---|---|---|",
        cost_row("Configured (`.env`)", policy_cost),
        cost_row("Cost-optimal on validation", best_cost),
        "",
        "The cost-optimal row was chosen by sweeping "
        f"{len(GRID)}x{len(GRID)} threshold pairs **on validation**, then scored once on "
        "test. Choosing thresholds on the set you report from is the same mistake as "
        "tuning on test, one level down.",
        "",
    ]

    gap = policy_cost["total"] - best_cost["total"]
    if abs(gap) < 1:
        lines += ["The configured thresholds are already cost-optimal.", ""]
    elif gap > 0:
        lines += [
            f"Moving `RISK_REVIEW_THRESHOLD` to {best_cost['review_threshold']:.2f} and "
            f"`RISK_BLOCK_THRESHOLD` to {best_cost['block_threshold']:.2f} would save a "
            f"further **{gap:,.0f}** over this window, at the cost of "
            f"{best_cost['reviewed'] - policy_cost['reviewed']:+,} transactions in the "
            "review queue. Thresholds are policy and live in `.env`; nothing needs "
            "retraining to change them.",
            "",
        ]
    else:
        lines += [
            "The configured thresholds beat the validation-optimal pair on test by "
            f"{-gap:,.0f}. That gap is selection noise, not a reason to move them.",
            "",
        ]

    lines += [
        f"## At the configured REVIEW threshold ({policy_review}), on test",
        "",
        "| Metric | Value |",
        "|---|---|",
        *band(review),
        "",
        f"## At the configured BLOCK threshold ({policy_block}), on test",
        "",
        "| Metric | Value |",
        "|---|---|",
        *band(block),
        "",
        "## Financial exposure (test)",
        "",
        f"- Total fraud value in the test window: **{total_fraud_value:,.0f}**",
        f"- Caught at REVIEW: **{review['fraud_value_caught']:,.0f}** "
        f"({review['fraud_value_caught'] / total_fraud_value * 100:.1f}%)",
        f"- Missed at REVIEW: **{review['fraud_value_missed']:,.0f}**",
        f"- Legitimate value sent for review: **{review['legit_value_flagged']:,.0f}** "
        f"across {review['fp']:,} transactions",
        "",
        "Every false positive is a real customer inconvenienced, so the legitimate value "
        "flagged is a cost, not a rounding error. That is exactly what the cost model "
        "above prices.",
        "",
        "## Top 20 features by gain",
        "",
        "| Feature | Gain |",
        "|---|---|",
        *[f"| {name} | {gain:,.0f} |" for name, gain in top],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    try:
        decisions = read_column_decisions()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if "test_dt" not in decisions:
        raise SystemExit(
            "feature_columns.json predates the three-way split. "
            "Re-run `python -m ml.audit` first."
        )
    val_dt = float(decisions["split_dt"])
    test_dt = float(decisions["test_dt"])

    print("Loading IEEE-CIS...", flush=True)
    try:
        df = load_raw()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    # Two applications of the same two-way split. `split` stays the only place
    # that knows a boundary is chronological.
    train_df, rest = split(df, val_dt)
    val_df, test_df = split(rest, test_dt)
    sizes = (len(train_df), len(val_df), len(test_df))
    print("Train {:,} | Validation {:,} | Test {:,}".format(*sizes), flush=True)

    # Fitted on training rows only. Fitting on everything would leak future card
    # frequencies backwards into the training features.
    print("Fitting feature spec on training rows...", flush=True)
    spec = fit(train_df)
    X_train = transform(train_df, spec)
    X_val, X_test = transform(val_df, spec), transform(test_df, spec)
    y_train = train_df["isFraud"].to_numpy()
    y_val, y_test = val_df["isFraud"].to_numpy(), test_df["isFraud"].to_numpy()
    val_amounts = val_df["TransactionAmt"].copy()
    test_amounts = test_df["TransactionAmt"].copy()
    print(f"Feature matrix: {X_train.shape[1]} columns", flush=True)

    # The raw frame is 1.8GB and nothing below needs it. Holding it through
    # training pushes a 16GB machine into swap.
    del df, rest, train_df, val_df, test_df
    gc.collect()

    # 1:28 imbalance - without this the model optimises for the majority class.
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        **PARAMS, scale_pos_weight=pos_weight, random_state=settings.random_seed
    )

    print("Training...", flush=True)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    val_scores = model.predict_proba(X_val)[:, 1]
    val_pr_auc = average_precision_score(y_val, val_scores)

    print("Sweeping thresholds on validation...", flush=True)
    best = sweep(y_val, val_scores, val_amounts)[0]
    print(
        f"Cost-optimal on validation: review {best['review_threshold']:.2f}, "
        f"block {best['block_threshold']:.2f}",
        flush=True,
    )

    # Test is touched here for the first time, with the model and the thresholds
    # already frozen.
    test_scores = model.predict_proba(X_test)[:, 1]

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / settings.model_path.name
    model.save_model(model_path)
    spec.save(SPEC_PATH)

    metrics_path = ROOT / "docs" / "metrics.md"
    metrics_path.parent.mkdir(exist_ok=True)
    metrics_path.write_text(
        report(
            model=model,
            spec=spec,
            test_y=y_test,
            test_scores=test_scores,
            test_amounts=test_amounts,
            val_pr_auc=val_pr_auc,
            best=best,
            sizes=sizes,
            boundaries=(val_dt, test_dt),
            best_iter=model.best_iteration,
        ),
        encoding="utf-8",
    )

    print(
        f"\nTest PR-AUC {average_precision_score(y_test, test_scores):.4f} | "
        f"ROC-AUC {roc_auc_score(y_test, test_scores):.4f} "
        f"(validation PR-AUC was {val_pr_auc:.4f})"
    )
    for path in (model_path, SPEC_PATH, metrics_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
