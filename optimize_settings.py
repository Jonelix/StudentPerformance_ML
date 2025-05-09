"""
optimize_settings.py  — reduced search space (bug‑fixed)

Tunes four key notebook settings with Optuna, averages F1 over the
six models defined in the student‑performance notebook, and writes the
best parameter set to *best_settings.json*.

Usage
-----
python optimize_settings.py \
       --data data/student_portugal/combined_noDup_processed.csv \
       --n_trials 80 \
       --n_runs 5
"""

import argparse
import json
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    LogisticRegression,
)

# ──────────────────────────────────────────────────────────
# Default notebook constants (only some will be tuned)
# ──────────────────────────────────────────────────────────
DEFAULT_SETTINGS: Dict[str, any] = {
    "TARGET_COLUMN": "G1",
    "TARGET_OPERATOR": "<",
    "TARGET_VALUE": 10,

    "RANDOM_SEED": None,
    "SPLIT_PERCENTAGE_TESTING": 0.12,

    "BALANCE_DATA": True,
    "BALANCE_QUERY_NON_TARGET_PERCENTAGE": 0.50,  # leave fixed

    "REGULARISER_WEIGHT": 0.01,
    "MODEL_THRESHOLD_LOG": 0.5,
}

# ──────────────────────────────────────────────────────────
# Hyper‑parameter bounds (reduced as requested)
# ──────────────────────────────────────────────────────────
BOUNDS = {
    "SPLIT_PERCENTAGE_TESTING": (0.05, 0.30),
    "BALANCE_DATA": (0, 1),  # Boolean flag
    "REGULARISER_WEIGHT": (1e-4, 2.0, "log"),  # log‑uniform
    "MODEL_THRESHOLD_LOG": (0.3, 0.7),
}

# ──────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────

def make_target(series: pd.Series, op: str, value: float) -> np.ndarray:
    """Convert a continuous mark into a 0/1 target."""
    if op == "<":
        return (series < value).astype(int).values
    elif op == ">":
        return (series > value).astype(int).values
    raise ValueError("TARGET_OPERATOR must be '<' or '>'")


def maybe_balance(
    X: pd.DataFrame,
    y: np.ndarray,
    non_target_pct: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Undersample / oversample negatives so P(y=0)=non_target_pct."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    desired_neg = int(len(pos_idx) * non_target_pct / (1 - non_target_pct))
    if desired_neg > len(neg_idx):
        sel_neg_idx = rng.choice(neg_idx, desired_neg, replace=True)
    else:
        sel_neg_idx = rng.choice(neg_idx, desired_neg, replace=False)

    keep = np.concatenate([pos_idx, sel_neg_idx])
    return X.iloc[keep].reset_index(drop=True), y[keep]


# ──────────────────────────────────────────────────────────
# Model training & evaluation
# ──────────────────────────────────────────────────────────

def train_models(
    X_train: pd.DataFrame,
    y_train_reg: np.ndarray,
    y_train_cls: np.ndarray,
    weight: float,
):
    """Fit the six models used in the original notebook."""
    models = {
        # Regression family
        "reg_none": LinearRegression().fit(X_train, y_train_reg),
        "reg_l1": Lasso(alpha=weight).fit(X_train, y_train_reg),
        "reg_l2": Ridge(alpha=weight).fit(X_train, y_train_reg),
        # Logistic family
        # NB: use solver='lbfgs' for penalty='none' (liblinear does NOT support it)
        "log_none": LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=400
        ).fit(X_train, y_train_cls),
        "log_l1": LogisticRegression(
            penalty="l1", C=1 / weight, solver="liblinear", max_iter=400
        ).fit(X_train, y_train_cls),
        "log_l2": LogisticRegression(
            penalty="l2", C=1 / weight, solver="liblinear", max_iter=400
        ).fit(X_train, y_train_cls),
    }
    return models


def evaluate_models(
    models,
    X_test,
    y_test_reg,
    y_test_cls,
    thresh_value: float,
    thresh_log: float,
):
    """Return mean (accuracy, f1) across the six models."""
    accs, f1s = [], []

    # Regression → threshold continuous prediction
    for name in ("reg_none", "reg_l1", "reg_l2"):
        y_pred_bin = (models[name].predict(X_test) < thresh_value).astype(int)
        accs.append(accuracy_score(y_test_cls, y_pred_bin))
        f1s.append(f1_score(y_test_cls, y_pred_bin))

    # Logistic → threshold probability
    for name in ("log_none", "log_l1", "log_l2"):
        prob = models[name].predict_proba(X_test)[:, 1]
        y_pred = (prob >= thresh_log).astype(int)
        accs.append(accuracy_score(y_test_cls, y_pred))
        f1s.append(f1_score(y_test_cls, y_pred))

    return float(np.mean(accs)), float(np.mean(f1s))


# ──────────────────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, data: pd.DataFrame, n_runs: int, base: Dict[str, any]):
    """One Optuna trial → average F1 across *n_runs* random splits."""
    settings = dict(base)

    # --- sample hyper‑parameters --------------------------------------------
    settings["SPLIT_PERCENTAGE_TESTING"] = trial.suggest_float(
        "SPLIT_PERCENTAGE_TESTING", *BOUNDS["SPLIT_PERCENTAGE_TESTING"]
    )
    settings["BALANCE_DATA"] = bool(
        trial.suggest_int("BALANCE_DATA", *BOUNDS["BALANCE_DATA"])
    )
    low, high, _ = BOUNDS["REGULARISER_WEIGHT"]
    settings["REGULARISER_WEIGHT"] = trial.suggest_float(
        "REGULARISER_WEIGHT", low, high, log=True
    )
    settings["MODEL_THRESHOLD_LOG"] = trial.suggest_float(
        "MODEL_THRESHOLD_LOG", *BOUNDS["MODEL_THRESHOLD_LOG"]
    )

    rng_global = np.random.default_rng(settings["RANDOM_SEED"])
    f1_runs = []

    for _ in range(n_runs):
        rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1))

        # Prepare data split
        X = data.drop(columns=[settings["TARGET_COLUMN"]])
        y_cont = data[settings["TARGET_COLUMN"]].values
        y_cls = make_target(
            data[settings["TARGET_COLUMN"]],
            settings["TARGET_OPERATOR"],
            settings["TARGET_VALUE"],
        )

        X_tr, X_te, y_reg_tr, y_reg_te, y_cls_tr, y_cls_te = train_test_split(
            X,
            y_cont,
            y_cls,
            test_size=settings["SPLIT_PERCENTAGE_TESTING"],
            random_state=rng.integers(0, 2**32 - 1),
            stratify=y_cls if settings["BALANCE_DATA"] else None,
        )

        if settings["BALANCE_DATA"]:
            X_tr, y_cls_tr = maybe_balance(
                X_tr,
                y_cls_tr,
                settings["BALANCE_QUERY_NON_TARGET_PERCENTAGE"],
                rng,
            )
            # keep regression labels aligned after re‑indexing
            y_reg_tr = data.loc[X_tr.index, settings["TARGET_COLUMN"]].values

        models = train_models(
            X_tr, y_reg_tr, y_cls_tr, settings["REGULARISER_WEIGHT"]
        )
        _, f1 = evaluate_models(
            models,
            X_te,
            y_reg_te,
            y_cls_te,
            settings["TARGET_VALUE"],
            settings["MODEL_THRESHOLD_LOG"],
        )
        f1_runs.append(f1)

    return float(np.mean(f1_runs))


# ──────────────────────────────────────────────────────────
# CLI entry‑point
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Bayesian optimiser for notebook settings")
    parser.add_argument("--data", required=True, help="Path to processed CSV")
    parser.add_argument("--n_trials", type=int, default=60, help="Optuna trials")
    parser.add_argument("--n_runs", type=int, default=5, help="Monte‑Carlo runs per trial")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")
    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.data)

    # Base constants + optional seed
    base = dict(DEFAULT_SETTINGS)
    base["RANDOM_SEED"] = args.seed

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(
        lambda t: objective(t, data, args.n_runs, base),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print("\nBest mean F1:", study.best_value)
    print(json.dumps(study.best_params, indent=2))

    with open("best_settings.json", "w") as fp:
        json.dump(study.best_params, fp, indent=2)
    print("Saved best settings to best_settings.json")


if __name__ == "__main__":
    main()
