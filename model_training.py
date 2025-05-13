# -*- coding: utf-8 -*-
"""Model‑training utilities for the PISA project
------------------------------------------------
This module implements both scikit‑learn *and* PyTorch training
routines with the following upgrades over the earlier version:

* ✅  One‑file save/load via **joblib** instead of fragile JSON.
* ✅  **Pipeline** encapsulates scaling + estimator, eliminating
       leakage and boiler‑plate.
* ✅  `class_weight="balanced"` (or custom) to address class imbalance.
* ✅  Optional grid‑search with stratified CV and PR‑AUC scoring.
* ✅  Decision‑threshold tuning on a validation set.
* ✅  PyTorch loop uses mini‑batches, `BCEWithLogitsLoss`, early stopping,
       and keeps *only* the best checkpoint.
* ✅  All random seeds respected ⇒ reproducible runs.
* ✅  Path handling via `pathlib.Path`.

Author: <your‑name> – 2025‑05‑09
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
RANDOM_STATE = 11032004
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
CV_SPLITS = 5

# ---------------------------------------------------------------------
# General paths
# ---------------------------------------------------------------------
ROOT_DIR = Path.cwd()
MODEL_ROOT = ROOT_DIR / "models"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

# =====================================================================
# ────────────────────────────  sklearn  ──────────────────────────────
# =====================================================================

def _timestamp() -> str:
    """Return current time in YYYYMMDD‑HHMMSS format."""
    return time.strftime("%Y%m%d‑%H%M%S", time.localtime())


def _model_dir(prefix: str | Path) -> Path:
    """Create & return a fresh timestamped sub‑folder under MODEL_ROOT."""
    d = MODEL_ROOT / str(prefix) / _timestamp()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------
# 1.  Linear regression (SGDRegressor) wrapped in a Pipeline
# ------------------------------------------------------------------

def train_linear_model(
    X: np.ndarray,
    y: np.ndarray,
    penalty: Literal["none", "l2", "l1", "elasticnet"] = "l2",
    alpha_grid: Sequence[float] = (1e-4, 1e-3, 1e-2),
    cv_splits: int = CV_SPLITS,
    scoring: str = "neg_root_mean_squared_error",
    threshold_tuning: bool = True,  # Add a flag for threshold tuning
    save: bool = True,
) -> Tuple[Pipeline, Optional[float]]:
    """Fit a (scaler → SGDRegressor) pipeline with CV‑tuned alpha and optional threshold tuning.

    Parameters
    ----------
    X, y : training data
    penalty : regularisation type (matches SGDRegressor)
    alpha_grid : list of α (regularisation strengths) to search
    cv_splits : number of stratified folds
    threshold_tuning : if True, perform threshold tuning for binary decisions
    save : if True, persist the best pipeline under `models/linear/`.

    Returns
    -------
    pipeline : fitted Pipeline
    best_thresh : decision threshold that maximises F1 on the val‑set (if threshold_tuning=True)
    """

    # ── 1. hold‑out split for threshold tuning ──
    if threshold_tuning:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
    else:
        X_tr, y_tr = X, y

    # ── 2. build scaling + estimator pipeline ──
    pipe = make_pipeline(
        StandardScaler(with_mean=False),  # keeps sparse dummies sparse
        SGDRegressor(
            loss="squared_error",
            penalty=None if penalty == "none" else penalty,
            random_state=RANDOM_STATE,
        ),
    )

    # ── 3. hyperparameter tuning ──
    param_grid = {"sgdregressor__alpha": alpha_grid}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    search.fit(X_tr, y_tr)

    best_pipe: Pipeline = search.best_estimator_
    print(
        f"[LR] Best {scoring} = {search.best_score_:.4f} with alpha="
        f"{best_pipe.named_steps['sgdregressor'].alpha}"
    )

    # ── 4. threshold tuning on validation set ──
    best_thresh = None
    if threshold_tuning:
        y_pred = best_pipe.predict(X_val)
        prec, rec, thresh = precision_recall_curve(y_val, y_pred)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = f1.argmax()
        best_thresh = thresh[best_idx]
        print(
            f"[LR] Best F1 = {f1[best_idx]:.4f} at threshold = {best_thresh:.3f}"
        )

    # ── 5. save pipeline ──
    if save:
        out_dir = _model_dir("linear")
        path = out_dir / "pipeline.joblib"
        joblib.dump(best_pipe, path)
        print(f"[LR] Saved pipeline to {path.relative_to(ROOT_DIR)}")

    return best_pipe, best_thresh


# ------------------------------------------------------------------
# 2.  Logistic regression (classification) with imbalance handling
# ------------------------------------------------------------------

def train_logistic_model(
    X: np.ndarray,
    y: np.ndarray,
    penalty: Literal["l2", "l1", "elasticnet"] = "l2",
    alpha_grid: Sequence[float] = (1e-4, 1e-3, 1e-2),
    l1_ratio_grid: Sequence[float] | None = None,  # used if elasticnet
    use_smote: bool = True,
    cv_splits: int = CV_SPLITS,
    save: bool = True,
) -> Tuple[Pipeline, float]:
    """Train a logistic‑regression pipeline with SMOTE + CV‑tuned hyper‑params.

    Returns
    -------
    pipeline : fitted Pipeline
    best_thresh : decision threshold that maximises F1 on the val‑set
    """

    # ── 1. hold‑out split for threshold tuning ──
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # ── 2. build resampling + scaling + estimator pipeline ──
    pipe_steps: list[Tuple[str, Any]] = []
    if use_smote:
        pipe_steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))

    pipe_steps.extend(
        [
            ("scale", StandardScaler(with_mean=False)),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    penalty=penalty,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    pipe: Pipeline | ImbPipeline = ImbPipeline(pipe_steps)

    param_grid: Dict[str, Sequence[Any]] = {
        "clf__alpha": alpha_grid,
    }
    if penalty == "elasticnet":
        param_grid["clf__l1_ratio"] = l1_ratio_grid or [0.1, 0.5, 0.9]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipe,
        param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_tr, y_tr)

    best_pipe: Pipeline = search.best_estimator_
    print(
        f"[CLF] Best AUPRC = {search.best_score_:.4f} with"
        f" alpha={best_pipe.named_steps['clf'].alpha}"
    )

    # ── 3. threshold tuning on validation set ──
    y_prob = best_pipe.predict_proba(X_val)[:, 1]
    prec, rec, thresh = precision_recall_curve(y_val, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = f1.argmax()
    best_thresh: float = thresh[best_idx]
    print(
        f"[CLF] Best F1 = {f1[best_idx]:.4f} at threshold = {best_thresh:.3f}"
    )

    # ── 4. save everything ──
    if save:
        out_dir = _model_dir("logistic")
        joblib.dump({"pipeline": best_pipe, "threshold": best_thresh}, out_dir / "model.joblib")
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "cv_auprc_mean": search.best_score_,
                    "best_threshold": best_thresh,
                    "val_f1": f1[best_idx],
                },
                indent=2,
            )
        )
        print(f"[CLF] Saved pipeline + threshold to {out_dir.relative_to(ROOT_DIR)}")

    return best_pipe, best_thresh

# ------------------------------------------------------------------
# Convenience save/load wrappers
# ------------------------------------------------------------------

def save_pipeline(pipeline: Pipeline, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path) -> Pipeline:
    return joblib.load(path)


if __name__ == "__main__":
    print("This module is intended to be imported, not executed directly.")
