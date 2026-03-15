from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model_training import (
    train_linear_model,
    train_logistic_model,
    load_pipeline,               # convenience wrapper
)

# ------------------------------------------------------------------
# 1.  Load & split your data  (X : 2‑D ndarray,  y : 1‑D ndarray)
# ------------------------------------------------------------------

PISA_MODEL = True
REGULARIZER = "l2"  # Can be "none", "l1", "l2"
ALPHA_GRID = tuple(10.0 ** np.arange(-6, 3))  # for SGDClassifier / SGDRegressor regularization strength
LIN_THRESHOLD_TUNING = True  
CV_SPLITS = 5  # for both models
TEST_SIZE=0.3
RANDOM_STATE = 110304
#-------------SMOTE--------------------
LOG_REG_USE_SMOTE = False  # for logistic regression only
LOG_REG_USE_SMOTE_TOMEK = False  # for logistic regression only
LOG_REG_USE_SMOTE_ENN = False #for logistic regression only
#---------------------------------


if PISA_MODEL:
    FAIL = "GENERAL_SCORE"
    DATA_PATH = "data/pisa_2022/pisa_clean.csv"
    PISA_PERCENTAGE = 30
    GRADE_THRESH = 400
else:
    FAIL = "G3"
    DATA_PATH = "data/student_portugal/portugal_clean.csv"
    GRADE_THRESH = 10

df = pd.read_csv(DATA_PATH)

if PISA_MODEL:
    GRADE_THRESH = df['GENERAL_SCORE'].quantile(PISA_PERCENTAGE / 100)

df["FAIL"] = (df[FAIL] >= GRADE_THRESH).astype(int)
df["FAIL"] = 1 - df["FAIL"]
target = "FAIL"  # target column name

# 2. define a list of columns that leak target information
LEAK_COLS = [FAIL]


# 3. build the feature matrix
X_df = df.drop(columns=["FAIL"] + [c for c in LEAK_COLS if c in df.columns])
feature_names = X_df.columns.to_numpy()
X        = X_df.values             # the array the model sees

y = df[target].values                  # numeric 0 / 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ------------------------------------------------------------------
# 2A.  Train logistic‑regression (sklearn)  – handles imbalance + SMOTE
# ------------------------------------------------------------------
pipe, best_thresh = train_logistic_model(
    X_train,
    y_train,
    penalty=REGULARIZER,                
    alpha_grid=ALPHA_GRID,
    use_smote=LOG_REG_USE_SMOTE,              # default
    use_smote_tomek=LOG_REG_USE_SMOTE_TOMEK,  # default
    use_smote_enn=LOG_REG_USE_SMOTE_ENN,      # default
    cv_splits=CV_SPLITS,               # default
)

# ------------------------------------------------------------------
# 2B.  Train a linear regressor (RMSE tuned)
# ------------------------------------------------------------------
lin_pipe, lin_best_thresh = train_linear_model(
    X_train, 
    y_train, 
    penalty=REGULARIZER, 
    alpha_grid=ALPHA_GRID,
    threshold_tuning=LIN_THRESHOLD_TUNING,  # default
    cv_splits=CV_SPLITS,               # default
)

# ──────────────────────────────────────────────────────────────────
# Visualise the results
# ──────────────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

def plot_cm(y_true, y_pred, title, ax=None, font_size=16):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Pass (0)", "Fail (1)"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)

    if ax is not None:
        ax.set_title(title)

    # Fix: iterate through each text object in the ndarray
    for text_obj in disp.text_.ravel():  # ravel flattens the array
        text_obj.set_fontsize(font_size)

    return cm



# Keep track so we can show all three figures side‑by‑side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Logistic regression confusion matrix
prob_test = pipe.predict_proba(X_test)[:, 1]
y_pred = (prob_test > best_thresh).astype(int)
cm_log = plot_cm(y_test, y_pred, "Logistic pipeline", ax=axes[0])
print("\n[Logistic] report\n", classification_report(y_test, y_pred, digits=3))

# Linear regression confusion matrix
prob_lin = lin_pipe.predict(X_test)           # raw continuous output
prob_lin = np.clip(prob_lin, 0, 1)            # ensure 0‑1 range
if lin_best_thresh is not None:
    y_pred_lin = (prob_lin > lin_best_thresh).astype(int)
else:
    y_pred_lin = (prob_lin > 0.5).astype(int)

cm_lin = plot_cm(y_test, y_pred_lin, "Linear pipeline", ax=axes[1])
print("\n[Linear] report\n", classification_report(y_test, y_pred_lin, digits=3))

plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────
# Helpers for “best model per penalty” reports
# ──────────────────────────────────────────────────────────────────
from pprint import pprint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
                            accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

def evaluate_best(model_type: str,
                  df_metrics: pd.DataFrame,
                  model_store: dict,
                  penalty: str,
                  font_size: int = 16,
                  tick_label_size: int = 14):
    """
    Pulls the best (highest-F1) run for a given penalty, rebuilds the identical
    train/test split, and prints a performance summary + confusion matrix.
    Enlarges text in confusion matrix for better readability.
    """
    # ① locate the best run
    best_row = df_metrics[df_metrics["penalty"] == penalty]       \
                         .loc[lambda d: d["f1"].idxmax()]
    run_id   = int(best_row["run"])
    pipe, thr = model_store[(run_id, penalty)]

    # ② reproduce that split exactly
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=run_id
    )

    if model_type == "Logistic":
        y_hat = (pipe.predict_proba(X_te)[:, 1] > thr).astype(int)
    else:  # Linear
        y_prob = np.clip(pipe.predict(X_te), 0, 1)
        thr_eval = 0.5 if thr is None else thr
        y_hat = (y_prob > thr_eval).astype(int)

    # ③ metrics + confusion-matrix
    acc  = accuracy_score(y_te, y_hat)
    f1   = f1_score(y_te, y_hat, zero_division=0)
    prec = precision_score(y_te, y_hat, zero_division=0)
    rec  = recall_score(y_te, y_hat, zero_division=0)
    cm   = confusion_matrix(y_te, y_hat, labels=[0, 1])

    print(f"\n===  {model_type}  |  penalty = {penalty.upper()}  |  best run #{run_id}  ===")
    print(f"Threshold: {thr:.3f}")
    print(f"Accuracy : {acc:.3f}\nF1-score: {f1:.3f}\nPrecision: {prec:.3f}\nRecall   : {rec:.3f}")

    print("\nHyper-parameters:")
    pprint(pipe.named_steps["clf"].get_params())

    disp = ConfusionMatrixDisplay(cm, display_labels=["Pass (0)", "Fail (1)"])
    disp.plot(values_format="d", cmap="Blues")
    plt.title(f"{model_type} | {penalty.upper()} | run {run_id}")

    # 🔠 Enlarge cell numbers
    for text_obj in disp.text_.ravel():
        text_obj.set_fontsize(font_size)

    # 🔠 Enlarge axis tick labels
    disp.ax_.tick_params(labelsize=tick_label_size)


# ──────────────────────────────────────────────────────────────────
# Trial 1: How does model performance change with different regularizers? (None, L1, L2) | n_runs = 1000
# ──────────────────────────────────────────────────────────────────
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

N_RUNS     = 100         # ↳ heavy – trim if your machine struggles
PENALTIES  = ("none", "l1", "l2")

log_records, lin_records = defaultdict(list), defaultdict(list)
trained_log_models, trained_lin_models = {}, {}
rng = np.random.RandomState(RANDOM_STATE)   # reproducible but different split each run

for run in range(N_RUNS):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=run
    )

    for pen in PENALTIES:
        #  LOGISTIC
        log_pipe, log_thr = train_logistic_model(
            X_tr, y_tr,
            penalty=pen,
            alpha_grid=ALPHA_GRID,
            use_smote=LOG_REG_USE_SMOTE,
            use_smote_tomek=LOG_REG_USE_SMOTE_TOMEK,
            use_smote_enn=LOG_REG_USE_SMOTE_ENN,
            cv_splits=CV_SPLITS,
            save=False,
        )
        trained_log_models[(run, pen)] = (log_pipe, log_thr)

        y_hat = (log_pipe.predict_proba(X_te)[:, 1] > log_thr).astype(int)

        nnz = np.count_nonzero(np.abs(log_pipe.named_steps["clf"].coef_))

        for k, v in dict(
            run=run, penalty=pen,
            accuracy=accuracy_score(y_te, y_hat),
            f1=f1_score(y_te, y_hat, zero_division=0),
            precision=precision_score(y_te, y_hat, zero_division=0),
            recall=recall_score(y_te, y_hat, zero_division=0),
            threshold=round(log_thr, 3),
            nnz=nnz
        ).items():
            log_records[k].append(v)

        #  LINEAR
        lin_pipe, lin_thr = train_linear_model(
            X_tr, y_tr,
            penalty=pen,
            alpha_grid=ALPHA_GRID,
            threshold_tuning=LIN_THRESHOLD_TUNING,
            cv_splits=CV_SPLITS,
            save=False,
        )
        trained_lin_models[(run, pen)] = (lin_pipe, lin_thr)
        y_prob = np.clip(lin_pipe.predict(X_te), 0, 1)
        thr    = 0.5 if lin_thr is None else lin_thr
        y_hat  = (y_prob > thr).astype(int)

        for k, v in dict(
            run=run, penalty=pen,
            accuracy=accuracy_score(y_te, y_hat),
            f1=f1_score(y_te, y_hat, zero_division=0),
            precision=precision_score(y_te, y_hat, zero_division=0),
            recall=recall_score(y_te, y_hat, zero_division=0),
            threshold=round(thr, 3)
        ).items():
            lin_records[k].append(v)

joblib.dump(trained_log_models, "models/cached_logistic_models.joblib")

# Collate & display
df_log = pd.DataFrame(log_records)
df_lin = pd.DataFrame(lin_records)
df_log.to_csv("./models/df/df_log.csv", index=False)
df_lin.to_csv("./models/df/df_lin.csv", index=False)


print("\n=== Logistic regression (mean of {:,} runs) ===".format(N_RUNS))
print(df_log.groupby("penalty")[["accuracy", "f1", "precision", "recall"]].mean().round(3))

print("\n=== Linear regression (mean of {:,} runs) ===".format(N_RUNS))
print(df_lin.groupby("penalty")[["accuracy", "f1", "precision", "recall"]].mean().round(3))

for pen in PENALTIES:
    evaluate_best("Logistic", df_log, trained_log_models, pen)
    evaluate_best("Linear",   df_lin, trained_lin_models, pen)

plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────
# Trial 2 – aggregate *coefficient strength* instead of frequency
#               (now for BOTH logistic and linear pipelines)
# ──────────────────────────────────────────────────────────────────
from collections import defaultdict
import pandas as pd
import joblib

TOP_N_RUNS     = 10        # best runs per penalty
COEF_MIN       = 1e-10      # importance threshold
TOP_M_FEATURES = 20        # how many lines to print per penalty
LOG_PATH       = "models/df/df_log.csv"
LIN_PATH       = "models/df/df_lin.csv"

# ------------------------------------------------------------------
# Helper that accumulates |coef| and signed coef over top-runs
# ------------------------------------------------------------------
def aggregate_strength(df_metrics, model_store, label: str):
    """
    Returns two nested dicts (strength_sum, signed_sum) keyed by
    penalty → feature → value.
    """
    strength_sum = defaultdict(lambda: defaultdict(float))
    signed_sum   = defaultdict(lambda: defaultdict(float))

    for pen, grp in df_metrics.groupby("penalty"):
        top_runs = grp.nlargest(TOP_N_RUNS, "f1")          # best F1 per penalty

        for run_id in top_runs["run"]:
            pipe, _ = model_store[(run_id, pen)]
            coefs   = pipe.named_steps["clf"].coef_.ravel()

            for feat, coef in zip(feature_names, coefs):
                if abs(coef) >= COEF_MIN:
                    strength_sum[pen][feat] += abs(coef)
                    signed_sum  [pen][feat] += coef      # preserve sign

    return strength_sum, signed_sum


# ── 2·A  LOGISTIC  ────────────────────────────────────────────────
df_log = pd.read_csv(LOG_PATH)

try:
    trained_log_models
except NameError:     # if you started from a fresh session
    trained_log_models = joblib.load("models/cached_logistic_models.joblib")

log_strength, log_signed = aggregate_strength(df_log, trained_log_models, "Logistic")

for pen in log_strength:
    rows = [
        (feat, mag, "+" if log_signed[pen][feat] > 0
                else ("–" if log_signed[pen][feat] < 0 else "0"))
        for feat, mag in log_strength[pen].items()
    ]
    rows = sorted(rows, key=lambda t: t[1], reverse=True)[:TOP_M_FEATURES]

    print(f"\nLogistic Regression – {pen.upper()}  "
          f"(top {TOP_N_RUNS} runs, coef ≥ {COEF_MIN})")
    print("-" * 72)
    print(f"{'Feature':30s} {'Σ|coef|':>12s}  {'Sign'}")
    for feat, mag, sign in rows:
        print(f"{feat:<30s} {mag:12.4f}   {sign}")


# ── 2·B  LINEAR   ────────────────────────────────────────────────
df_lin = pd.read_csv(LIN_PATH)

# save the cache once per session so we can reload later if needed
if "trained_lin_models" in globals():
    joblib.dump(trained_lin_models, "models/cached_linear_models.joblib")

try:
    trained_lin_models
except NameError:
    trained_lin_models = joblib.load("models/cached_linear_models.joblib")

lin_strength, lin_signed = aggregate_strength(df_lin, trained_lin_models, "Linear")

for pen in lin_strength:
    rows = [
        (feat, mag, "+" if lin_signed[pen][feat] > 0
                else ("–" if lin_signed[pen][feat] < 0 else "0"))
        for feat, mag in lin_strength[pen].items()
    ]
    rows = sorted(rows, key=lambda t: t[1], reverse=True)[:TOP_M_FEATURES]

    print(f"\nLinear Regression – {pen.upper()}  "
          f"(top {TOP_N_RUNS} runs, coef ≥ {COEF_MIN})")
    print("-" * 72)
    print(f"{'Feature':30s} {'Σ|coef|':>12s}  {'Sign'}")
    for feat, mag, sign in rows:
        print(f"{feat:<30s} {mag:12.4f}   {sign}")


# ── 2·C  show the best single model for each penalty (both types) ─
BEST_METRIC      = "f1"
MAX_FEATS_SHOWN  = 25

def report_best(df_metrics, model_store, model_type: str):
    for pen, grp in df_metrics.groupby("penalty"):
        best_row   = grp.loc[grp[BEST_METRIC].idxmax()]
        best_run   = int(best_row["run"])
        best_score = best_row[BEST_METRIC]

        pipe, best_thr = model_store[(best_run, pen)]
        coefs = pipe.named_steps["clf"].coef_.ravel()
        feat_coef = sorted(zip(feature_names, coefs),
                           key=lambda t: abs(t[1]), reverse=True)

        print(f"\n=====  BEST {model_type} {pen.upper()} MODEL  =====")
        print(f"run {best_run} | {BEST_METRIC} = {best_score:.3f} | "
              f"threshold = {None if best_thr is None else best_thr:.3f}")
        print("-" * 50)
        print(f"{'Feature':30s} {'Coef':>10}")
        for feat, c in feat_coef[:MAX_FEATS_SHOWN]:
            print(f"{feat:<30s} {c:10.4f}")

# Logistic best-model tables
report_best(df_log, trained_log_models, "Logistic")

# Linear best-model tables
report_best(df_lin, trained_lin_models, "Linear")

exit(0)

# ──────────────────────────────────────────────────────────────────
# Trial 3: Can we find better models by adjusting the PISA_PERCENTAGE? (10‑50) | n_runs = 1000
# ──────────────────────────────────────────────────────────────────
if PISA_MODEL:
    from sklearn.metrics import f1_score, accuracy_score

    N_RUNS        = 100
    PERC_GRID     = range(10, 51)       # 10 … 50 inclusive
    pisa_records  = []

    for perc in PERC_GRID:
        tmp_df   = df.copy()
        thresh   = tmp_df["GENERAL_SCORE"].quantile(perc / 100)
        tmp_df["FAIL"] = (tmp_df["GENERAL_SCORE"] >= thresh).astype(int)

        X_tmp = tmp_df.drop(columns=["FAIL"] + [c for c in LEAK_COLS if c in tmp_df.columns]).values
        y_tmp = tmp_df["FAIL"].values

        for run in range(N_RUNS):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_tmp, y_tmp, test_size=0.30, stratify=y_tmp, random_state=run
            )
            pipe, thr = train_logistic_model(
                X_tr, y_tr,
                penalty=REGULARIZER,
                alpha_grid=ALPHA_GRID,
                use_smote=LOG_REG_USE_SMOTE,
                cv_splits=CV_SPLITS,
                save=False,
            )
            y_hat = (pipe.predict_proba(X_te)[:, 1] > thr).astype(int)
            pisa_records.append(
                dict(percentile=perc,
                     accuracy=accuracy_score(y_te, y_hat),
                     f1=f1_score(y_te, y_hat, zero_division=0))
            )

    df_pisa = pd.DataFrame(pisa_records)
    summary = df_pisa.groupby("percentile")[["accuracy", "f1"]].mean().round(3)
    best_by_f1 = summary["f1"].idxmax()

    print("\nAverage test performance over {:,} runs per percentile".format(N_RUNS))
    print(summary)
    print(f"\n>>> Best percentile (by mean F1): {best_by_f1}%")
else:
    print("\n[Trial 3 skipped]  Set PISA_MODEL = True to run this sweep.")


# ------------------------------------------------------------------
# 4.  Reload later
# ------------------------------------------------------------------
bundle = joblib.load(Path("models/logistic")   # pick the folder you want
                     .rglob("model.joblib")
                     .__next__())              # first hit
pipe_loaded = bundle["pipeline"]
thresh_loaded = bundle["threshold"]
