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

PISA_MODEL = False
REGULARIZER = "l2"  # Can be "none", "l1", "l2"
ALPHA_GRID = tuple(10.0 ** np.arange(-6, 3))  # for SGDClassifier / SGDRegressor regularization strength
LIN_THRESHOLD_TUNING = False  
CV_SPLITS = 3  # for both models
TEST_SIZE=0.2
RANDOM_STATE = 42
#-------------SMOTE--------------------
LOG_REG_USE_SMOTE = True  # for logistic regression only
LOG_REG_USE_SMOTE_TOMEK = False  # for logistic regression only
LOG_REG_USE_SMOTE_ENN = False #for logistic regression only
#---------------------------------


if PISA_MODEL:
    PASS = "GENERAL_SCORE"
    DATA_PATH = "data/pisa_2022/pisa_clean.csv"
    PISA_PERCENTAGE = 34
    GRADE_THRESH = 400
else:
    PASS = "G1"
    DATA_PATH = "data/student_portugal/portugal_clean.csv"
    GRADE_THRESH = 10

df = pd.read_csv(DATA_PATH)

if PISA_MODEL:
    GRADE_THRESH = df['GENERAL_SCORE'].quantile(PISA_PERCENTAGE / 100)

df["PASS"] = (df[PASS] >= GRADE_THRESH).astype(int)
target = "PASS"  # target column name

# 2. define a list of columns that leak target information
LEAK_COLS = [PASS]
df["PASS"] = 1 - df["PASS"]

# 3. build the feature matrix
X_df = df.drop(columns=["PASS"] + [c for c in LEAK_COLS if c in df.columns])
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

def plot_cm(y_true, y_pred, title, ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Pass (0)", "Fail (1)"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    if ax is not None:
        ax.set_title(title)  # Explicitly set the title on the provided axes
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
# Trial 1: How does model performance change with different regularizers? (None, L1, L2) | n_runs = 1000
# ──────────────────────────────────────────────────────────────────
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

N_RUNS     = 100         # ↳ heavy – trim if your machine struggles
PENALTIES  = ("none", "l1", "l2")

log_records, lin_records = defaultdict(list), defaultdict(list)
trained_log_models = {}
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



# ──────────────────────────────────────────────────────────────────
# Trial 2 – aggregate *coefficient strength* instead of frequency
# ──────────────────────────────────────────────────────────────────
from collections import defaultdict

TOP_N_RUNS     = 10        # best runs per penalty
COEF_MIN       = 1e-2      # importance threshold
TOP_M_FEATURES = 20        # how many lines to print per penalty
LOG_PATH       = "models/df/df_log.csv"

df_log = pd.read_csv(LOG_PATH)

try:
    trained_log_models
except NameError:
    trained_log_models = joblib.load("models/cached_logistic_models.joblib")

# ② two nested dicts per penalty:  strength_sum  &  signed_sum
strength_sum = defaultdict(lambda: defaultdict(float))
signed_sum   = defaultdict(lambda: defaultdict(float))

for pen, grp in df_log.groupby("penalty"):
    top_runs = grp.nlargest(TOP_N_RUNS, "f1")

    for run_id in top_runs["run"]:
        pipe, _ = trained_log_models[(run_id, pen)]
        coefs = pipe.named_steps["clf"].coef_.ravel()

        for feat, coef in zip(feature_names, coefs):
            if abs(coef) >= COEF_MIN:
                strength_sum[pen][feat] += abs(coef)
                signed_sum[pen][feat]   += coef          # keep sign

# ③ pretty‑print with a sign column
for pen in strength_sum:
    rows = []
    for feat, total_strength in strength_sum[pen].items():
        s = signed_sum[pen][feat]
        sign = "+" if s > 0 else ("–" if s < 0 else "0")
        rows.append((feat, total_strength, sign))

    rows = sorted(rows, key=lambda t: t[1], reverse=True)[:TOP_M_FEATURES]

    print(f"\nLogistic Regression – {pen.upper()}  "
          f"(top {TOP_N_RUNS} runs, coef ≥ {COEF_MIN})")
    print("-" * 72)
    print(f"{'Feature':30s} {'Σ|coef|':>12s}  {'Sign'}")
    for feat, mag, sign in rows:
        print(f"{feat:<30s} {mag:12.4f}   {sign}")

# ──────────────────────────────────────────────────────────────────
# ④ show the best single model for each penalty
# ──────────────────────────────────────────────────────────────────
BEST_METRIC   = "f1"          # or "accuracy", "precision", …
MAX_FEATS_SHOWN = 25          # how many lines per best‑model table

for pen, grp in df_log.groupby("penalty"):

    # find the run‑id of the best model for this penalty
    best_row   = grp.loc[grp[BEST_METRIC].idxmax()]
    best_run   = int(best_row["run"])
    best_score = best_row[BEST_METRIC]

    best_pipe, best_thr = trained_log_models[(best_run, pen)]

    # pull coefficients (still on scaled basis)
    coefs = best_pipe.named_steps["clf"].coef_.ravel()
    feat_coef = sorted(zip(feature_names, coefs),
                       key=lambda t: abs(t[1]), reverse=True)

    print(f"\n=====  BEST {pen.upper()} MODEL  =====")
    print(f"run {best_run} | {BEST_METRIC} = {best_score:.3f} | "
          f"prob‑threshold = {best_thr:.3f}")
    print("-" * 50)
    print(f"{'Feature':30s} {'Coef':>10}")
    for feat, c in feat_coef[:MAX_FEATS_SHOWN]:
        print(f"{feat:<30s} {c:10.4f}")


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
        tmp_df["PASS"] = (tmp_df["GENERAL_SCORE"] >= thresh).astype(int)

        X_tmp = tmp_df.drop(columns=["PASS"] + [c for c in LEAK_COLS if c in tmp_df.columns]).values
        y_tmp = tmp_df["PASS"].values

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
