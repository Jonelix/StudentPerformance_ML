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
ALPHA_GRID = (1e-4, 1e-3, 1e-2)  # for SGDClassifier / SGDRegressor regularization strength
LOG_REG_USE_SMOTE = True  # for logistic regression only
LIN_THRESHOLD_TUNING = True  # for both models
CV_SPLITS = 5  # for both models

if PISA_MODEL:
    PASS = "GENERAL_SCORE"
    DATA_PATH = "data/pisa_2022/pisa_clean.csv"
    PERCENTILE = 36
    GRADE_THRESH = 400
else:
    PASS = "G1"
    DATA_PATH = "data/student_portugal/portugal_clean.csv"
    GRADE_THRESH = 10

df = pd.read_csv(DATA_PATH)

if PISA_MODEL:
    GRADE_THRESH = df['GENERAL_SCORE'].quantile(PERCENTILE / 100)

df["PASS"] = (df[PASS] >= GRADE_THRESH).astype(int)
target = "PASS"  # target column name

# 2. define a list of columns that leak target information
LEAK_COLS = [PASS]



# 3. build the feature matrix
X_df = df.drop(columns=["PASS"] + [c for c in LEAK_COLS if c in df.columns])
feature_names = X_df.columns.to_numpy()
X        = X_df.values             # the array the model sees

y = df[target].values                  # numeric 0 / 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
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
    cv_splits=CV_SPLITS,               # default
)

# pipe & threshold are auto‑saved under  models/logistic/<timestamp>/
# ------------------------------------------------------------------
# 2B.  OR…  Train a linear regressor (RMSE tuned)
# ------------------------------------------------------------------
lin_pipe, lin_best_thresh = train_linear_model(
    X_train, 
    y_train, 
    penalty=REGULARIZER, 
    alpha_grid=ALPHA_GRID,
    threshold_tuning=LIN_THRESHOLD_TUNING,  # default
    cv_splits=CV_SPLITS,               # default
)

# ------------------------------------------------------------------
# 3.  Inference with the sklearn pipeline
# ------------------------------------------------------------------
prob_test = pipe.predict_proba(X_test)[:, 1]
y_pred = (prob_test > best_thresh).astype(int)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, digits=3))

# ──────────────────────────────────────────────────────────────────
# Helper to make one nice confusion‑matrix plot
# ──────────────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

def plot_cm(y_true, y_pred, title, ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fail (0)", "Pass (1)"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    if ax is not None:
        ax.set_title(title)  # Explicitly set the title on the provided axes
    return cm

# Keep track so we can show all three figures side‑by‑side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Logistic regression confusion matrix
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


# ------------------------------------------------------------------
# 4.  Reload later
# ------------------------------------------------------------------
bundle = joblib.load(Path("models/logistic")   # pick the folder you want
                     .rglob("model.joblib")
                     .__next__())              # first hit
pipe_loaded = bundle["pipeline"]
thresh_loaded = bundle["threshold"]
