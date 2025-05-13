from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model_training import (
    train_linear_model,
    train_logistic_model,
    train_torch_logreg,
    load_pipeline,               # convenience wrapper
)

# ------------------------------------------------------------------
# 1.  Load & split your data  (X : 2‑D ndarray,  y : 1‑D ndarray)
# ------------------------------------------------------------------

PISA_MODEL = False

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
    penalty="l2",                # or "elasticnet"
    alpha_grid=(1e-4, 1e-3, 1e-2),
    use_smote=True,              # default
)

# pipe & threshold are auto‑saved under  models/logistic/<timestamp>/
# ------------------------------------------------------------------
# 2B.  OR…  Train a linear regressor (RMSE tuned)
# ------------------------------------------------------------------
lin_pipe = train_linear_model(
    X_train, y_train, penalty="l2", alpha_grid=(1e-4, 1e-3, 1e-2)
)

# ------------------------------------------------------------------
# 2C.  OR…  Train a PyTorch logistic model (mini‑batches + early stop)
# ------------------------------------------------------------------
torch_model = train_torch_logreg(
    X_train,
    y_train,
    batch_size=256,
    epochs=100,
    lr=1e-2,
)
# best checkpoint stored in  models/torch_logreg/<timestamp>/best_state.pt

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
import torch

def plot_cm(y_true, y_pred, title, ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Fail (0)", "Pass (1)"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.title(title)
    return cm

# Keep track so we can show all three figures side‑by‑side
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# ──────────────────────────────────────────────────────────────────
# A. the logistic‑pipeline you already evaluated (for completeness)
# ──────────────────────────────────────────────────────────────────
cm_log = plot_cm(y_test, y_pred, "Logistic‑pipeline", ax=axes[0])
print("\n[Logistic] report\n", classification_report(y_test, y_pred, digits=3))

# ──────────────────────────────────────────────────────────────────
# B. Linear‑regression pipeline   (probability ≈ clip(ŷ) )
# ──────────────────────────────────────────────────────────────────
prob_lin = lin_pipe.predict(X_test)           # raw continuous output
prob_lin = np.clip(prob_lin, 0, 1)            # ensure 0‑1 range
y_pred_lin = (prob_lin > 0.5).astype(int)     # basic 0.5 threshold

cm_lin = plot_cm(y_test, y_pred_lin, "Linear‑pipeline", ax=axes[1])
print("\n[Linear] report\n", classification_report(y_test, y_pred_lin, digits=3))

# ──────────────────────────────────────────────────────────────────
# C. PyTorch logistic model
# ──────────────────────────────────────────────────────────────────
torch_model.eval()
with torch.no_grad():
    logits = torch_model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
prob_torch = 1 / (1 + np.exp(-logits))        # sigmoid
y_pred_torch = (prob_torch > best_thresh).astype(int)

cm_torch = plot_cm(y_test, y_pred_torch, "PyTorch logistic", ax=axes[2])
print("\n[Torch] report\n", classification_report(y_test, y_pred_torch, digits=3))

plt.tight_layout()
plt.show()


import joblib
import numpy as np
from pathlib import Path

# 1️⃣  reload the bundle you just saved
log_dir = max(Path("models/logistic").iterdir(), key=lambda d: d.stat().st_mtime)
bundle  = joblib.load(log_dir / "model.joblib")
pipe    = bundle["pipeline"]                 # fitted pipeline

# 2️⃣  grab raw coefficients  (shape: [1, n_features])
coef = pipe.named_steps["clf"].coef_.ravel()

# 3️⃣  pick the 10 with largest |coef|
top_idx = np.argsort(np.abs(coef))[::-1][:10]
top_features = [(feature_names[i], coef[i]) for i in top_idx]

print("Top 10 log‑reg features (coef, sign matters):")
for name, weight in top_features:
    print(f"{name:35s}  {weight:+.4f}   (odds ×{np.exp(weight):.2f})")

lin_dir = max(Path("models/linear").iterdir(), key=lambda d: d.stat().st_mtime)
lin_pipe = joblib.load(lin_dir / "pipeline.joblib")

coef = lin_pipe.named_steps["sgdregressor"].coef_.ravel()
top_idx = np.argsort(np.abs(coef))[::-1][:10]
print("\nTop 10 linear‑reg predictors (per–SD impact on score):")
for i in top_idx:
    print(f"{feature_names[i]:35s}  {coef[i]:+.3f} points")

import torch

# reload best checkpoint; you already have 'torch_model' in RAM,
# but here's how to load from disk:
torch_dir = max(Path("models/torch_logreg").iterdir(), key=lambda d: d.stat().st_mtime)
state_dict = torch.load(torch_dir / "best_state.pt", map_location="cpu")

from model_training import TorchLogReg   # same class definition
torch_model = TorchLogReg(len(feature_names))
torch_model.load_state_dict(state_dict)
torch_model.eval()

weights = torch_model.linear.weight.detach().numpy().ravel()
top_idx  = np.argsort(np.abs(weights))[::-1][:10]

print("\nTop 10 PyTorch log‑reg features:")
for i in top_idx:
    w = weights[i]
    print(f"{feature_names[i]:35s}  {w:+.4f}   (odds ×{np.exp(w):.2f})")




# ------------------------------------------------------------------
# 4.  Reload later
# ------------------------------------------------------------------
bundle = joblib.load(Path("models/logistic")   # pick the folder you want
                     .rglob("model.joblib")
                     .__next__())              # first hit
pipe_loaded = bundle["pipeline"]
thresh_loaded = bundle["threshold"]
