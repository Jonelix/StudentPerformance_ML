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
