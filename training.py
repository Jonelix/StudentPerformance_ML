import numpy as np
from models import train_linear_model, train_logistic_model

# Simulated dataset (replace with real student data)
X_train = np.random.rand(100, 5)  # 100 samples, 5 features
y_train_reg = np.random.rand(100)  # Continuous target for regression
y_train_cls = np.random.randint(0, 2, 100)  # Binary target for classification

regularizer_weight = 0.01  # Regularization strength

# Train regression models
model1 = train_linear_model(X_train, y_train_reg, reg_type=None)  # Standard
model2 = train_linear_model(X_train, y_train_reg, reg_type="l1", alpha=regularizer_weight)  # Lasso
model3 = train_linear_model(X_train, y_train_reg, reg_type="l2", alpha=regularizer_weight)  # Ridge

# Train logistic models
model4 = train_logistic_model(X_train, y_train_cls, reg_type=None)  # Standard
model5 = train_logistic_model(X_train, y_train_cls, reg_type="l1", alpha=regularizer_weight)  # Lasso
model6 = train_logistic_model(X_train, y_train_cls, reg_type="l2", alpha=regularizer_weight)  # Ridge
