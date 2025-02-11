import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from models import train_linear_model, train_logistic_model
from sklearn.metrics import mean_squared_error, accuracy_score

# Data processing
# 1. Load data
# 2. Separate data into training and testing sets
# 3. Balance the data
# 4. Normalize the data

# Load dataset
relative_path = "data/student_portugal/transformed_student_data.csv"
data_file = os.path.join(os.getcwd(), relative_path)
df = pd.read_csv(data_file)
print("Loaded datafile")

# Ensure all rows are unique
df = df.drop_duplicates()

# Split into training and testing sets (85% training, 15% testing)
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
print("Data split into training and testing sets")

# Define features and target variable
X_train = train_df.drop(columns=['failures'])
y_train_reg = train_df['failures']
y_train_cls = (train_df['failures'] > 0).astype(int)

X_test = test_df.drop(columns=['failures'])
y_test_reg = test_df['failures']
y_test_cls = (test_df['failures'] > 0).astype(int)

regularizer_weight = 0.01  # Regularization strength

print("Classifier identified")

# Train regression models
print("Training RegN")
model1 = train_linear_model(X_train, y_train_reg, reg_type=None)  # Standard
print("Training Reg1")
model2 = train_linear_model(X_train, y_train_reg, reg_type="l1", alpha=regularizer_weight)  # Lasso
print("Training Reg2")
model3 = train_linear_model(X_train, y_train_reg, reg_type="l2", alpha=regularizer_weight)  # Ridge

# Train logistic models
print("Training LogN")
model4 = train_logistic_model(X_train, y_train_cls, reg_type=None)  # Standard
print("Training Log1")
model5 = train_logistic_model(X_train, y_train_cls, reg_type="l1", alpha=regularizer_weight)  # Lasso
print("Training Log2")
model6 = train_logistic_model(X_train, y_train_cls, reg_type="l2", alpha=regularizer_weight)  # Ridge

# Testing Section
print("Evaluating models")

# Evaluate regression models using accuracy
acc1_reg = accuracy_score((y_test_reg > 0).astype(int), (model1.predict(X_test) > 0).astype(int))
acc2_reg = accuracy_score((y_test_reg > 0).astype(int), (model2.predict(X_test) > 0).astype(int))
acc3_reg = accuracy_score((y_test_reg > 0).astype(int), (model3.predict(X_test) > 0).astype(int))
print(f"Accuracy for Standard Linear Regression: {acc1_reg}")
print(f"Accuracy for Lasso Regression: {acc2_reg}")
print(f"Accuracy for Ridge Regression: {acc3_reg}")

# Evaluate logistic models
acc1 = accuracy_score(y_test_cls, model4.predict(X_test))
acc2 = accuracy_score(y_test_cls, model5.predict(X_test))
acc3 = accuracy_score(y_test_cls, model6.predict(X_test))
print(f"Accuracy for Standard Logistic Regression: {acc1}")
print(f"Accuracy for Lasso Logistic Regression: {acc2}")
print(f"Accuracy for Ridge Logistic Regression: {acc3}")
