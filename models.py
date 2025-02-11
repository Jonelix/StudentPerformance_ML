import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.preprocessing import StandardScaler
import os

# Automatically select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_SAVE_PATH = "trained_models/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

class LinearRegressionModel(nn.Module):
    """
    Standard linear regression using stochastic gradient descent (SGD)
    """
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1).to(device)  # Move model to device

    def forward(self, x):
        return self.linear(x)

def train_linear_model(X_train, y_train, reg_type=None, alpha=0.01):
    """
    Train a linear regression model using SGD with optional regularization.
    Automatically uses GPU if available.
    """
    if reg_type is None:
        model = SGDRegressor(loss="squared_error", penalty=None, learning_rate="optimal")
        MODEL_SAVE_PATH = "trained_models/LiRe_NoReg"
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    elif reg_type == "l1":
        model = SGDRegressor(loss="squared_error", penalty="l1", alpha=alpha, learning_rate="optimal")
        MODEL_SAVE_PATH = "trained_models/LiRe_L1"
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    elif reg_type == "l2":
        model = SGDRegressor(loss="squared_error", penalty="l2", alpha=alpha, learning_rate="optimal")
        MODEL_SAVE_PATH = "trained_models/LiRe_L2"
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    else:
        raise ValueError("Invalid regularization type. Choose 'l1' or 'l2'.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Scale the input data
    model.fit(X_train, y_train)
    return model

class LogisticRegressionModel(nn.Module):
    """
    Logistic regression using stochastic gradient descent (SGD)
    """
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1).to(device)  # Move model to device

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def train_logistic_model(X_train, y_train, reg_type=None, alpha=0.01):
    """
    Train a logistic regression model using SGD with optional regularization.
    Automatically uses GPU if available.
    """
    if reg_type is None:
        model = SGDClassifier(loss="log_loss", penalty=None, learning_rate="optimal")
        MODEL_SAVE_PATH = "trained_models/LoRe_NoReg"
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    elif reg_type == "l1":
        model = SGDClassifier(loss="log_loss", penalty="l1", alpha=alpha, learning_rate="optimal")
        MODEL_SAVE_PATH = "trained_models/LoRe_L1"
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    elif reg_type == "l2":
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, learning_rate="optimal")
        MODEL_SAVE_PATH = "trained_models/LoRe_L2"
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    else:
        raise ValueError("Invalid regularization type. Choose 'l1' or 'l2'.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model.fit(X_train, y_train)
    return model

def train_pytorch_linear_model(X_train, y_train, epochs=100, lr=0.01):
    """
    Train a PyTorch-based linear regression model using GPU if available.
    """
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    model = LinearRegressionModel(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Save model at each epoch
        model_filename = os.path.join(MODEL_SAVE_PATH, f"LiRe_NoReg_Epoch_{epoch:04d}.pth")
        torch.save(model.state_dict(), model_filename)
        print(f"Epoch {epoch:04d}: Model saved as {model_filename}")

    return model

