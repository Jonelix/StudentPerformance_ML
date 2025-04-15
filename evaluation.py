import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def fscore(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if(tp + fp == 0 or tp + fn == 0):
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plots a confusion matrix with a red color scale (white to red gradient).
    
    Parameters:
    y_true (array-like): True class labels.
    y_pred (array-like): Predicted class labels.
    model_name (str): Name of the model.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    cm_matrix = np.array([[tp, fn], [fp, tn]])
    labels = np.array([["TP", "FN"], ["FP", "TN"]])
    
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Reds', cbar=True, linewidths=1, linecolor='black',
                     xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"], annot_kws={"size": 14, "weight": "bold"})
    
    for i in range(2):
        for j in range(2):
            text = labels[i, j]
            ax.text(j + 0.1, i + 0.1, text, ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(model_name)
    plt.show()

def plot_model_acc_and_fscore(model_accs, models_fscores, model_names):
    x = np.arange(len(model_names))  # X locations for the groups
    width = 0.4  # Width of the bars
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Create first axis for accuracy
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    bars1 = ax1.bar(x - width/2, model_accs, width, label='Accuracy', color='tab:blue', alpha=0.7)
    ax1.set_ylim(0, 1.1)  # Assuming accuracy is between 0 and 1
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Create second y-axis for F-score
    ax2 = ax1.twinx()
    ax2.set_ylabel('F-score', color='tab:red')
    bars2 = ax2.bar(x + width/2, models_fscores, width, label='F-score', color='tab:red', alpha=0.7)
    ax2.set_ylim(0, 1.1)  # Assuming F-score is between 0 and 1
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Add labels and title
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=30, ha='right')
    fig.suptitle('Model Accuracy and F-score Comparison')
    
    # Create a single legend
    fig.legend([bars1, bars2], ['Accuracy', 'F-score'], loc='upper left', bbox_to_anchor=(0.12, 0.975))
    
    plt.show()

def display_model_weights(model, df, num_features=5):
    
    feature_names = df.columns  # Assuming last column is target
    weights = model.coef_.ravel()
    
    weight_df = pd.DataFrame({
        "Feature": feature_names,
        "Absolute value weight": np.abs(weights),
        "Weight": weights
    })
    
    weight_df = weight_df.sort_values(by="Absolute value weight", ascending=False)
    
    
    print(weight_df[:num_features])
    print("Displayed sorted feature weights")

from sklearn.metrics import accuracy_score
def evaluate_model(model, X_test, y_test, model_threshold=0, model_name="Model"):
    y_pred = model.predict(X_test) > model_threshold
    #Add line here
    acc = accuracy_score(y_test, y_pred)
    f1 = fscore(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, model_name)
    print(f"{model_name} accuracy: {acc}")
    print(f"{model_name} f1 score: {f1}")
    return acc, f1

def evaluate_model_debug(model, X_test, y_test, model_threshold=0.5, model_name="Model"):
    # Get prediction probabilities (assumes binary classification)
    y_scores = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

    # Threshold the scores
    y_pred = y_scores > model_threshold

    # Debug: see first few prediction scores
    print(f"Sample predicted scores (positive class): {y_scores[:10]}")
    print(f"Applied threshold: {model_threshold}")
    print(f"Predicted labels: {y_pred[:10]}")

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    f1 = fscore(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, model_name)
    print(f"{model_name} accuracy: {acc}")
    print(f"{model_name} f1 score: {f1}")

    return acc, f1

def evaluate_model_thresholded_regression(model, X_test, y_test, model_threshold=0.5, model_name="Thresholded Regressor", debug=False):
    y_scores = model.predict(X_test)
    y_pred = y_scores < model_threshold
    if debug:
        print(f"[{model_name}] Sample predicted scores: {y_scores[:10]}")
        print(f"[{model_name}] Applied threshold: {model_threshold}")
        print(f"[{model_name}] Predicted labels: {y_pred[:10]}")
        print(f"[{model_name}] Sample true values: {y_test[:10].values}")

    acc = accuracy_score(y_test, y_pred)
    f1 = fscore(y_test, y_pred)
    print(f"[{model_name}] Accuracy: {acc}")
    print(f"[{model_name}] F1 Score: {f1}")
    plot_confusion_matrix(y_test, y_pred, model_name)

    return acc, f1

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model_regression(model, X_test, y_test, model_name="Regressor"):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[{model_name}] RMSE: {rmse:.4f}")
    print(f"[{model_name}] MAE: {mae:.4f}")
    print(f"[{model_name}] R² Score: {r2:.4f}")

    return rmse, mae, r2

def evaluate_model_classifier(model, X_test, y_test, model_threshold=0.5, model_name="Classifier", debug=False):
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = y_scores > model_threshold
    if debug:
        print(f"[{model_name}] Sample predicted probabilities: {y_scores[:10]}")
        print(f"[{model_name}] Applied threshold: {model_threshold}")
        print(f"[{model_name}] Predicted labels: {y_pred[:10]}")
        print(f"[{model_name}] Sample true values: {y_test[:10].values}")

    acc = accuracy_score(y_test, y_pred)
    f1 = fscore(y_test, y_pred)
    
    print(f"[{model_name}] Accuracy: {acc}")
    print(f"[{model_name}] F1 Score: {f1}")
    plot_confusion_matrix(y_test, y_pred, model_name)

    return acc, f1


