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

plot_model_acc_and_fscore([0.8, 0.85, 0.9], [0.75, 0.8, 0.88], ['Model A', 'Model B', 'Model C'])
    