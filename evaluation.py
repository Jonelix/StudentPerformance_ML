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
