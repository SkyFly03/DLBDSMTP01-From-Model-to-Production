# confusion_matrix_heatmap.py 
# --------------------------------------------------
# Generates a visually enhanced heatmap-style confusion matrix.
# Includes annotations and color scaling for clarity.
# --------------------------------------------------

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(model, val_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Compute and normalize confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    labels = val_loader.dataset.classes

    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels, square=True,
                cbar_kws={"shrink": 0.75})
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Normalized Confusion Matrix (Validation Set)", fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # plot_confusion_matrix(model, val_loader)
    # Enables direct script execution to plot confusion matrix.
    # Useful for manual testing or isolated visual evaluation.
    pass