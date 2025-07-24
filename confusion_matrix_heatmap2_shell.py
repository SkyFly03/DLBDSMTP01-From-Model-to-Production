# confusion_matrix_heatmap_shell.py
# --------------------------------------------------
# Script to generate a normalized confusion matrix
# for the trained refund classification model using
# the validation set after shell-script training.
# --------------------------------------------------

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# --- Configuration ---
model_path = "refund_classifier_final.pt"
val_dir = os.path.join("data", "full", "full", "val")
output_path = "model_images/confusion_matrix_shell_model.png"

# --- Load model ---
model = torch.load(model_path, weights_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Load validation data ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_data = ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
class_names = val_data.classes

# --- Predict ---
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- Confusion matrix ---
cm = confusion_matrix(all_labels, all_preds, normalize="true")

# --- Plot heatmap ---
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Proportion"})
plt.title("Normalized Confusion Matrix (Shell-Trained Model)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(output_path)
print(f"Saved: {output_path}")
