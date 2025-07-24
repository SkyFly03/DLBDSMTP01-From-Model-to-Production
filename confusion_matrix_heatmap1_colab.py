# confusion_matrix_heatmap1_colab.py
# ---------------------------------------------------------
# Generates confusion matrix heatmap using the model 
# trained in Google Colab.
# Output is saved as an image file.
# ---------------------------------------------------------

import torch
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIGURATION ---
model_path = "refund_classifier_final.pt"
val_dir = os.path.join("data", "full", "val")
output_path = "model_images/confusion_matrix_colab_model.png"

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- DATA LOADING ---
val_data = datasets.ImageFolder(val_dir, transform=transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)

# --- LOAD MODEL ---
model = torch.load(model_path)
model.eval()

# --- PREDICT AND COLLECT RESULTS ---
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# --- CONFUSION MATRIX ---
cm = confusion_matrix(all_labels, all_preds)
labels = val_data.classes

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Colab-Trained Model")
plt.tight_layout()
plt.savefig(output_path)
print(f"Saved: {output_path}")
