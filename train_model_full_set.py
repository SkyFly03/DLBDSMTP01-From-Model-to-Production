# train_model_full_set.py
# ---------------------------------------------------------
# Trains a ResNet50 image classification model on refund data.
# Logs training and validation metrics using MLflow for tracking.
# Saves the final model as refund_classifier_final.pt for the API.
# Designed for reproducible and scalable model development.
# ---------------------------------------------------------

import os
import logging
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# --- CONFIGURATION ---
EPOCHS = 30
LR = 0.001
BATCH_SIZE = 32
MODEL_PATH = "refund_classifier_final.pt"
DATA_DIR = "/content/data/full"

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # --- TRANSFORMS ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # --- DATA LOADERS ---
    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    logging.info(f"Train class distribution: {Counter(label for _, label in train_data.imgs)}")
    logging.info(f"Val class distribution:   {Counter(label for _, label in val_data.imgs)}")

    # --- MODEL SETUP ---
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(train_data.classes))
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # --- TRAINING WITH MLflow ---
    mlflow.set_experiment("Refund_Classifier")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": EPOCHS,
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "optimizer": "SGD",
            "model": "resnet50"
        })

        best_val_acc = 0
        patience = 5
        trigger = 0

        for epoch in range(EPOCHS):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = correct / total

            # --- VALIDATION ---
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            logging.info(f"Epoch {epoch+1}/{EPOCHS} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                trigger = 0
            else:
                trigger += 1

            if trigger >= patience:
                logging.info("Early stopping triggered.")
                break

        # --- SAVE MODEL ---
        torch.save(model, MODEL_PATH)
        mlflow.pytorch.log_model(model, "model")
        logging.info(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
