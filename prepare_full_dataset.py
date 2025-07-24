# prepare_full_dataset.py
# ----------------------------------------------------------
# Splits the refund dataset into training and validation sets.
# Outputs are saved under data/full/train and data/full/val.
# ----------------------------------------------------------

import os
import shutil
import random

SOURCE_DIR = "data/refund_full_set"
DEST_DIR = "data/full/full/"
TRAIN_RATIO = 0.8

# Create destination folders
train_path = os.path.join(DEST_DIR, "train")
val_path = os.path.join(DEST_DIR, "val")
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Process each class folder
for category in os.listdir(SOURCE_DIR):
    category_path = os.path.join(SOURCE_DIR, category)
    if not os.path.isdir(category_path):
        continue

    print(f"Processing class: {category}")
    images = os.listdir(category_path)
    random.shuffle(images)
    split = int(len(images) * TRAIN_RATIO)

    train_images = images[:split]
    val_images = images[split:]

    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(val_path, category), exist_ok=True)

    for img in train_images:
        shutil.copy2(os.path.join(category_path, img), os.path.join(train_path, category, img))
    for img in val_images:
        shutil.copy2(os.path.join(category_path, img), os.path.join(val_path, category, img))

print("Dataset split completed: data/full/full/train and data/full/full/val created.")
