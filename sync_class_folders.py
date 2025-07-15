# sync_class_folders.py
# -----------------------------------------------------
# Ensures that data/full/train and data/full/val
# contain the same set of class folders.
# Creates any missing folders if they do not exist.
# -----------------------------------------------------

import os

train_path = "data/full/train"
val_path = "data/full/val"

train_classes = set(os.listdir(train_path))
val_classes = set(os.listdir(val_path))

# Combine all class names found in either folder
all_classes = sorted(train_classes.union(val_classes))

# Create any missing class folders
for cls in all_classes:
    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(val_path, cls), exist_ok=True)

print(f"Synchronized class folders across train/val. Total classes: {len(all_classes)}")
