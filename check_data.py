# clean_class_folders.py
# ----------------------------------------------------------
# Removes any class folders from train or val that do not
# appear in both directories. Keeps only shared class folders.
# ----------------------------------------------------------

import os
import shutil

# Define input paths
train_path = "data/full/train"
val_path = "data/full/val"
base_paths = [train_path, val_path]

# Find shared class folder names
train_classes = set(os.listdir(train_path))
val_classes = set(os.listdir(val_path))
shared_classes = sorted(train_classes.intersection(val_classes))

# Remove any non-shared folders
for base in base_paths:
    for cls in os.listdir(base):
        full_path = os.path.join(base, cls)
        if cls not in shared_classes and os.path.isdir(full_path):
            shutil.rmtree(full_path)
            print(f"Removed: {full_path}")
