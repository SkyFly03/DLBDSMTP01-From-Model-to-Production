# clean_class_folders.py (VS Code / Windows version)
# ----------------------------------------------------------
# Strictly enforces class folder alignment between train/val.
# Only retains folders that are shared and contain images.
# ----------------------------------------------------------

# clean_class_folders.py
import os
import shutil

base_paths = ["data/full/train", "data/full/val"]

shared_classes = sorted(set(os.listdir(base_paths[0])).intersection(os.listdir(base_paths[1])))

for base in base_paths:
    for cls in os.listdir(base):
        if cls not in shared_classes:
            full_path = os.path.join(base, cls)
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
                print(f"Removed: {full_path}")