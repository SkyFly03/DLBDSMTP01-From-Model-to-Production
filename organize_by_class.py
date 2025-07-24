# organize_by_class.py
# --------------------------------------------------
# Organizes training and validation images into
# class folders using labels from styles.csv.
# Required for compatibility with torchvision ImageFolder.
# --------------------------------------------------

import os
import shutil
import pandas as pd

# Paths
CSV_PATH = "data/styles.csv"
TRAIN_DIR = "data/full/full/train"
VAL_DIR = "data/full/full/val"

# Read metadata
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
df = df.dropna(subset=["id", "articleType"])
id_to_label = dict(zip(df["id"].astype(str), df["articleType"].astype(str)))

# Helper function to organize images into class folders
def organize_images_by_label(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    for img_file in image_files:
        img_id = img_file.split(".")[0]
        label = id_to_label.get(img_id)
        if label:
            target_dir = os.path.join(folder_path, label)
            os.makedirs(target_dir, exist_ok=True)
            src_path = os.path.join(folder_path, img_file)
            dst_path = os.path.join(target_dir, img_file)
            shutil.move(src_path, dst_path)

# Organize both train and val datasets
organize_images_by_label(TRAIN_DIR)
organize_images_by_label(VAL_DIR)

print("Images successfully organized into class folders in train and val.")
