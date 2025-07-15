# check_data.py
# --------------------------------------------
# This script checks whether all image files listed in styles.csv exist in the image folder.
# It helps ensure the dataset is clean before creating a subset or full training set.
# --------------------------------------------

import os
import pandas as pd

# Define dataset paths
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "styles.csv")

# Load metadata from styles.csv
df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
image_files = set(os.listdir(IMAGE_DIR))

# Track images listed in the CSV but missing from the folder
missing_images = []
for img_id in df['id']:
    img_file = f"{int(img_id)}.jpg"
    if img_file not in image_files:
        missing_images.append(img_file)

# Report dataset status
print(f"Total entries in styles.csv: {len(df)}")
print(f"Total image files found: {len(image_files)}")
print(f"Missing images: {len(missing_images)}")

