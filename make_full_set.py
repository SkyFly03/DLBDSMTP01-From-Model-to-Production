# make_full_set.py
# ------------------------------------------------
# Builds the full refund training dataset by copying
# matching image files listed in styles.csv.
# Creates data/refund_full_set/images/ and a filtered
# styles.csv for later dataset preparation.
# ------------------------------------------------

import os
import pandas as pd
import shutil

print("Creating full dataset...")

# Define input and output paths
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_DIR = os.path.join(DATA_DIR, "refund_full_set")
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# Load and filter metadata
styles_path = os.path.join(DATA_DIR, "styles.csv")
df = pd.read_csv(styles_path, on_bad_lines="skip")
df = df.dropna(subset=["id", "articleType"])
df["filename"] = df["id"].astype(str) + ".jpg"

# Filter for images that actually exist
available_images = set(os.listdir(IMAGE_DIR))
filtered_df = df[df["filename"].isin(available_images)]

# Copy matching image files to refund_full_set/images/
copied = 0
for img_file in filtered_df["filename"]:
    src = os.path.join(IMAGE_DIR, img_file)
    dst = os.path.join(OUTPUT_DIR, "images", img_file)
    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1

# Save filtered metadata
filtered_df.drop(columns=["filename"]).to_csv(
    os.path.join(OUTPUT_DIR, "styles.csv"), index=False
)

print(f"Copied {copied} images to {OUTPUT_DIR}/images")
