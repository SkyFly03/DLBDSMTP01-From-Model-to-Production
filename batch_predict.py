# batch_predict.py
# ------------------------------------------------
# Runs predictions on a folder of new refund images
# using a trained PyTorch model. Each result is logged
# to a PostgreSQL database with timestamp information.
# Supports daily batch automation via scheduling.
# ------------------------------------------------

import os
import time
import torch
import psycopg2
import schedule
from PIL import Image
from datetime import datetime
from torchvision import transforms

# --- CONFIGURATION ---
IMAGE_DIR = "./new_images"
MODEL_PATH = "refund_classifier_final.pt"

DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'port': os.getenv("DB_PORT", "5432"),
    'dbname': os.getenv("DB_NAME", "refunds"),
    'user': os.getenv("DB_USER", "postgres"),
    'password': os.getenv("DB_PASSWORD", "postgres")
}

# --- IMAGE TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- DATABASE LOGGING ---
def log_prediction(filename, predicted_class):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                filename TEXT,
                predicted_class TEXT,
                timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute(
            "INSERT INTO predictions (filename, predicted_class) VALUES (%s, %s)",
            (filename, predicted_class)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Logging failed for {filename}: {e}")

# --- PREDICTION FUNCTION ---
def run_batch_predictions():
    print("Running scheduled batch prediction...")
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.eval()

    for file in os.listdir(IMAGE_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(IMAGE_DIR, file)
            try:
                image = Image.open(path).convert("RGB")
                img_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.nn.functional.softmax(output[0], dim=0)
                    pred_idx = torch.argmax(prob).item()
                    pred_class = str(pred_idx)
                    print(f"{file} → Predicted class: {pred_class}")
                    log_prediction(file, pred_class)
            except Exception as e:
                print(f"Prediction failed for {file}: {e}")

# --- DAILY SCHEDULING ---
schedule.every().day.at("03:00").do(run_batch_predictions)

if __name__ == "__main__":
    print("Batch prediction scheduler started. Waiting for daily run at 03:00...")
    while True:
        schedule.run_pending()
        time.sleep(60)
