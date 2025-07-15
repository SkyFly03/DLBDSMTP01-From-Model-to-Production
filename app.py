# app.py
# ------------------------------------------------
# Flask API for refund item image classification.
# Accepts an image via POST request, predicts its class
# using a trained PyTorch model, and logs the result
# to a PostgreSQL database.
# ------------------------------------------------

from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import psycopg2
import os

# --- Load trained model ---
model = torch.load("refund_classifier_final.pt", map_location=torch.device("cpu"))
model.eval()

# --- Initialize Flask app ---
app = Flask(__name__)

# --- Image transformation pipeline ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Database connection settings ---
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'port': os.getenv("DB_PORT", "5432"),
    'dbname': os.getenv("DB_NAME", "refunds"),
    'user': os.getenv("DB_USER", "postgres"),
    'password': os.getenv("DB_PASSWORD", "postgres")
}

# --- Log prediction to PostgreSQL ---
def log_prediction(filename, prediction):
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
            (filename, prediction)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("Database logging failed:", e)

# --- API route: /predict ---
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a POST request with an image file and returns a predicted class.
    Logs the result to a PostgreSQL database.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probs).item()
            predicted_class = str(predicted_idx)

        # Log to database
        log_prediction(file.filename, predicted_class)

        return jsonify({
            "filename": file.filename,
            "predicted_class": predicted_class,
            "probabilities": probs.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run app locally ---
if __name__ == "__main__":
    app.run(debug=True)
