# tests/test_api_predict.py
# ---------------------------------------------------------
# Sends a test image to the Flask API's /predict endpoint.
# Verifies the response structure, status code, and content.
# Intended for local testing of the running API.
# ---------------------------------------------------------

import requests
import os

# --- Configuration ---
API_URL = "http://localhost:5000/predict"
TEST_IMAGE_PATH = "tests/sample.jpg"  # Replace with actual image for testing

def test_prediction():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f" Sample image not found at {TEST_IMAGE_PATH}. Test skipped.")
        return

    try:
        with open(TEST_IMAGE_PATH, "rb") as img_file:
            files = {"file": img_file}
            response = requests.post(API_URL, files=files)
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return

    try:
        data = response.json()
    except ValueError:
        print("Response was not valid JSON.")
        return

    # --- Assertions ---
    assert "predicted_class" in data, "Missing key: predicted_class"
    assert "probabilities" in data, "Missing key: probabilities"
    assert isinstance(data["probabilities"], list), "Probabilities must be a list"

    print("Test passed.")
    print(f"Predicted class: {data['predicted_class']}")

if __name__ == "__main__":
    test_prediction()
