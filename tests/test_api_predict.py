# tests/test_api_predict.py
# ---------------------------------------------------------
# Sends a test image to the Flask API's /predict endpoint.
# Verifies the response structure, status code, and content.
# Intended for local testing of the running API.
# ---------------------------------------------------------

import requests
import os

API_URL = "http://localhost:5000/predict"
SAMPLE_IMAGE = "tests/sample.jpg"  # Replace with a real image path

def test_prediction():
    if not os.path.exists(SAMPLE_IMAGE):
        print("Sample image not found. Test skipped.")
        return

    with open(SAMPLE_IMAGE, "rb") as img:
        files = {"file": img}
        response = requests.post(API_URL, files=files)

    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    data = response.json()

    assert "predicted_class" in data, "Response missing 'predicted_class'"
    assert "probabilities" in data, "Response missing 'probabilities'"
    assert isinstance(data["probabilities"], list), "Probabilities should be a list"

    print("Test passed. Predicted class:", data["predicted_class"])

if __name__ == "__main__":
    test_prediction()
