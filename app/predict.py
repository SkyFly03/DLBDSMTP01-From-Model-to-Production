# predict.py
# --------------------------------------------
# Loads a trained model and predicts product categories from local images.
# Can be used for quick testing or integration in batch scripts.
# --------------------------------------------

import os
import torch
from PIL import Image
from torchvision import transforms
from model import load_model
from . import CLASS_NAMES

# Define image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_path, model, device=None):
    """
    Predicts the product category for a single image.

    Args:
        image_path (str): Path to the image file
        model (torch.nn.Module): Loaded PyTorch model
        device (torch.device, optional): CPU or CUDA device

    Returns:
        str: Predicted class name
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return CLASS_NAMES[predicted.item()]

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("refund_classifier_final.pt", device=device)

    # Folder of test images
    test_folder = "data/refund_subset/images"
    for file in os.listdir(test_folder):
        if file.lower().endswith(".jpg"):
            path = os.path.join(test_folder, file)
            prediction = predict_image(path, model, device)
            print(f"{file} → {prediction}")
