# utils.py
# --------------------------------------------
# Utility functions for model prediction, device setup, and file handling.
# Keeps code clean and reusable across scripts.
# --------------------------------------------

import torch

# Standard class names used in training and prediction
from . import CLASS_NAMES

def get_device():
    """
    Returns the best available device: CUDA if available, else CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_prediction(filename, predicted_index):
    """
    Converts a predicted index into a readable dictionary with class name.

    Args:
        filename (str): Original image file name
        predicted_index (int): Index predicted by the model

    Returns:
        dict: Result containing filename and predicted class
    """
    return {
        "filename": filename,
        "predicted_class": CLASS_NAMES[predicted_index]
    }
