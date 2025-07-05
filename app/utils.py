# app/utils.py
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((28, 28))  # Resize to match training
    image_array = np.array(image)
    image_array = image_array.flatten() / 255.0  # Normalize
    return image_array
