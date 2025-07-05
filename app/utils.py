# app/utils.py

import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize to 8x8 like sklearn's digits dataset
    image = image.resize((8, 8))

    # Convert to numpy array
    image_array = np.array(image)

    # Normalize pixel values: sklearn's digits are in range 0–16
    image_array = 16 - (image_array / 16)

    # Flatten to match model input shape (1, 64)
    return image_array.flatten()
