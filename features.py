import numpy as np
from PIL import Image

def extract_features(image_path: str) -> np.ndarray:
    """
    Baseline feature extractor:
    - Resize to 64x64
    - Normalize to 0..1
    - Flatten into 1D feature vector
    Returns shape: (1, n_features)
    """
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    feat = arr.flatten()
    return feat.reshape(1, -1)