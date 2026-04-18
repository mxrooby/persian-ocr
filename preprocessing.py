"""
preprocessing.py
─────────────────
Robust OpenCV preprocessing pipeline.
Converts ANY real-world photo of a Persian character into a clean
white-background black-text image matching the training data format.

Pipeline:
  1. Grayscale
  2. Denoise
  3. Adaptive threshold (handles varied lighting/shadows)
  4. Find character bounding box and crop with padding
  5. Invert if needed (ensure black text on white)
  6. Resize to target size
"""
import cv2
import numpy as np

def preprocess_image(image_np, target_size=64):
    """
    Takes a numpy RGB image (from Gradio upload).
    Returns a preprocessed grayscale numpy array ready for CNN.
    """
    # Step 1: Grayscale
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()

    # Step 2: Denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 3: Adaptive threshold — handles uneven lighting, shadows, phone photos
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10
    )

    # Step 4: Ensure black text on white background
    # Count black vs white pixels — if more black than white, invert
    black_pixels = np.sum(binary == 0)
    white_pixels = np.sum(binary == 255)
    if black_pixels > white_pixels:
        binary = cv2.bitwise_not(binary)

    # Step 5: Morphological cleanup — remove noise dots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Step 6: Find the character bounding box and crop
    # Invert to find contours (contours need white object on black bg)
    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get bounding box of ALL contours combined (the whole character)
        x_min = min(cv2.boundingRect(c)[0] for c in contours)
        y_min = min(cv2.boundingRect(c)[1] for c in contours)
        x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
        y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)

        # Add padding (15% of character size)
        h_img, w_img = binary.shape
        pad_x = max(10, int((x_max - x_min) * 0.15))
        pad_y = max(10, int((y_max - y_min) * 0.15))
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w_img, x_max + pad_x)
        y_max = min(h_img, y_max + pad_y)

        cropped = binary[y_min:y_max, x_min:x_max]

        # Only use crop if it's a reasonable size
        if cropped.size > 100:
            binary = cropped

    # Step 7: Resize to target size with padding to maintain aspect ratio
    h, w = binary.shape
    scale = (target_size - 10) / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to square
    canvas = np.ones((target_size, target_size), dtype=np.uint8) * 255
    y_off  = (target_size - new_h) // 2
    x_off  = (target_size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    return canvas