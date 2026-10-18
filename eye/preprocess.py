import cv2

MODEL_SIZE = (128, 96)      # (width, height) the forward model sees
LOG_SIZE = (320, 180)       # full frame, stored wide enough to cut virtual saccades offline
PREVIEW_SIZE = (640, 360)   # color video for a human to watch while labeling


def to_gray_small(img, size):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def static_model_input(log_gray):
    """Model input for a fixed camera, from a LOG_SIZE grayscale frame: the centered 4:3 crop at
    full height (the middle 75% of the width), resized to MODEL_SIZE. Training and the live loop
    both go through this, so the model always sees exactly what it was trained on."""
    h, w = log_gray.shape
    cw = h * MODEL_SIZE[0] // MODEL_SIZE[1]
    x0 = (w - cw) // 2
    return cv2.resize(log_gray[:, x0:x0 + cw], MODEL_SIZE, interpolation=cv2.INTER_AREA)
