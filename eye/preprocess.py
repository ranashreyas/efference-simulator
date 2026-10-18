import cv2

MODEL_SIZE = (128, 96)      # (width, height) the forward model sees
LOG_SIZE = (320, 180)       # full frame, stored wide enough to cut virtual saccades offline


def to_gray_small(img, size):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
