"""Face detection with OpenCV's bundled Haar cascade (no download, CPU only).
Upgrade path: MediaPipe Face Detection or YuNet (cv2.FaceDetectorYN) for better recall."""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_faces(img: Image.Image, min_size_ratio: float = 0.06) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    m = int(min(gray.shape) * min_size_ratio)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(m, m))
    return [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in faces]
