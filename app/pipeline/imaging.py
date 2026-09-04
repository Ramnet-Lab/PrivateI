"""Page image normalisation.

Deskew and contrast only.  Nothing here may alter content: no denoise that
eats faint pencil, no binarisation that drops a light annotation.  The source
file in 01_raw remains the authority; these images exist to make OCR and the
VLM work, and any doubt is resolved by looking at the original.
"""
from __future__ import annotations

import cv2
import numpy as np

MAX_SKEW_DEG = 15.0     # beyond this the estimate is noise, not skew
MIN_DIM = 400           # do not try to deskew a thumbnail


def _estimate_skew(gray: np.ndarray) -> float:
    """Angle in degrees from the minimum-area rectangle around dark pixels."""
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    return float(angle)


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if min(h, w) < MIN_DIM:
        return image, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    angle = _estimate_skew(gray)
    if abs(angle) < 0.2 or abs(angle) > MAX_SKEW_DEG:
        return image, 0.0
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def enhance(image: np.ndarray) -> np.ndarray:
    """CLAHE on the luminance channel.

    Local contrast, so faint pencil on one part of a page lifts without
    blowing out a dark photocopied block elsewhere.  Colour is preserved -
    ink colour sometimes carries meaning in an investigative note.
    """
    if image.ndim == 2:
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def normalize(image: np.ndarray) -> tuple[np.ndarray, dict]:
    out, angle = deskew(image)
    out = enhance(out)
    h, w = out.shape[:2]
    return out, {"deskew_deg": round(angle, 3), "width": int(w), "height": int(h)}
