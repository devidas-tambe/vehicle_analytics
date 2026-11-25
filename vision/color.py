"""Heuristic HSV color estimation on the central region of a vehicle crop (BGR)."""
import cv2
import numpy as np


def classify_color(crop):
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 6:
        return "unknown"

    h, w = crop.shape[:2]
    y1 = int(h * 0.35)
    y2 = max(int(h * 0.80), y1 + 1)
    x1 = int(w * 0.20)
    x2 = max(int(w * 0.80), x1 + 1)

    region = crop[y1:y2, x1:x2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue, sat, val = (hsv[:, :, i] for i in range(3))

    dark = val < 55
    neutral = (sat < 48) & ~dark
    chromatic = ~dark & ~neutral

    masks = {
        "black": dark,
        "white": neutral & (val >= 195),
        "silver": neutral & (val >= 125) & (val < 195),
        "gray": neutral & (val < 125),
        "red": chromatic & ((hue < 10) | (hue >= 170)),
        "orange": chromatic & (hue >= 10) & (hue < 23) & (val >= 120),
        "brown": chromatic & (hue >= 10) & (hue < 23) & (val < 120),
        "yellow": chromatic & (hue >= 23) & (hue < 35),
        "green": chromatic & (hue >= 35) & (hue < 85),
        "blue": chromatic & (hue >= 85) & (hue < 135),
    }

    label, fraction = max(
        ((name, float(np.mean(mask))) for name, mask in masks.items()),
        key=lambda pair: pair[1],
    )
    return label if fraction >= 0.30 else "unknown"
