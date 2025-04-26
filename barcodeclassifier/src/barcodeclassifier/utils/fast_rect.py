"""
fast_rect.py
------------
Light-weight helpers to spot a pale rectangular ROI (barcode *label*) and
return basic brightness / contrast stats in Lab colour space.

find_white_rect(tensor_img)  -> (bool, np.ndarray | None)
lab_stats(rgb_roi)           -> (mean_L, std_gray)
"""

from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np
import torch


def tensor_to_rgb_np(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] float32 0-1 → uint8 RGB (H,W,3)."""
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return arr


def find_white_rect(tensor_img: torch.Tensor,
                    canny_th: Tuple[int, int] = (60, 140),
                    min_rect_area: float = 0.02) -> Tuple[bool, np.ndarray | None]:
    """
    Return (found_flag, rgb_roi).  The ROI is the *visible* inside of the rectangle
    (cropped out of the original patch).  If no plausible rectangle is found,
    returns (False, None).
    """
    rgb = tensor_to_rgb_np(tensor_img)
    h, w = rgb.shape[:2]

    # 1️⃣ Edge map → contours
    edges = cv2.Canny(rgb, *canny_th)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_score = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < min_rect_area * w * h * 0.5:
            continue                                    # too small

        # ---- fit rotated rectangle (handles perspective skew) ----
        rect = cv2.minAreaRect(cnt)                     # (cx,cy),(w,h),angle
        (cx, cy), (rw, rh), _ = rect
        rw, rh = float(rw), float(rh)

        # ensure portrait orientation: long side = height
        h_side, w_side = (rh, rw) if rh >= rw else (rw, rh)
        if h_side <= w_side * 1.15:
            continue                                    # not tall enough

        # whiteness test on the *rotated* ROI
        box_pts = np.int0(cv2.boxPoints(rect))          # 4×2
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [box_pts], 0, 255, -1)
        roi = cv2.bitwise_and(rgb, rgb, mask=mask)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        if hsv[..., 2][mask == 255].mean() < 140:       # not bright
            continue

        # score = visible area; keep the best
        score = h_side * w_side
        if score > best_score:
            best_score = score
            best_rect = roi 

    if best_rect is None:
        return False, None

    x, y, bw, bh = best_rect
    return True, rgb[y:y+bh, x:x+bw]



def lab_stats(rgb_roi: np.ndarray) -> Tuple[float, float]:
    """
    Compute average lightness (L*) and grayscale std-dev using OpenCV only.
    Returns:
        mean_L  – range 0-100 (CIE-Lab lightness)
        std_gray – pixel-intensity std-dev (0-255)
    """
    # RGB → CIE-Lab (OpenCV uses 0-255 scale)
    lab = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_L = float(lab[..., 0].mean() * 100 / 255.0)   # rescale to 0-100

    gray = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2GRAY)
    std_gray = float(gray.std())
    return mean_L, std_gray
