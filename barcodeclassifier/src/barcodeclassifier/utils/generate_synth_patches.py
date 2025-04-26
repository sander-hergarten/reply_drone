"""
generate_synth_patches.py
-------------------------
Creates `data/patches/{pos,neg}` with PNG files ready for TinyCNN training.

Run, for example:
    rye run python -m cratescout_vision.utils.generate_synth_patches --n_pos 12000 --n_neg 20000
"""

from __future__ import annotations
import argparse, random
from pathlib import Path
import cv2
import numpy as np
from tqdm import trange

PATCH = 32

def make_background() -> np.ndarray:
    """
    Return 128×128 RGB canvas with noise plus random shapes to mimic crates,
    logos, screws, etc.
    """
    canvas = np.random.randint(0, 256, (128, 128, 3), np.uint8)

    # add 3–6 random coloured rectangles/circles
    for _ in range(random.randint(3, 6)):
        col = np.random.randint(30, 200, 3).tolist()
        if random.random() < 0.5:
            # rectangle
            x0, y0 = random.randint(0, 96), random.randint(0, 96)
            w, h = random.randint(10, 32), random.randint(10, 32)
            cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), col, -1)
        else:
            # circle
            cx, cy = random.randint(20, 108), random.randint(20, 108)
            r = random.randint(5, 16)
            cv2.circle(canvas, (cx, cy), r, col, -1)
    return canvas   

def add_label(canvas: np.ndarray) -> np.ndarray:
    """
    Build a portrait sticker (off-white frame + bottom barcode zone),
    rotate, perspective-warp, and paste it on a 128×128 background
    *without* any pad leakage, then down-sample to 32×32.
    """
    # 0 ─ portrait sticker bitmap
    w = random.randint(20, 30)
    h = random.randint(int(w * 1.4), int(w * 1.9))
    bg_val = random.randint(225, 240)          # off-white
    sticker = np.full((h, w, 3), bg_val, np.uint8)
    mask    = np.ones((h, w),  np.uint8)       # binary mask (1 = sticker)

    # barcode zone
    bz_w, bz_h = int(w * 0.8), int(h * 0.4)
    x0, y0 = (w - bz_w) // 2, h - bz_h
    bar_col = 0 if random.random() < .8 else 35
    step = random.randint(2, 4)
    for x in range(x0, x0 + bz_w, step * 2):
        cv2.rectangle(sticker, (x, y0),
                      (min(x + step, x0 + bz_w), y0 + bz_h),
                      (bar_col, bar_col, bar_col), -1)

    # 1 ─ embed in square canvas (diag) so rotation never clips
    diag = int(np.ceil(np.hypot(w, h)))
    sq_img = np.full((diag, diag, 3), 255, np.uint8)
    sq_msk = np.zeros((diag, diag),    np.uint8)
    ox, oy = (diag - w) // 2, (diag - h) // 2
    sq_img[oy:oy + h, ox:ox + w] = sticker
    sq_msk[oy:oy + h, ox:ox + w] = mask

    # 2 ─ rotation
    ang = random.uniform(-30, 30)
    M_rot = cv2.getRotationMatrix2D((diag / 2, diag / 2), ang, 1.0)
    rot_img = cv2.warpAffine(sq_img, M_rot, (diag, diag), borderValue=255)
    rot_msk = cv2.warpAffine(sq_msk, M_rot, (diag, diag), borderValue=0)

    # 3 ─ pad, then perspective-warp img + mask together
    pad = diag // 2
    big = np.full((diag + 2*pad, diag + 2*pad, 3), 255, np.uint8)
    big_m = np.zeros((diag + 2*pad, diag + 2*pad),   np.uint8)
    big[pad:pad+diag, pad:pad+diag] = rot_img
    big_m[pad:pad+diag, pad:pad+diag] = rot_msk

    src = np.float32([[pad, pad],
                      [pad+diag, pad],
                      [pad+diag, pad+diag],
                      [pad, pad+diag]])
    jit = 0.2
    dst = src + np.float32([[random.uniform(-jit, jit)*diag,
                              random.uniform(-jit, jit)*diag] for _ in range(4)])
    H = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(big,  H, big.shape[:2][::-1], borderValue=255)
    warp_m = cv2.warpPerspective(big_m, H, big.shape[:2][::-1], borderValue=0)

    # 4 ─ crop tight using the mask (no thresholding!)
    ys, xs = np.where(warp_m > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    sticker_final = warp[y_min:y_max+1, x_min:x_max+1]
    mask_final    = warp_m[y_min:y_max+1, x_min:x_max+1]

    # 5 ─ paste on 128×128 background *through* the mask
    Hc, Wc = canvas.shape[:2]
    h, w = sticker_final.shape[:2]
    xo = random.randint(-int(w*.3), Wc - int(w*.7))
    yo = random.randint(-int(h*.3), Hc - int(h*.7))
    for y in range(h):
        yy = yo + y
        if 0 <= yy < Hc:
            for x in range(w):
                if mask_final[y, x]:          # paste only sticker pixels
                    xx = xo + x
                    if 0 <= xx < Wc:
                        canvas[yy, xx] = sticker_final[y, x]

    # 6 ─ blur + noise
    if random.random() < .2:
        canvas[:] = cv2.GaussianBlur(canvas, random.choice([(3,3),(5,5)]), 0)
    if random.random() < .3:
        noise = np.random.normal(0, 10, canvas.shape).astype(np.int16)
        canvas[:] = np.clip(canvas.astype(np.int16)+noise, 0, 255).astype(np.uint8)

    # 7 ─ down-sample to 32×32 with anti-aliasing
    return canvas




def save_patch(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

def main(n_pos: int, n_neg: int):
    root = Path("data/patches")
    pos_dir = root / "pos"; neg_dir = root / "neg"
    for i in trange(n_pos, desc="positives"):
        img = make_background()
        img = add_label(img)
        save_patch(img, pos_dir / f"{i:06}.png")
    for i in trange(n_neg, desc="negatives"):
        img = make_background()
        save_patch(img, neg_dir / f"{i:06}.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pos", type=int, default=12000)
    ap.add_argument("--n_neg", type=int, default=20000)
    args = ap.parse_args()
    main(args.n_pos, args.n_neg)
