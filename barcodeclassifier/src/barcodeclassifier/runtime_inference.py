"""
runtime_inference.py
--------------------
Expose `classify_all(all_cls: AllBarcodeClassifiers)
              -> AllBarcodeClassifierConfidences`

Bridges the vision module and the RL agent.
"""

from __future__ import annotations
from typing import Dict

import torch
from torchvision.transforms import Resize, ToTensor, Normalize, Compose

from barcodeclassifier.models.cnn import TinyCNN
from barcodeclassifier.utils.fast_rect import find_white_rect, lab_stats
from barcodeclassifier.APIs import (       
    AllBarcodeClassifiers,
    AllBarcodeClassifierConfidences,
    SingleBarcodeClassifier,
    SingleBarcodeClassifierConfidence,
)

import yaml, pathlib

cfg = yaml.safe_load(pathlib.Path("conf_formula.yaml").read_text())
w_presence = cfg["w_presence"]

# --------- load model once ---------
_MODEL = TinyCNN.load("models/tinycnn_labels.pt")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL.to(_DEVICE)

# --------- transforms --------------
_TF_32 = Compose([
    Resize((32, 32), antialias=True),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --------- confidence helpers ------
import math


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _score_single(sbc: SingleBarcodeClassifier) -> float:
    # tensor expected in torchvision Image (C,H,W) float32 0-1 or uint8 0-255
    img = sbc.image
    if img.dtype == torch.uint8:
        img = img.float() / 255.0

    # 1) Geometric pre-filter
    rect_ok, roi = find_white_rect(img)
    p_geom = 1.0 if rect_ok else 0.0

    if rect_ok:
        mean_L, std_g = lab_stats(roi)
        p_bright = _sigmoid((mean_L - 55) / 10)
        p_contr = _sigmoid((std_g - 8) / 4)
    else:
        p_bright = p_contr = 0.0

    # 2) CNN presence
    with torch.no_grad():
        patch = _TF_32(roi if rect_ok else (img * 255).byte()).unsqueeze(0).to(_DEVICE)
        p_net = float(_MODEL(patch).cpu().item())
    roi_h, roi_w = roi.shape[:2]
    area_frac = (roi_h * roi_w) / (img.shape[1] * img.shape[2])
    # 3) Fuse
    raw = 2.0 * p_net + 1.2 * p_geom + p_bright + p_contr + 1.5 * area_frac
    
    return _sigmoid(raw)


# --------- public API --------------
def classify_all(all_cls: AllBarcodeClassifiers) -> AllBarcodeClassifierConfidences:
    out: Dict[int, SingleBarcodeClassifierConfidence] = {}
    for bid, sbc in all_cls.id_to_barcode_map.items():
        conf = _score_single(sbc)
        out[bid] = SingleBarcodeClassifierConfidence(confidence_score=conf, id=bid)
    return AllBarcodeClassifierConfidences(id_to_confidence_map=out)

