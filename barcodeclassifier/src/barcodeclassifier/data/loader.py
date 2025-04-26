"""
Minimal Dataset that yields a tensor image + empty target dict.
We'll add rectangle/label logic later.
"""

from pathlib import Path
from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from PIL import Image


class LabelPatchDataset(Dataset):
    """
    Each sample is a crop delivered by upstream geometry code.
    image_dir should contain .jpg/.png files only (one label per file).
    """

    def __init__(self,
                 image_dir: str | Path,
                 img_size: int = 64):
        self.image_dir = Path(image_dir)
        self.paths: List[Path] = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
        )
        self.tf = Compose([
            Resize((img_size, img_size), antialias=True),
            ToTensor(),                                  # 0–1 float32, C-first
            Normalize([0.485, 0.456, 0.406],             # ImageNet stats
                      [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img = Image.open(self.paths[idx]).convert("RGB")
        tensor = self.tf(img)
        return tensor, {}                               # target dict to be filled later
