"""Dataset loader for diffusion model training."""

import io

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDiffusionDataset(Dataset):
    """Dataset that loads images from parquet and creates noisy-clean pairs."""

    def __init__(self, parquet_file: str, image_size: int = 512, transform=None):
        """
        Args:
            parquet_file: Path to the parquet file
            image_size: Target image size (default: 512)
            transform: Optional torchvision transforms
        """
        self.dataset = load_dataset("parquet", data_files=parquet_file, split="train")
        self.image_size = image_size
        self.num_noise_levels = 10  # image_0 to image_9

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns (noisy_image, clean_image, timestep)."""
        # Load clean image (image_9 - least noisy)
        clean_data = self.dataset[idx]["image_9"]
        if isinstance(clean_data, bytes):
            clean_image = Image.open(io.BytesIO(clean_data)).convert("RGB")
        else:
            clean_image = clean_data.convert("RGB")

        # Randomly select a noise level (0 = most noisy, 9 = least noisy)
        # We'll use this to select which image to use as noisy input
        noise_level = np.random.randint(0, 9)  # 0-8, excluding 9 (clean)

        # Load corresponding noisy image
        noisy_data = self.dataset[idx][f"image_{noise_level}"]
        if isinstance(noisy_data, bytes):
            noisy_image = Image.open(io.BytesIO(noisy_data)).convert("RGB")
        else:
            noisy_image = noisy_data.convert("RGB")

        # Apply transforms
        clean_tensor = self.transform(clean_image)
        noisy_tensor = self.transform(noisy_image)

        # Timestep: 0 = most noisy, 1 = clean
        # Map noise_level (0-8) to timestep (0.0-0.9)
        timestep = noise_level / 9.0

        return {
            "noisy": noisy_tensor,
            "clean": clean_tensor,
            "timestep": torch.tensor(timestep, dtype=torch.float32),
        }

