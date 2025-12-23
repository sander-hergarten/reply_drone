"""Inference script for the Bernoulli Diffusion Model."""

import argparse
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

from .dataset import ImageDiffusionDataset
from .model import UNet


def load_model(
    checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Load trained model from checkpoint."""
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=128,
        channel_multipliers=(1, 2, 4, 8),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs, loss: {checkpoint['loss']:.4f}")

    return model


def denoise_image(
    model: torch.nn.Module,
    noisy_image: torch.Tensor,
    timestep: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Denoise a single image."""
    model.eval()
    with torch.no_grad():
        noisy_image = noisy_image.unsqueeze(0).to(device)
        timestep_tensor = torch.tensor([timestep], dtype=torch.float32).to(device)

        pred_clean = model(noisy_image, timestep_tensor)
        pred_clean = torch.clamp(pred_clean, 0, 1)

        return pred_clean.squeeze(0).cpu()


def inference_from_dataset(
    checkpoint_path: str,
    parquet_file: str,
    num_samples: int = 5,
    image_size: int = 512,
    output_dir: str = "./inference_outputs",
):
    """Run inference on samples from the dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = load_model(checkpoint_path, device)

    # Load dataset
    dataset = ImageDiffusionDataset(parquet_file, image_size=image_size)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Sample random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    print(f"\nRunning inference on {num_samples} samples...")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        noisy = sample["noisy"]
        clean = sample["clean"]
        timestep = sample["timestep"].item()

        # Denoise
        pred_clean = denoise_image(model, noisy, timestep, device)

        # Convert to numpy for visualization
        noisy_np = noisy.permute(1, 2, 0).numpy()
        clean_np = clean.permute(1, 2, 0).numpy()
        pred_np = pred_clean.permute(1, 2, 0).numpy()

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(noisy_np)
        axes[0].set_title(f"Noisy Input (t={timestep:.2f})")
        axes[0].axis("off")

        axes[1].imshow(pred_np)
        axes[1].set_title("Predicted Clean")
        axes[1].axis("off")

        axes[2].imshow(clean_np)
        axes[2].set_title("Ground Truth Clean")
        axes[2].axis("off")

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"inference_{i+1}_idx_{idx}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved inference result {i+1} to {output_path}")

    print(f"\nInference complete! Results saved to {output_dir}")


def inference_from_image_file(
    checkpoint_path: str,
    image_path: str,
    timestep: float = 0.0,
    output_path: str = "./denoised_output.png",
    image_size: int = 512,
):
    """Denoise a single image file."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = load_model(checkpoint_path, device)

    # Load and preprocess image
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    # Denoise
    print(f"Denoising image with timestep {timestep}...")
    pred_clean = denoise_image(model, image_tensor, timestep, device)

    # Convert to PIL and save
    pred_np = pred_clean.permute(1, 2, 0).numpy()
    pred_np = (pred_np * 255).astype(np.uint8)
    pred_image = Image.fromarray(pred_np)

    pred_image.save(output_path)
    print(f"Denoised image saved to {output_path}")


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(description="Run inference with trained diffusion model")
    subparsers = parser.add_subparsers(dest="mode", help="Inference mode")

    # Dataset inference
    dataset_parser = subparsers.add_parser("dataset", help="Inference on dataset samples")
    dataset_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    dataset_parser.add_argument(
        "--parquet-file", type=str, default="dataset.parquet", help="Path to parquet file"
    )
    dataset_parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of samples to process"
    )
    dataset_parser.add_argument("--image-size", type=int, default=512, help="Image size")
    dataset_parser.add_argument(
        "--output-dir", type=str, default="./inference_outputs", help="Output directory"
    )

    # Single image inference
    image_parser = subparsers.add_parser("image", help="Inference on single image file")
    image_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    image_parser.add_argument(
        "--image-path", type=str, required=True, help="Path to input image"
    )
    image_parser.add_argument(
        "--timestep",
        type=float,
        default=0.0,
        help="Timestep (0.0 = most noisy, 1.0 = clean)",
    )
    image_parser.add_argument(
        "--output-path", type=str, default="./denoised_output.png", help="Output image path"
    )
    image_parser.add_argument("--image-size", type=int, default=512, help="Image size")

    args = parser.parse_args()

    if args.mode == "dataset":
        inference_from_dataset(
            checkpoint_path=args.checkpoint,
            parquet_file=args.parquet_file,
            num_samples=args.num_samples,
            image_size=args.image_size,
            output_dir=args.output_dir,
        )
    elif args.mode == "image":
        inference_from_image_file(
            checkpoint_path=args.checkpoint,
            image_path=args.image_path,
            timestep=args.timestep,
            output_path=args.output_path,
            image_size=args.image_size,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

