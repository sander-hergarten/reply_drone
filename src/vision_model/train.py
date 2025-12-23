"""Training script for Bernoulli diffusion model."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ImageDiffusionDataset
from .model import UNet
from .trainer import BernoulliDiffusionModel


def train(
    parquet_file: str,
    output_dir: str = "./checkpoints",
    batch_size: int = 4,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    image_size: int = 512,
    num_workers: int = 4,
    save_every: int = 10,
):
    """Train the diffusion model."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset and dataloader
    print("Loading dataset...")
    dataset = ImageDiffusionDataset(parquet_file, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {len(dataloader)}")

    # Model
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=128,
        channel_multipliers=(1, 2, 4, 8),
    )

    trainer = BernoulliDiffusionModel(
        model=model,
        device=device,
        learning_rate=learning_rate,
    )

    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            result = trainer.train_step(batch)
            epoch_losses.append(result["loss"])
            pbar.set_postfix({"loss": f"{result['loss']:.4f}"})

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            trainer.save_checkpoint(checkpoint_path, epoch + 1, avg_loss)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training complete!")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train Bernoulli Diffusion Model")
    parser.add_argument(
        "--parquet-file",
        type=str,
        default="dataset.parquet",
        help="Path to the parquet dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size (assumes square images)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )

    args = parser.parse_args()

    train(
        parquet_file=args.parquet_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        num_workers=args.num_workers,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()

