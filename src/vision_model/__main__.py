"""Main entry point for vision_model module."""

import argparse
import sys

from . import inference, train


def main():
    """Main entry point when running as module: python -m vision_model."""
    parser = argparse.ArgumentParser(description="Vision Model - Bernoulli Diffusion")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training command
    train_parser = subparsers.add_parser("train", help="Train the diffusion model")
    train_parser.add_argument(
        "--parquet-file",
        type=str,
        default="dataset.parquet",
        help="Path to the parquet dataset file",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size (assumes square images)",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    train_parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )

    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_subparsers = inference_parser.add_subparsers(
        dest="mode", help="Inference mode"
    )

    # Dataset inference
    dataset_parser = inference_subparsers.add_parser(
        "dataset", help="Inference on dataset samples"
    )
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
    image_parser = inference_subparsers.add_parser(
        "image", help="Inference on single image file"
    )
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

    if args.command == "train":
        train.train(
            parquet_file=args.parquet_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            num_workers=args.num_workers,
            save_every=args.save_every,
        )
    elif args.command == "inference":
        if args.mode == "dataset":
            inference.inference_from_dataset(
                checkpoint_path=args.checkpoint,
                parquet_file=args.parquet_file,
                num_samples=args.num_samples,
                image_size=args.image_size,
                output_dir=args.output_dir,
            )
        elif args.mode == "image":
            inference.inference_from_image_file(
                checkpoint_path=args.checkpoint,
                image_path=args.image_path,
                timestep=args.timestep,
                output_path=args.output_path,
                image_size=args.image_size,
            )
        else:
            inference_parser.print_help()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

