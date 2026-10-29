"""Train a fixed-budget U-Net using a geographically partitioned scene manifest."""
import argparse
from nisar_water.neural import train_unet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-scene", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metadata = train_unet(args.manifest, args.output, epochs=args.epochs, patch_size=args.patch_size,
                          patches_per_scene=args.patches_per_scene, batch_size=args.batch_size, device=args.device)
    print(metadata["model_id"], metadata["evaluation_status"], "epoch", metadata["best_epoch"])

