
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from satellite_dataset import SatelliteFloodDataset
from unet_model import UNet


PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
MODEL_DIR = PROJECT_DIR / "models"
DEVICE = torch.device("cpu")
THRESHOLD = 0.50


def evaluate_checkpoint(checkpoint_path, loader):
    model = UNet(in_channels=15, base_channels=16).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tp = fp = fn = tn = 0

    with torch.no_grad():
        for images, labels, valid_mask in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).bool()
            valid = valid_mask.to(DEVICE).bool()

            logits = model(images)
            predictions = torch.sigmoid(logits[:, 0]) >= THRESHOLD

            tp += int((predictions & labels & valid).sum())
            fp += int((predictions & ~labels & valid).sum())
            fn += int((~predictions & labels & valid).sum())
            tn += int((~predictions & ~labels & valid).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "epoch": checkpoint.get("epoch", "unknown"),
        "accuracy": accuracy,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    models = {
        "Original U-Net": MODEL_DIR / "unet_best.pth",
        "Augmented U-Net": MODEL_DIR / "unet_augmented_best.pth",
    }

    for name, path in models.items():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    dataset = SatelliteFloodDataset("val", augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    print("Validation samples:", len(dataset))
    print("Threshold:", THRESHOLD)
    print("\nEvaluating both models...")

    results = {}

    for name, path in models.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_checkpoint(path, loader)

    print("\n========== VALIDATION COMPARISON ==========")
    print(
        f"{'Model':18} {'Epoch':>6} {'Accuracy':>10} "
        f"{'IoU':>9} {'Precision':>11} {'Recall':>9} {'F1':>9}"
    )

    for name, result in results.items():
        print(
            f"{name:18} {str(result['epoch']):>6} "
            f"{result['accuracy']:10.4f} "
            f"{result['iou']:9.4f} "
            f"{result['precision']:11.4f} "
            f"{result['recall']:9.4f} "
            f"{result['f1']:9.4f}"
        )

    print("\nComparison complete.")
    print("Both models were evaluated on the same validation samples.")


if __name__ == "__main__":
    main()
