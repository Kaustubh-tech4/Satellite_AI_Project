
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from satellite_dataset import SatelliteFloodDataset
from unet_model import UNet


PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
CHECKPOINT = PROJECT_DIR / "models" / "unet_augmented_best.pth"

DEVICE = torch.device("cpu")
THRESHOLD = 0.40


def main():
    print("Loading augmented U-Net checkpoint...")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    dataset = SatelliteFloodDataset("test", augment=False)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(in_channels=15, base_channels=16).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tp = fp = fn = tn = 0

    print("Test samples:", len(dataset))
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Prediction threshold:", THRESHOLD)
    print("Evaluating model...")

    with torch.no_grad():
        for index, (images, labels, valid_mask) in enumerate(
            loader, start=1
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).bool()
            valid = valid_mask.to(DEVICE).bool()

            logits = model(images)
            probabilities = torch.sigmoid(logits[:, 0])
            predictions = probabilities >= THRESHOLD

            tp += int((predictions & labels & valid).sum().item())
            fp += int((predictions & ~labels & valid).sum().item())
            fn += int((~predictions & labels & valid).sum().item())
            tn += int((~predictions & ~labels & valid).sum().item())

            if index % 10 == 0 or index == len(loader):
                print(
                    f"Evaluated {index}/{len(loader)} test samples"
                )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    total_valid = tp + fp + fn + tn

    predicted_water_pct = (
        100 * (tp + fp) / max(total_valid, 1)
    )
    reference_water_pct = (
        100 * (tp + fn) / max(total_valid, 1)
    )

    print("\n========== TEST RESULTS ==========")
    print("Model: Augmented U-Net")
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Water IoU: {iou:.4f} ({iou * 100:.2f}%)")
    print(f"Water precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Water recall: {recall:.4f} ({recall * 100:.2f}%)")
    print(f"Water F1-score: {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"Pixel accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Predicted water pixels: {predicted_water_pct:.2f}%")
    print(f"Reference water pixels: {reference_water_pct:.2f}%")
    print("Valid test pixels:", total_valid)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
