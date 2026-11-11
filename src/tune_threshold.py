
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from satellite_dataset import SatelliteFloodDataset
from unet_model import UNet


PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
CHECKPOINT = PROJECT_DIR / "models" / "unet_augmented_best.pth"
DEVICE = torch.device("cpu")

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def main():
    dataset = SatelliteFloodDataset("val")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    model = UNet(in_channels=15, base_channels=16)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    counts = {
        t: {"tp": 0, "fp": 0, "fn": 0}
        for t in THRESHOLDS
    }

    with torch.no_grad():
        for index, (images, labels, valid_mask) in enumerate(loader, 1):
            logits = model(images)
            probabilities = torch.sigmoid(logits[:, 0])

            labels = labels.bool()
            valid = valid_mask.bool()

            for threshold in THRESHOLDS:
                pred = probabilities >= threshold
                counts[threshold]["tp"] += int(
                    (pred & labels & valid).sum()
                )
                counts[threshold]["fp"] += int(
                    (pred & ~labels & valid).sum()
                )
                counts[threshold]["fn"] += int(
                    (~pred & labels & valid).sum()
                )

            if index % 10 == 0 or index == len(loader):
                print(f"Processed {index}/{len(loader)} validation samples")

    print("\nVALIDATION THRESHOLD COMPARISON")
    print("Threshold | Water IoU | Precision | Recall | F1")

    results = []

    for threshold, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]

        iou = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        results.append((f1, threshold, iou, precision, recall))

        print(
            f"{threshold:9.2f} | {iou:9.4f} | "
            f"{precision:9.4f} | {recall:6.4f} | {f1:.4f}"
        )

    best = max(results, key=lambda x: x[0])
    print(f"\nBest validation F1 threshold: {best[1]:.2f}")
    print(f"Validation F1: {best[0]:.4f}")
    print("Use validation results to select a threshold; do not tune on test data.")


if __name__ == "__main__":
    main()
