
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgb_flood_dataset import RGBFloodDataset
from rgb_unet_model import RGBUNet


PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
CHECKPOINT = PROJECT_DIR / "models" / "rgb_unet_best.pth"

DEVICE = torch.device("cpu")
THRESHOLDS = [i / 10 for i in range(1, 10)]


def calculate_metrics(tp, fp, fn, tn):
    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return precision, recall, f1, iou, accuracy


def main():
    print("=" * 60)
    print("RGB U-NET VALIDATION THRESHOLD EVALUATION")
    print("=" * 60)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    dataset = RGBFloodDataset(split="val", augment=False)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model = RGBUNet(
        in_channels=checkpoint.get("in_channels", 3),
        base_channels=checkpoint.get("base_channels", 8),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Saved validation F1 at threshold 0.5: "
        f"{checkpoint['validation_f1']:.4f}"
    )

    # One confusion matrix per threshold.
    counts = {
        threshold: [0, 0, 0, 0]  # TP, FP, FN, TN
        for threshold in THRESHOLDS
    }

    skipped = 0
    valid_pixel_total = 0

    with torch.no_grad():
        for index, (images, labels, valid) in enumerate(loader):

            images = images.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)
            valid = valid.unsqueeze(1).to(DEVICE).bool()

            if not valid.any().item():
                skipped += 1
                continue

            probabilities = torch.sigmoid(model(images))

            # Evaluate only pixels with valid labels.
            probs = probabilities[valid]
            targets = labels[valid] >= 0.5

            valid_pixel_total += targets.numel()

            for threshold in THRESHOLDS:
                predictions = probs >= threshold

                tp = (predictions & targets).sum().item()
                fp = (predictions & ~targets).sum().item()
                fn = (~predictions & targets).sum().item()
                tn = (~predictions & ~targets).sum().item()

                current = counts[threshold]
                current[0] += tp
                current[1] += fp
                current[2] += fn
                current[3] += tn

            if (index + 1) % 20 == 0:
                print(
                    f"Processed {index + 1}/{len(loader)} "
                    "validation samples"
                )

    print(f"\nValid pixels evaluated: {valid_pixel_total:,}")
    print(f"Samples skipped: {skipped}")

    best_threshold = None
    best_f1 = -1.0

    print("\nThreshold comparison")
    print("-" * 85)
    print(
        f"{'Threshold':>9} {'Precision':>10} {'Recall':>10} "
        f"{'F1':>10} {'IoU':>10} {'Accuracy':>10}"
    )
    print("-" * 85)

    for threshold in THRESHOLDS:
        tp, fp, fn, tn = counts[threshold]

        precision, recall, f1, iou, accuracy = (
            calculate_metrics(tp, fp, fn, tn)
        )

        print(
            f"{threshold:9.1f} "
            f"{precision:10.4f} "
            f"{recall:10.4f} "
            f"{f1:10.4f} "
            f"{iou:10.4f} "
            f"{accuracy:10.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("-" * 85)
    print(f"Best validation threshold: {best_threshold:.1f}")
    print(f"Best validation water F1: {best_f1:.4f}")
    print(
        "\nUse this threshold for subsequent evaluation. "
        "Do not use the test set to choose the threshold."
    )


if __name__ == "__main__":
    main()
