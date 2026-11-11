
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgb_flood_dataset import RGBFloodDataset
from rgb_unet_model import RGBUNet


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
CHECKPOINT_PATH = PROJECT_DIR / "models" / "rgb_unet_best.pth"

DEVICE = torch.device("cpu")

# Fixed using validation results; do not tune on test data.
THRESHOLD = 0.3


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(tp, fp, fn, tn):
    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return precision, recall, f1, iou, accuracy


# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------

def main():
    print("=" * 60)
    print("RGB U-NET FINAL TEST EVALUATION")
    print("=" * 60)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model = RGBUNet(
        in_channels=checkpoint.get("in_channels", 3),
        base_channels=checkpoint.get("base_channels", 8),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = RGBFloodDataset(
        split="test",
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    tp = fp = fn = tn = 0
    skipped = 0
    evaluated_samples = 0
    valid_pixels_total = 0

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Test samples: {len(dataset)}")
    print("\nEvaluating...")

    with torch.no_grad():

        for index, (images, labels, valid) in enumerate(loader):

            images = images.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)
            valid = valid.unsqueeze(1).to(DEVICE).bool()

            # Ignore samples that contain no valid labels.
            if not valid.any().item():
                skipped += 1
                continue

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            predictions = probabilities >= THRESHOLD
            targets = labels >= 0.5

            tp += (predictions & targets & valid).sum().item()
            fp += (predictions & ~targets & valid).sum().item()
            fn += (~predictions & targets & valid).sum().item()
            tn += (~predictions & ~targets & valid).sum().item()

            valid_pixels_total += int(valid.sum().item())
            evaluated_samples += 1

            if (index + 1) % 15 == 0:
                print(
                    f"Processed {index + 1}/{len(dataset)} samples"
                )

    if valid_pixels_total == 0:
        raise RuntimeError(
            "No valid test pixels were found."
        )

    precision, recall, f1, iou, accuracy = calculate_metrics(
        tp, fp, fn, tn
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Samples evaluated: {evaluated_samples}")
    print(f"Samples skipped: {skipped}")
    print(f"Valid pixels evaluated: {valid_pixels_total:,}")

    print("\nConfusion counts")
    print(f"True positives:  {tp:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True negatives:  {tn:,}")

    print("\nWater segmentation metrics")
    print(f"Water precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Water recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"Water F1-score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"Water IoU:       {iou:.4f} ({iou * 100:.2f}%)")
    print(f"Pixel accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nTest evaluation completed.")
    print("The saved model checkpoint was not modified.")


if __name__ == "__main__":
    main()
