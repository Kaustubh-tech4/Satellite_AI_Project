
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgb_flood_dataset import RGBFloodDataset
from rgb_unet_model import RGBUNet


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_DIR / "models" / "rgb_unet_v2_best.pth"
)

DEVICE = torch.device("cpu")
BATCH_SIZE = 1


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn)
        else 0.0
    )

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0

    return precision, recall, f1, iou, accuracy


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("RGB U-NET V2 - TEST EVALUATION")
    print("=" * 60)

    # Check checkpoint
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Load V2 checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain 'model_state_dict'."
        )

    # Use the threshold selected on validation data
    threshold = float(
        checkpoint.get("validation_threshold", 0.3)
    )

    # Restore the exact model architecture
    model = RGBUNet(
        in_channels=checkpoint.get("in_channels", 3),
        base_channels=checkpoint.get("base_channels", 8)
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Best epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Validation F1: {checkpoint.get('validation_f1', 'N/A')}")
    print(f"Test threshold: {threshold:.2f}")

    # Load the existing test dataset.
    # The dataset class handles the test split path internally.
    dataset = RGBFloodDataset(
        split="test",
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Test samples: {len(dataset)}")
    print("-" * 60)

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    skipped = 0

    # ========================================================
    # EVALUATE ON TEST DATA
    # ========================================================

    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):

            # RGBFloodDataset returns:
            # image, labels, valid
            images, labels, valid = batch

            images = images.to(
                DEVICE,
                dtype=torch.float32
            )

            labels = labels.to(DEVICE)
            valid = valid.to(DEVICE)

            # Ensure labels and valid masks have shape
            # (batch_size, 1, height, width).
            if labels.ndim == 3:
                labels = labels.unsqueeze(1)

            if valid.ndim == 3:
                valid = valid.unsqueeze(1)

            valid = valid.bool()

            # Skip samples without valid labeled pixels
            if not valid.any():
                skipped += 1
                continue

            # Predict water probabilities
            logits = model(images)
            probabilities = torch.sigmoid(logits)

            predictions = probabilities >= threshold
            targets = labels == 1

            # Evaluate only valid labeled pixels
            predictions = predictions[valid]
            targets = targets[valid]

            tp += int(
                (predictions & targets).sum().item()
            )

            fp += int(
                (predictions & ~targets).sum().item()
            )

            fn += int(
                (~predictions & targets).sum().item()
            )

            tn += int(
                (~predictions & ~targets).sum().item()
            )

            if index % 10 == 0 or index == len(loader):
                print(
                    f"Processed {index}/{len(loader)} samples"
                )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    precision, recall, f1, iou, accuracy = calculate_metrics(
        tp, fp, fn, tn
    )

    valid_pixels = tp + fp + fn + tn
    predicted_water = tp + fp
    reference_water = tp + fn

    print("\n" + "=" * 60)
    print("RGB U-NET V2 TEST RESULTS")
    print("=" * 60)

    print(f"Best checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Threshold:              {threshold:.2f}")
    print(f"Samples evaluated:      {len(dataset) - skipped}")
    print(f"Samples skipped:        {skipped}")
    print(f"Valid test pixels:      {valid_pixels:,}")

    print("-" * 60)

    print(f"True positives (TP):  {tp:,}")
    print(f"False positives (FP): {fp:,}")
    print(f"False negatives (FN): {fn:,}")
    print(f"True negatives (TN):  {tn:,}")

    print("-" * 60)

    print(
        f"Water precision: {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Water recall:    {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"Water F1-score:  {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"Water IoU:       {iou:.4f} "
        f"({iou * 100:.2f}%)"
    )

    print(
        f"Pixel accuracy:  {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    if valid_pixels > 0:
        print(
            f"Predicted water pixels: "
            f"{predicted_water / valid_pixels * 100:.2f}%"
        )

        print(
            f"Reference water pixels: "
            f"{reference_water / valid_pixels * 100:.2f}%"
        )

    print("=" * 60)
    print("TEST EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
