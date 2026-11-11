
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.satellite_dataset import SatelliteFloodDataset
from src.unet_model import UNet

MODEL_PATH = PROJECT_ROOT / "models" / "unet_augmented_best.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

THRESHOLD = 0.40
BATCH_SIZE = 1


# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    device = torch.device("cpu")

    model = UNet(
        in_channels=15,
        base_channels=16
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    return model, device


# --------------------------------------------------
# 3. Calculate binary classification metrics
# --------------------------------------------------

def calculate_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn

    iou_denominator = tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn

    iou = (
        tp / iou_denominator
        if iou_denominator else 0.0
    )

    precision = (
        tp / precision_denominator
        if precision_denominator else 0.0
    )

    recall = (
        tp / recall_denominator
        if recall_denominator else 0.0
    )

    f1 = (
        2 * tp / f1_denominator
        if f1_denominator else 0.0
    )

    accuracy = (
        (tp + tn) / total
        if total else 0.0
    )

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "valid_pixels": total
    }


# --------------------------------------------------
# 4. Evaluate the complete test dataset
# --------------------------------------------------

def main():
    print("=== Full Flood Test Dataset Evaluation ===")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SatelliteFloodDataset(
        split="test",
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model, device = load_model()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    per_image_results = []
    skipped_images = 0

    with torch.inference_mode():
        for index, batch in enumerate(
            tqdm(loader, desc="Evaluating test images")
        ):
            # Expected dataset output:
            # image, labels, valid pixel mask
            images, labels, valid_masks = batch

            images = images.to(device)
            labels = labels.to(device)
            valid_masks = valid_masks.to(device).bool()

            # Model output is usually (B, 1, H, W).
            logits = model(images)

            if logits.ndim == 4 and logits.shape[1] == 1:
                logits = logits[:, 0, :, :]

            # Remove optional channel dimensions from labels/masks.
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels[:, 0, :, :]

            if valid_masks.ndim == 4 and valid_masks.shape[1] == 1:
                valid_masks = valid_masks[:, 0, :, :]

            # Validate dimensions before computing metrics.
            if not (
                logits.shape == labels.shape == valid_masks.shape
            ):
                raise ValueError(
                    f"Shape mismatch at test index {index}: "
                    f"logits={tuple(logits.shape)}, "
                    f"labels={tuple(labels.shape)}, "
                    f"valid_masks={tuple(valid_masks.shape)}"
                )

            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= THRESHOLD

            # Dataset labels are 0 (non-water) or 1 (water);
            # invalid pixels are excluded using valid_masks.
            references = labels >= 0.5

            batch_size = images.shape[0]

            for item in range(batch_size):
                valid = valid_masks[item]

                valid_count = int(valid.sum().item())

                if valid_count == 0:
                    skipped_images += 1
                    continue

                pred = predictions[item][valid]
                ref = references[item][valid]

                tp = int((pred & ref).sum().item())
                fp = int((pred & ~ref).sum().item())
                fn = int((~pred & ref).sum().item())
                tn = int((~pred & ~ref).sum().item())

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                metrics = calculate_metrics(tp, fp, fn, tn)

                predicted_water_pct = (
                    int(pred.sum().item()) / valid_count * 100
                )

                reference_water_pct = (
                    int(ref.sum().item()) / valid_count * 100
                )

                per_image_results.append({
                    "test_index": index + item,
                    "valid_pixels": valid_count,
                    "water_iou_percent": metrics["iou"] * 100,
                    "precision_percent": metrics["precision"] * 100,
                    "recall_percent": metrics["recall"] * 100,
                    "f1_percent": metrics["f1"] * 100,
                    "pixel_accuracy_percent": metrics["accuracy"] * 100,
                    "predicted_water_percent": predicted_water_pct,
                    "reference_water_percent": reference_water_pct
                })

    if not per_image_results:
        raise RuntimeError(
            "No test images contained valid pixels."
        )

    # --------------------------------------------------
    # 5. Aggregate pixel-level metrics
    # --------------------------------------------------

    aggregate = calculate_metrics(
        total_tp,
        total_fp,
        total_fn,
        total_tn
    )

    print("\n=== Aggregate Test Results ===")
    print(f"Test images evaluated: {len(per_image_results)}")
    print(f"Images skipped: {skipped_images}")
    print(f"Valid pixels evaluated: {aggregate['valid_pixels']:,}")
    print(f"Decision threshold: {THRESHOLD:.2f}")
    print(f"Water IoU:       {aggregate['iou'] * 100:.2f}%")
    print(f"Precision:       {aggregate['precision'] * 100:.2f}%")
    print(f"Recall:          {aggregate['recall'] * 100:.2f}%")
    print(f"F1-score:        {aggregate['f1'] * 100:.2f}%")
    print(f"Pixel accuracy:  {aggregate['accuracy'] * 100:.2f}%")

    # --------------------------------------------------
    # 6. Save evaluation results
    # --------------------------------------------------

    metrics_path = RESULTS_DIR / "full_test_metrics.csv"
    per_image_path = RESULTS_DIR / "full_test_per_image.csv"

    metrics_df = pd.DataFrame([
        {"metric": "Water IoU (%)", "value": aggregate["iou"] * 100},
        {"metric": "Precision (%)", "value": aggregate["precision"] * 100},
        {"metric": "Recall (%)", "value": aggregate["recall"] * 100},
        {"metric": "F1-score (%)", "value": aggregate["f1"] * 100},
        {"metric": "Pixel accuracy (%)", "value": aggregate["accuracy"] * 100},
        {"metric": "Decision threshold", "value": THRESHOLD},
        {"metric": "Images evaluated", "value": len(per_image_results)},
        {"metric": "Images skipped", "value": skipped_images},
        {"metric": "Valid pixels evaluated", "value": aggregate["valid_pixels"]}
    ])

    per_image_df = pd.DataFrame(per_image_results)

    metrics_df.to_csv(metrics_path, index=False)
    per_image_df.to_csv(per_image_path, index=False)

    print("\nSaved results:")
    print(f"Aggregate metrics: {metrics_path}")
    print(f"Per-image metrics: {per_image_path}")
    print("\nFull test evaluation completed successfully.")


if __name__ == "__main__":
    main()
