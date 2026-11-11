
from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader

from unified_model import UnifiedWaterUNet
from unified_dataset import UnifiedSatelliteDataset


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "unified_water_unet_best.pth"
RESULTS_DIR = PROJECT_DIR / "results"

THRESHOLD = 0.40
BASE_CHANNELS = 16
BATCH_SIZE = 1

MODALITIES = ("sar", "optical", "fused")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def extract_state_dict(checkpoint):
    """Support common checkpoint formats."""

    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "model_state",
    ):
        if key in checkpoint:
            value = checkpoint[key]

            if isinstance(value, torch.nn.Module):
                return value.state_dict()

            if isinstance(value, dict):
                # Some checkpoints wrap the state dict again.
                try:
                    return extract_state_dict(value)
                except ValueError:
                    pass

    # Checkpoint may directly contain the model's parameters.
    if checkpoint and all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise ValueError(
        "Could not find model weights in the checkpoint. "
        "Inspect the checkpoint format before proceeding."
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "valid_pixels": total,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IoU": iou,
        "pixel_accuracy": accuracy,
    }


# ============================================================
# EVALUATION
# ============================================================

def main():
    print("=" * 65)
    print("UNIFIED SATELLITE MODEL - TEST EVALUATION")
    print("=" * 65)

    print(f"Project: {PROJECT_DIR}")
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Threshold: {THRESHOLD}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the saved model without retraining.
    model = UnifiedWaterUNet(base_channels=BASE_CHANNELS)

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

    state_dict = extract_state_dict(checkpoint)

    # Accommodate checkpoints saved by DataParallel.
    if state_dict and all(
        key.startswith("module.")
        for key in state_dict
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    print("\nCheckpoint loaded successfully.")

    all_results = []

    for modality in MODALITIES:
        print("\n" + "-" * 65)
        print(f"Evaluating modality: {modality.upper()}")
        print("-" * 65)

        dataset = UnifiedSatelliteDataset(
            split="test",
            modality=modality,
            augment=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        tp = fp = fn = tn = 0

        with torch.inference_mode():
            for index, batch in enumerate(loader, start=1):
                images, labels, valid, _ = batch

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                valid = valid.to(DEVICE).bool()

                # Use the correct input adapter for this modality.
                logits = model(images, modality)
                probabilities = torch.sigmoid(logits)

                predictions = probabilities >= THRESHOLD
                actual = labels >= 0.5

                # Ignore invalid labels and invalid imagery.
                mask = valid & torch.isfinite(labels)

                tp += int((predictions & actual & mask).sum().item())
                fp += int((predictions & ~actual & mask).sum().item())
                fn += int((~predictions & actual & mask).sum().item())
                tn += int((~predictions & ~actual & mask).sum().item())

                if index % 10 == 0 or index == len(dataset):
                    print(
                        f"Processed {index}/{len(dataset)} samples",
                        flush=True,
                    )

        metrics = calculate_metrics(tp, fp, fn, tn)
        metrics["modality"] = modality

        all_results.append(metrics)

        print(f"\nResults for {modality.upper()}:")
        print(f"Valid pixels:   {metrics['valid_pixels']:,}")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1-score:       {metrics['F1']:.4f}")
        print(f"IoU:            {metrics['IoU']:.4f}")
        print(f"Pixel accuracy: {metrics['pixel_accuracy']:.4f}")

    # Save metrics to CSV for reporting and comparison.
    output_file = RESULTS_DIR / "unified_test_metrics.csv"

    fieldnames = [
        "modality",
        "valid_pixels",
        "TP",
        "FP",
        "FN",
        "TN",
        "precision",
        "recall",
        "F1",
        "IoU",
        "pixel_accuracy",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(
        f"{'Modality':<12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'IoU':>12}"
        f"{'Accuracy':>12}"
    )

    for result in all_results:
        print(
            f"{result['modality']:<12}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['F1']:>12.4f}"
            f"{result['IoU']:>12.4f}"
            f"{result['pixel_accuracy']:>12.4f}"
        )

    print(f"\nMetrics saved to:\n{output_file}")
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
