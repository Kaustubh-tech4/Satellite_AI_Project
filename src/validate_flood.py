
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sen1floods11_v1.1"
    / "data"
)

SAMPLE_ID = "Bolivia_103757"
THRESHOLD = 0.40

PREDICTION_PATH = RESULTS_DIR / "new_flood_probabilities.npy"
LABEL_PATH = DATA_DIR / "LabelHand" / f"{SAMPLE_ID}_LabelHand.tif"
S1_PATH = DATA_DIR / "S1GRDHand" / f"{SAMPLE_ID}_S1Hand.tif"
S2_PATH = DATA_DIR / "S2L1CHand" / f"{SAMPLE_ID}_S2Hand.tif"


def main():
    print("=== Flood Model Validation ===")

    # Check required files
    for path in (PREDICTION_PATH, LABEL_PATH, S1_PATH, S2_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load saved model probabilities
    probabilities = np.load(PREDICTION_PATH)

    # Load ground-truth label
    with rasterio.open(LABEL_PATH) as src:
        labels = src.read(1)

    # Match the label convention used by the dataset:
    # -1 = invalid, 0 = non-water, 1 = water
    valid = (labels == 0) | (labels == 1)

    # Exclude invalid pixels from the Sentinel-1/Sentinel-2 inputs
    with rasterio.open(S1_PATH) as src:
        s1 = src.read().astype(np.float32)

    with rasterio.open(S2_PATH) as src:
        s2 = src.read().astype(np.float32)

    s1_valid = np.isfinite(s1).all(axis=0)
    s1_valid &= ~np.all(s1 == 0, axis=0)

    s2_valid = np.isfinite(s2).all(axis=0)
    s2_valid &= ~np.all(s2 == 0, axis=0)

    valid &= s1_valid & s2_valid

    if probabilities.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: prediction {probabilities.shape}, "
            f"label {labels.shape}"
        )

    if not valid.any():
        raise ValueError("No valid pixels available for evaluation.")

    # Binary prediction and reference
    predicted = probabilities >= THRESHOLD
    reference = labels == 1

    tp = int(np.sum(predicted & reference & valid))
    fp = int(np.sum(predicted & ~reference & valid))
    fn = int(np.sum(~predicted & reference & valid))
    tn = int(np.sum(~predicted & ~reference & valid))

    denominator_iou = tp + fp + fn
    denominator_precision = tp + fp
    denominator_recall = tp + fn

    iou = tp / denominator_iou if denominator_iou else 0.0
    precision = (
        tp / denominator_precision if denominator_precision else 0.0
    )
    recall = tp / denominator_recall if denominator_recall else 0.0

    f1_denominator = 2 * tp + fp + fn
    f1 = (
        2 * tp / f1_denominator if f1_denominator else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn else 0.0
    )

    print(f"\nSample: {SAMPLE_ID}")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Valid pixels evaluated: {valid.sum():,}")
    print(f"True positives: {tp:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True negatives: {tn:,}")

    print("\n=== Evaluation Metrics ===")
    print(f"Water IoU:       {iou * 100:.2f}%")
    print(f"Precision:       {precision * 100:.2f}%")
    print(f"Recall:          {recall * 100:.2f}%")
    print(f"F1-score:        {f1 * 100:.2f}%")
    print(f"Pixel accuracy:  {accuracy * 100:.2f}%")

    # Save a comparison visualization
    prediction_display = np.where(valid, predicted.astype(float), np.nan)
    reference_display = np.where(valid, reference.astype(float), np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(
        prediction_display, cmap="Blues", vmin=0, vmax=1
    )
    axes[0].set_title("Model Prediction")

    axes[1].imshow(
        reference_display, cmap="Blues", vmin=0, vmax=1
    )
    axes[1].set_title("Ground-Truth Label")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = RESULTS_DIR / "flood_validation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nComparison image saved: {output_path}")
    print("\nValidation completed.")


if __name__ == "__main__":
    main()
