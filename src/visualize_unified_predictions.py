
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# PATHS AND SETTINGS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from unified_dataset import UnifiedSatelliteDataset
from unified_model import UnifiedWaterUNet

MODEL_PATH = PROJECT_DIR / "models" / "unified_water_unet_best.pth"

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "prediction_visualizations_threshold_060"
)

THRESHOLD = 0.60
NUM_SAMPLES = 5

DEVICE = torch.device("cpu")

# Test samples to visualize, if present in the test split.
PREFERRED_SAMPLES = [
    "Ghana_313799",
    "India_772630",
    "Paraguay_232281",
    "Spain_7387658",
    "USA_758178",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = UnifiedWaterUNet(base_channels=16)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Support either a raw state dictionary or a checkpoint
    # containing the model weights.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unrecognized checkpoint format.")

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Prediction threshold: {THRESHOLD}")

    return model


# ------------------------------------------------------------
# DISPLAY HELPERS
# ------------------------------------------------------------

def get_rgb(image):
    """
    Fused channel order:
      0-1  : Sentinel-1 SAR
      2-14 : Sentinel-2 optical bands

    Optical order assumed by this project:
      B1, B2, B3, B4, B5, B6, B7,
      B8, B8A, B9, B10, B11, B12

    True-color RGB = B4, B3, B2.
    In the fused input, these are indices 5, 4, 3.
    """
    if image.shape[0] != 15:
        raise ValueError(
            f"Expected 15 fused channels; got {image.shape[0]}"
        )

    rgb = image[[5, 4, 3], :, :].transpose(1, 2, 0)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)

    # Improve display contrast without changing model inputs.
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb


def save_visualization(
    sample_id,
    image,
    actual,
    predicted,
    valid,
    probabilities,
):
    rgb = get_rgb(image)

    # Mask invalid pixels so they do not appear as meaningful labels.
    actual_display = np.ma.masked_where(~valid, actual)
    predicted_display = np.ma.masked_where(~valid, predicted)

    errors = (predicted != actual) & valid
    false_water = (predicted == 1) & (actual == 0) & valid
    missed_water = (predicted == 0) & (actual == 1) & valid

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Sentinel-2 True-Color Image")

    axes[0, 1].imshow(
        actual_display,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_title("Actual Water Labels")

    axes[0, 2].imshow(
        predicted_display,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    axes[0, 2].set_title(
        f"Predicted Water (threshold {THRESHOLD:.2f})"
    )

    axes[1, 0].imshow(
        probabilities,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title("Water Probability")

    # Error map: false water predictions in red,
    # missed water predictions in yellow.
    error_map = np.zeros((*actual.shape, 3), dtype=np.float32)
    error_map[false_water] = [1.0, 0.0, 0.0]
    error_map[missed_water] = [1.0, 1.0, 0.0]

    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(error_map, alpha=0.75)
    axes[1, 1].set_title("Errors: Red = False Water, Yellow = Missed")

    axes[1, 2].imshow(
        errors,
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    axes[1, 2].set_title("All Incorrect Pixels")

    for ax in axes.flat:
        ax.axis("off")

    valid_count = int(valid.sum())
    if valid_count:
        false_water_count = int(false_water.sum())
        actual_water_count = int(((actual == 1) & valid).sum())
        predicted_water_count = int(((predicted == 1) & valid).sum())

        fig.suptitle(
            f"{sample_id} | Valid pixels: {valid_count:,} | "
            f"Actual water: {actual_water_count:,} | "
            f"Predicted water: {predicted_water_count:,} | "
            f"False water: {false_water_count:,}",
            fontsize=11,
        )

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{sample_id}_threshold_060.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

def main():
    model = load_model()

    dataset = UnifiedSatelliteDataset(
        split="test",
        modality="fused",
        augment=False,
    )

    sample_ids = dataset.sample_ids

    preferred = [
        sample_id
        for sample_id in PREFERRED_SAMPLES
        if sample_id in sample_ids
    ]

    # Use preferred examples first, then fill from the test set.
    selected_ids = preferred[:NUM_SAMPLES]

    for sample_id in sample_ids:
        if len(selected_ids) >= NUM_SAMPLES:
            break
        if sample_id not in selected_ids:
            selected_ids.append(sample_id)

    print(f"\nVisualizing {len(selected_ids)} test samples.")

    index_by_id = {
        sample_id: index
        for index, sample_id in enumerate(sample_ids)
    }

    with torch.no_grad():
        for sample_id in selected_ids:
            sample = dataset[index_by_id[sample_id]]

            # Dataset contract: image, labels, valid, modality.
            if not isinstance(sample, (tuple, list)) or len(sample) < 3:
                raise ValueError(
                    "Unexpected dataset output. Expected at least "
                    "(image, labels, valid)."
                )

            image_tensor, labels_tensor, valid_tensor = sample[:3]

            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = torch.as_tensor(image_tensor)
            if not isinstance(labels_tensor, torch.Tensor):
                labels_tensor = torch.as_tensor(labels_tensor)
            if not isinstance(valid_tensor, torch.Tensor):
                valid_tensor = torch.as_tensor(valid_tensor)

            image_batch = image_tensor.unsqueeze(0).to(DEVICE).float()

            logits = model(image_batch, modality="fused")
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

            actual = labels_tensor.squeeze().cpu().numpy().astype(np.uint8)
            valid = valid_tensor.squeeze().cpu().numpy().astype(bool)

            predicted = (probabilities >= THRESHOLD).astype(np.uint8)

            # Only score/display valid reference pixels.
            predicted[~valid] = 0
            actual[~valid] = 0

            image = image_tensor.cpu().numpy()

            save_visualization(
                sample_id=sample_id,
                image=image,
                actual=actual,
                predicted=predicted,
                valid=valid,
                probabilities=probabilities,
            )

    print("\nVisualization completed.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
