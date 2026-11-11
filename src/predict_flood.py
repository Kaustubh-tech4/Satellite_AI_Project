
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unet_model import UNet

MODEL_PATH = PROJECT_ROOT / "models" / "unet_augmented_best.pth"
OUTPUT_DIR = PROJECT_ROOT / "results"
THRESHOLD = 0.40


# --------------------------------------------------
# 2. Load and preprocess satellite imagery
# --------------------------------------------------

def load_satellite_image(path, expected_bands):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Satellite file not found: {path}")

    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)

    if image.shape[0] != expected_bands:
        raise ValueError(
            f"{path.name}: expected {expected_bands} bands, "
            f"but found {image.shape[0]}."
        )

    # A pixel is valid only when all bands contain finite values.
    valid = np.isfinite(image).all(axis=0)

    # Treat all-zero pixels as invalid for this dataset.
    valid &= ~np.all(image == 0, axis=0)

    if not valid.any():
        raise ValueError(f"No valid pixels found in {path.name}.")

    if expected_bands == 2:
        # Sentinel-1: standardize each band using finite pixels.
        for band_index in range(image.shape[0]):
            band = image[band_index]
            finite = np.isfinite(band)

            if not finite.any():
                raise ValueError(
                    f"Band {band_index + 1} in {path.name} "
                    "contains no finite values."
                )

            mean = band[finite].mean()
            std = band[finite].std()

            # Avoid division by zero.
            if std < 1e-6:
                std = 1.0

            image[band_index] = np.where(
                finite,
                (band - mean) / std,
                0.0
            )

    elif expected_bands == 13:
        # Sentinel-2: scale reflectance as in the training pipeline.
        image = image / 10000.0

    # Replace remaining invalid values and exclude them from scoring.
    image = np.where(np.isfinite(image), image, 0.0)
    image[:, ~valid] = 0.0

    return image, valid


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    device = torch.device("cpu")
    model = UNet(in_channels=15, base_channels=16).to(device)

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
# 4. Generate a prediction
# --------------------------------------------------

def main():
    print("\n=== Satellite Flood/Water Prediction ===")

    s1_path = input(
        "Enter Sentinel-1 GeoTIFF path: "
    ).strip().strip('"')

    s2_path = input(
        "Enter Sentinel-2 GeoTIFF path: "
    ).strip().strip('"')

    print("\nLoading satellite images...")

    s1, valid_s1 = load_satellite_image(s1_path, 2)
    s2, valid_s2 = load_satellite_image(s2_path, 13)

    if s1.shape[1:] != s2.shape[1:]:
        raise ValueError(
            "Sentinel-1 and Sentinel-2 dimensions do not match. "
            "The images must be aligned and have matching dimensions."
        )

    # Exclude locations that are invalid in either input.
    valid_mask = valid_s1 & valid_s2

    if not valid_mask.any():
        raise ValueError("No common valid pixels in the two images.")

    # Combine Sentinel-1 and Sentinel-2 into 15 input bands.
    combined = np.concatenate([s1, s2], axis=0)

    if not np.isfinite(combined).all():
        raise ValueError(
            "Invalid values remain after preprocessing."
        )

    print(f"Input shape: {combined.shape}")
    print(f"Valid pixels: {valid_mask.sum():,}")
    print("Loading trained model...")

    model, device = load_model()

    tensor = torch.from_numpy(combined).unsqueeze(0).to(device)

    print("Generating prediction...")

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Only valid pixels are included in the predicted water mask.
    water_mask = (probabilities >= THRESHOLD) & valid_mask

    valid_count = int(valid_mask.sum())
    water_count = int(water_mask.sum())

    water_percentage = water_count / valid_count * 100

    # Normalized binary predictive entropy, in the range 0 to 1.
    p = np.clip(probabilities, 1e-7, 1.0 - 1e-7)

    entropy = -(
        p * np.log2(p) +
        (1.0 - p) * np.log2(1.0 - p)
    )

    mean_entropy = float(entropy[valid_mask].mean())

    # --------------------------------------------------
    # 5. Save prediction outputs
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_output = OUTPUT_DIR / "new_flood_prediction.png"
    probability_output = OUTPUT_DIR / "new_flood_probabilities.npy"
    mask_output = OUTPUT_DIR / "new_flood_mask.npy"
    uncertainty_output = OUTPUT_DIR / "new_flood_uncertainty.npy"

    # Sentinel-2 RGB display uses bands 4, 3, 2 in 1-based notation.
    # This assumes the input bands follow the dataset's standard order.
    rgb = np.transpose(s2[[3, 2, 1]], (1, 2, 0))

    # Percentile stretch for a more useful display.
    rgb_display = np.zeros_like(rgb)

    for channel in range(3):
        band = rgb[:, :, channel]
        values = band[valid_mask]

        low, high = np.percentile(values, [2, 98])

        if high > low:
            rgb_display[:, :, channel] = np.clip(
                (band - low) / (high - low), 0, 1
            )

    # Mark invalid pixels as black in the visualization.
    rgb_display[~valid_mask] = 0

    # Do not interpret invalid pixels as low-uncertainty predictions.
    entropy_display = np.where(valid_mask, entropy, np.nan)
    water_display = np.where(valid_mask, water_mask.astype(float), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(rgb_display)
    axes[0].set_title("Sentinel-2 RGB")

    axes[1].imshow(water_display, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Predicted Water Mask")

    entropy_image = axes[2].imshow(
        entropy_display, cmap="inferno", vmin=0, vmax=1
    )
    axes[2].set_title("Prediction Uncertainty")
    fig.colorbar(entropy_image, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(image_output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.save(probability_output, probabilities)
    np.save(mask_output, water_mask.astype(np.uint8))
    np.save(uncertainty_output, entropy)

    # --------------------------------------------------
    # 6. Display summary
    # --------------------------------------------------

    print("\n=== Prediction Results ===")
    print(f"Decision threshold: {THRESHOLD:.2f}")
    print(f"Valid pixels: {valid_count:,}")
    print(f"Predicted water pixels: {water_count:,}")
    print(f"Predicted water coverage: {water_percentage:.2f}%")
    print(f"Mean normalized entropy: {mean_entropy:.4f}")

    print("\nSaved files:")
    print(f"Visualization: {image_output}")
    print(f"Probabilities: {probability_output}")
    print(f"Water mask: {mask_output}")
    print(f"Uncertainty: {uncertainty_output}")

    print("\nPrediction completed successfully.")
    print(
        "Note: This model predicts water regions, not confirmed flooding. "
        "Water coverage alone cannot distinguish floods from normal rivers "
        "or lakes."
    )


if __name__ == "__main__":
    main()
