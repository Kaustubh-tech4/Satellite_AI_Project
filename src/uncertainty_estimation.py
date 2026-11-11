
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from satellite_dataset import SatelliteFloodDataset
from unet_model import UNet


# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "unet_augmented_best.pth")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

THRESHOLD = 0.40
DEVICE = torch.device("cpu")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    print("Loading trained flood segmentation model...")

    dataset = SatelliteFloodDataset("test", augment=False)
    model = UNet(in_channels=15, base_channels=16).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Inspect the first test image
    image, labels, valid_mask = dataset[0]
    image_batch = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_batch)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Normalized binary predictive entropy: 0 = confident,
    # 1 = maximally uncertain (probability close to 0.5).
    eps = 1e-7
    p = np.clip(probabilities, eps, 1.0 - eps)

    uncertainty = -(
        p * np.log2(p) +
        (1.0 - p) * np.log2(1.0 - p)
    )

    predictions = (probabilities >= THRESHOLD).astype(np.uint8)

    labels_np = labels.cpu().numpy()
    valid_np = valid_mask.cpu().numpy().astype(bool)

    if valid_np.any():
        mean_uncertainty = float(uncertainty[valid_np].mean())
        high_uncertainty_pct = float(
            (uncertainty[valid_np] >= 0.8).mean() * 100
        )
    else:
        mean_uncertainty = float("nan")
        high_uncertainty_pct = float("nan")

    # Save numeric results for later analysis
    output_npz = os.path.join(
        RESULTS_DIR, "uncertainty_sample.npz"
    )

    np.savez_compressed(
        output_npz,
        probabilities=probabilities,
        predictions=predictions,
        uncertainty=uncertainty,
        labels=labels_np,
        valid_mask=valid_np
    )

    # Create an RGB visualization from Sentinel-2 channels
    rgb = image[[3, 2, 1]].cpu().numpy()
    rgb = np.moveaxis(rgb, 0, -1)

    rgb_min = np.percentile(rgb, 2)
    rgb_max = np.percentile(rgb, 98)

    if rgb_max > rgb_min:
        rgb = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)
    else:
        rgb = np.zeros_like(rgb)

    # Plot image, predicted water, and uncertainty
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")

    predicted_display = predictions.astype(float)
    predicted_display[~valid_np] = np.nan
    axes[1].imshow(predicted_display, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Predicted Water (threshold 0.40)")

    uncertainty_display = uncertainty.copy()
    uncertainty_display[~valid_np] = np.nan
    im = axes[2].imshow(
        uncertainty_display, cmap="magma", vmin=0, vmax=1
    )
    axes[2].set_title("Predictive Uncertainty")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    output_png = os.path.join(
        RESULTS_DIR, "uncertainty_sample.png"
    )
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sample_id = getattr(dataset, "sample_ids", None)
    sample_name = sample_id[0] if sample_id else "First test sample"

    print("\n--- Uncertainty Analysis ---")
    print(f"Sample: {sample_name}")
    print(f"Decision threshold: {THRESHOLD:.2f}")
    print(f"Mean normalized entropy: {mean_uncertainty:.4f}")
    print(
        "Valid pixels with entropy >= 0.8: "
        f"{high_uncertainty_pct:.2f}%"
    )
    print(f"Saved numeric results: {output_npz}")
    print(f"Saved visualization: {output_png}")
    print("\nUncertainty analysis completed successfully.")


if __name__ == "__main__":
    main()
