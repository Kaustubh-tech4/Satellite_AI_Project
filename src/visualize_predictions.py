
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from satellite_dataset import SatelliteFloodDataset
from unet_model import UNet


PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
CHECKPOINT = PROJECT_DIR / "models" / "unet_best.pth"
OUTPUT_DIR = PROJECT_DIR / "results"
DEVICE = torch.device("cpu")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SatelliteFloodDataset("test")
    model = UNet(in_channels=15, base_channels=16)

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Visualize the first test sample.
    image, labels, valid_mask = dataset[0]

    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        probabilities = torch.sigmoid(logits[0, 0]).numpy()

    predictions = probabilities >= 0.5

    # Combined input channels: S1 = 0-1, S2 = 2-14.
    # Sentinel-2 RGB composite uses B4, B3, B2.
    rgb = np.stack(
        [image[5].numpy(), image[4].numpy(), image[3].numpy()],
        axis=-1,
    )

    low, high = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - low) / max(high - low, 1e-8), 0, 1)

    labels_np = labels.numpy()
    valid_np = valid_mask.numpy().astype(bool)

    reference = np.where(valid_np, labels_np, np.nan)
    predicted_view = np.where(valid_np, predictions.astype(float), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB Image")

    axes[1].imshow(reference, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Reference Water Labels")

    axes[2].imshow(predicted_view, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("U-Net Water Predictions")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "unet_test_prediction.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Test sample:", dataset.sample_ids[0])
    print("Prediction image saved:", output_path)

    plt.show()


if __name__ == "__main__":
    main()
