
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
DATA_PATH = PROJECT_DIR / "data" / "processed" / "flood_sample.npz"
OUTPUT_DIR = PROJECT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data = np.load(DATA_PATH)

    s2 = data["sentinel2"]
    labels = data["labels"]
    valid_mask = data["valid_mask"]

    # Display an RGB composite using Sentinel-2 bands:
    # B4 = red, B3 = green, B2 = blue.
    rgb = np.stack([s2[3], s2[2], s2[1]], axis=-1)

    # Percentile-based contrast adjustment.
    low, high = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - low) / (high - low + 1e-8), 0, 1)

    # Mask invalid label pixels for visualization.
    label_view = np.ma.masked_where(~valid_mask, labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 Satellite Image")
    axes[0].axis("off")

    image = axes[1].imshow(label_view, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Flood Dataset Water Labels")
    axes[1].axis("off")

    fig.colorbar(image, ax=axes[1], ticks=[0, 1], label="0 = Non-water, 1 = Water")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "flood_sample_visualization.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print("Visualization saved to:", output_path)

    plt.show()


if __name__ == "__main__":
    main()
