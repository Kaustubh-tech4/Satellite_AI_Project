
from pathlib import Path
import numpy as np
import rasterio

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")

REPO_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "sen1floods11_repo"
    / "Sen1Floods11-master"
)

S1_PATH = REPO_DIR / "sample" / "S1" / "Spain_7370579_S1Hand.tif"
S2_PATH = REPO_DIR / "sample" / "S2" / "Spain_7370579_S2Hand.tif"
LABEL_PATH = REPO_DIR / "sample" / "Labels" / "Spain_7370579_LabelHand.tif"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Preparing satellite flood data...")
    
    with rasterio.open(S1_PATH) as src:
        s1 = src.read().astype(np.float32)
        
    with rasterio.open(S2_PATH) as src:
        s2 = src.read().astype(np.float32)
        
    with rasterio.open(LABEL_PATH) as src:
        labels = src.read(1)

    # Keep only pixels with valid binary labels: 0 or 1.
    valid_mask = (labels == 0) | (labels == 1)

    print("Sentinel-1 shape:", s1.shape)
    print("Sentinel-2 shape:", s2.shape)
    print("Label shape:", labels.shape)
    print("Valid labeled pixels:", int(valid_mask.sum()))
    print("Water pixels:", int((labels[valid_mask] == 1).sum()))
    print("Non-water pixels:", int((labels[valid_mask] == 0).sum()))

    # Save source arrays and the validity mask for the next stage.
    np.savez_compressed(
        OUTPUT_DIR / "flood_sample.npz",
        sentinel1=s1,
        sentinel2=s2,
        labels=labels.astype(np.int16),
        valid_mask=valid_mask
    )

    print("Saved:", OUTPUT_DIR / "flood_sample.npz")
    print("Preparation complete.")


if __name__ == "__main__":
    main()
