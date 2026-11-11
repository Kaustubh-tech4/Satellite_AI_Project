
from pathlib import Path

import numpy as np
import rasterio

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
BASE_DIR = PROJECT_DIR / "data" / "raw" / "sen1floods11_v1.1"
DATA_DIR = BASE_DIR / "data"
SPLIT_DIR = BASE_DIR / "splits"

SPLITS = {
    "train": "flood_train_data.txt",
    "val": "flood_valid_data.txt",
    "test": "flood_test_data.txt",
}


def inspect_raster(path):
    with rasterio.open(path) as src:
        data = src.read()
        return {
            "shape": data.shape,
            "dtype": str(data.dtype),
            "crs": str(src.crs),
            "transform": str(src.transform),
            "bounds": str(src.bounds),
            "nodata": src.nodata,
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
            "mean": float(np.nanmean(data)),
            "nan_count": int(np.isnan(data).sum()),
        }


def main():
    total_samples = 0

    for split, filename in SPLITS.items():
        split_path = SPLIT_DIR / filename
        sample_ids = [
            line.strip()
            for line in split_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        print(f"\n{'=' * 60}")
        print(f"SPLIT: {split.upper()}")
        print("Samples:", len(sample_ids))
        total_samples += len(sample_ids)

        water_pixels = 0
        nonwater_pixels = 0
        invalid_pixels = 0
        no_water_samples = 0
        no_valid_samples = 0

        for sample_id in sample_ids:
            s1_path = DATA_DIR / "S1GRDHand" / f"{sample_id}_S1Hand.tif"
            s2_path = DATA_DIR / "S2L1CHand" / f"{sample_id}_S2Hand.tif"
            label_path = DATA_DIR / "LabelHand" / f"{sample_id}_LabelHand.tif"

            with rasterio.open(label_path) as src:
                labels = src.read(1)

            values, counts = np.unique(
                labels, return_counts=True
            )
            label_counts = dict(
                zip(values.tolist(), counts.tolist())
            )

            valid = (labels == 0) | (labels == 1)
            water = labels == 1
            nonwater = labels == 0

            water_count = int(water.sum())
            valid_count = int(valid.sum())

            water_pixels += water_count
            nonwater_pixels += int(nonwater.sum())
            invalid_pixels += int((~valid).sum())

            if water_count == 0:
                no_water_samples += 1

            if valid_count == 0:
                no_valid_samples += 1

            if sample_id == sample_ids[0]:
                print("\nExample sample:", sample_id)
                print("Label values and counts:", label_counts)
                print("Sentinel-1:", inspect_raster(s1_path))
                print("Sentinel-2:", inspect_raster(s2_path))
                print("Label raster:", inspect_raster(label_path))

        valid_total = water_pixels + nonwater_pixels
        water_percent = (
            100.0 * water_pixels / valid_total
            if valid_total else 0.0
        )

        print("\nSplit summary")
        print("Water pixels:", water_pixels)
        print("Non-water pixels:", nonwater_pixels)
        print("Invalid pixels:", invalid_pixels)
        print(f"Water among valid pixels: {water_percent:.2f}%")
        print("Samples with zero water pixels:", no_water_samples)
        print("Samples with zero valid pixels:", no_valid_samples)

    print(f"\nTotal samples across splits: {total_samples}")


if __name__ == "__main__":
    main()
