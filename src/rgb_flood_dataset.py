
from pathlib import Path
import random

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")

BASE_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "sen1floods11_v1.1"
)

DATA_DIR = BASE_DIR / "data"
SPLIT_DIR = BASE_DIR / "splits"

S2_DIR = DATA_DIR / "S2L1CHand"
LABEL_DIR = DATA_DIR / "LabelHand"

SPLIT_FILES = {
    "train": "flood_train_data.txt",
    "val": "flood_valid_data.txt",
    "test": "flood_test_data.txt",
}


# --------------------------------------------------
# RGB FLOOD DATASET
# --------------------------------------------------

class RGBFloodDataset(Dataset):

    def __init__(self, split="train", augment=False):

        if split not in SPLIT_FILES:
            raise ValueError(
                f"Invalid split: {split}. "
                "Use 'train', 'val', or 'test'."
            )

        self.split = split

        # Augmentation is enabled only for training.
        self.augment = bool(augment and split == "train")

        self.s2_dir = S2_DIR
        self.label_dir = LABEL_DIR

        split_path = SPLIT_DIR / SPLIT_FILES[split]

        if not split_path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {split_path}"
            )

        self.sample_ids = [
            line.strip()
            for line in split_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        if not self.sample_ids:
            raise ValueError(
                f"No sample IDs found in {split_path}"
            )

        # Check all required files before starting.
        missing_files = []

        for sample_id in self.sample_ids:

            s2_path = (
                self.s2_dir
                / f"{sample_id}_S2Hand.tif"
            )

            label_path = (
                self.label_dir
                / f"{sample_id}_LabelHand.tif"
            )

            if not s2_path.is_file():
                missing_files.append(str(s2_path))

            if not label_path.is_file():
                missing_files.append(str(label_path))

        if missing_files:
            preview = "\n".join(missing_files[:10])

            raise FileNotFoundError(
                "Missing dataset files:\n"
                + preview
                + (
                    f"\n...and {len(missing_files) - 10} more"
                    if len(missing_files) > 10
                    else ""
                )
            )

        print(f"\nLoaded {split.upper()} split")
        print(f"Samples: {len(self.sample_ids)}")
        print(f"Augmentation: {self.augment}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):

        sample_id = self.sample_ids[index]

        s2_path = (
            self.s2_dir
            / f"{sample_id}_S2Hand.tif"
        )

        label_path = (
            self.label_dir
            / f"{sample_id}_LabelHand.tif"
        )

        # ------------------------------------------
        # READ SENTINEL-2 RGB BANDS
        # Rasterio uses 1-based band numbering.
        # B4 = Red, B3 = Green, B2 = Blue.
        # ------------------------------------------

        with rasterio.open(s2_path) as src:

            if src.count != 13:
                raise ValueError(
                    f"{sample_id}: expected 13 Sentinel-2 "
                    f"bands, found {src.count}"
                )

            descriptions = src.descriptions

            # Validate the bands when descriptions exist.
            if all(descriptions):

                expected = {
                    2: "B2",
                    3: "B3",
                    4: "B4",
                }

                for band_number, expected_name in expected.items():

                    actual_name = descriptions[
                        band_number - 1
                    ]

                    if actual_name != expected_name:
                        raise ValueError(
                            f"{sample_id}: band "
                            f"{band_number} should be "
                            f"{expected_name}, but found "
                            f"{actual_name}"
                        )

            # Shape: (3, height, width)
            rgb = src.read([4, 3, 2]).astype(
                np.float32
            )

        # ------------------------------------------
        # READ FLOOD LABELS
        # 0 = non-water
        # 1 = water
        # Other values = invalid/unlabeled
        # ------------------------------------------

        with rasterio.open(label_path) as src:
            labels = src.read(1)

        if rgb.shape[1:] != labels.shape:
            raise ValueError(
                f"{sample_id}: image and label sizes "
                f"do not match. RGB={rgb.shape[1:]}, "
                f"labels={labels.shape}"
            )

        # ------------------------------------------
        # NORMALIZE RGB
        # Sentinel-2 reflectance is scaled by 10000.
        # Use this same normalization at inference.
        # ------------------------------------------

        rgb = rgb / 10000.0

        rgb = np.nan_to_num(
            rgb,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ------------------------------------------
        # VALID PIXEL MASK
        # Do not treat unlabeled pixels as water.
        # ------------------------------------------

        valid_mask = (labels == 0) | (labels == 1)

        clean_labels = np.where(
            valid_mask,
            labels,
            0,
        ).astype(np.float32)

        # Copy arrays to avoid negative-stride issues.
        image = torch.from_numpy(
            np.ascontiguousarray(rgb)
        )

        label_tensor = torch.from_numpy(
            np.ascontiguousarray(clean_labels)
        )

        valid_tensor = torch.from_numpy(
            np.ascontiguousarray(valid_mask)
        ).bool()

        # ------------------------------------------
        # AUGMENT TRAINING DATA ONLY
        # Apply identical transformations to image,
        # labels, and valid mask.
        # ------------------------------------------

        if self.augment:

            if random.random() < 0.5:

                image = torch.flip(image, dims=[2])
                label_tensor = torch.flip(
                    label_tensor, dims=[1]
                )
                valid_tensor = torch.flip(
                    valid_tensor, dims=[1]
                )

            if random.random() < 0.5:

                image = torch.flip(image, dims=[1])
                label_tensor = torch.flip(
                    label_tensor, dims=[0]
                )
                valid_tensor = torch.flip(
                    valid_tensor, dims=[0]
                )

            k = random.randint(0, 3)

            if k:
                image = torch.rot90(
                    image, k, dims=[1, 2]
                )
                label_tensor = torch.rot90(
                    label_tensor, k, dims=[0, 1]
                )
                valid_tensor = torch.rot90(
                    valid_tensor, k, dims=[0, 1]
                )

        return image, label_tensor, valid_tensor

    # --------------------------------------------------
    # VALIDATION / DATASET DIAGNOSTICS
    # --------------------------------------------------

    def check_samples(self, limit=None):

        total_valid = 0
        total_water = 0
        zero_valid_samples = []

        count = (
            len(self.sample_ids)
            if limit is None
            else min(limit, len(self.sample_ids))
        )

        print(f"\nChecking {self.split.upper()} samples")

        for i in range(count):

            image, labels, valid = self[i]

            valid_count = int(valid.sum().item())

            water_count = int(
                ((labels == 1) & valid).sum().item()
            )

            total_valid += valid_count
            total_water += water_count

            if valid_count == 0:
                zero_valid_samples.append(
                    self.sample_ids[i]
                )

            print(
                f"{self.sample_ids[i]} | "
                f"Valid pixels: {valid_count} | "
                f"Water pixels: {water_count}"
            )

        print("\nSummary")
        print(f"Samples checked: {count}")
        print(f"Valid pixels: {total_valid}")
        print(f"Water pixels: {total_water}")
        print(
            "Zero-valid samples: "
            f"{len(zero_valid_samples)}"
        )

        if zero_valid_samples:
            print(
                "Samples with zero valid pixels:",
                ", ".join(zero_valid_samples[:20]),
            )

        if total_valid == 0:
            print(
                "WARNING: No valid labels were found "
                "in the samples checked."
            )

        return {
            "samples_checked": count,
            "valid_pixels": total_valid,
            "water_pixels": total_water,
            "zero_valid_samples": zero_valid_samples,
        }


# --------------------------------------------------
# TEST ALL THREE SPLITS
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("RGB FLOOD DATASET DIAGNOSTICS")
    print("=" * 60)

    for split in ("train", "val", "test"):

        dataset = RGBFloodDataset(
            split=split,
            augment=(split == "train"),
        )

        image, labels, valid = dataset[0]

        print(f"\n{split.upper()} DATASET")
        print(f"Samples: {len(dataset)}")
        print(f"Image shape: {tuple(image.shape)}")
        print(f"Label shape: {tuple(labels.shape)}")
        print(f"Valid mask shape: {tuple(valid.shape)}")
        print(f"Image dtype: {image.dtype}")
        print(f"Valid pixels in first sample: {int(valid.sum())}")
        print(
            "Water pixels in first sample: "
            f"{int(((labels == 1) & valid).sum())}"
        )
        print(
            f"Augmentation enabled: {dataset.augment}"
        )

        # Check every validation and test sample.
        # Check the first 10 training samples.
        if split == "train":
            dataset.check_samples(limit=10)
        else:
            dataset.check_samples()

    print("\nDataset diagnostics completed.")
