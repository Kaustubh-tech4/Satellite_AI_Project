
from pathlib import Path
import random

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
BASE_DIR = PROJECT_DIR / "data" / "raw" / "sen1floods11_v1.1"
DATA_DIR = BASE_DIR / "data"
SPLIT_DIR = BASE_DIR / "splits"


class SatelliteFloodDataset(Dataset):
    def __init__(self, split="train", augment=True):
        split_files = {
            "train": "flood_train_data.txt",
            "val": "flood_valid_data.txt",
            "test": "flood_test_data.txt",
        }

        if split not in split_files:
            raise ValueError("split must be train, val, or test")

        self.split = split

        # Augmentation is applied only to training samples.
        self.augment = augment and split == "train"

        self.s1_dir = DATA_DIR / "S1GRDHand"
        self.s2_dir = DATA_DIR / "S2L1CHand"
        self.label_dir = DATA_DIR / "LabelHand"

        split_path = SPLIT_DIR / split_files[split]

        if not split_path.is_file():
            raise FileNotFoundError(
                f"Split file not found: {split_path}"
            )

        self.sample_ids = [
            line.strip()
            for line in split_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        # Verify the required files before training.
        for sample_id in self.sample_ids:
            paths = [
                self.s1_dir / f"{sample_id}_S1Hand.tif",
                self.s2_dir / f"{sample_id}_S2Hand.tif",
                self.label_dir / f"{sample_id}_LabelHand.tif",
            ]

            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing file for {sample_id}: {path}"
                    )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]

        s1_path = self.s1_dir / f"{sample_id}_S1Hand.tif"
        s2_path = self.s2_dir / f"{sample_id}_S2Hand.tif"
        label_path = self.label_dir / f"{sample_id}_LabelHand.tif"

        # Read satellite imagery.
        with rasterio.open(s1_path) as src:
            s1 = src.read().astype(np.float32)

        with rasterio.open(s2_path) as src:
            s2 = src.read().astype(np.float32)

        with rasterio.open(label_path) as src:
            labels = src.read(1)

        if (
            s1.shape[1:] != s2.shape[1:]
            or s1.shape[1:] != labels.shape
        ):
            raise ValueError(
                f"Image dimensions do not match: {sample_id}"
            )

        # Standardize each Sentinel-1 band per image.
        for band in range(s1.shape[0]):
            values = s1[band]
            mean = values.mean()
            std = values.std()
            s1[band] = (values - mean) / max(float(std), 1e-6)

        # Scale Sentinel-2 L1C digital values.
        s2 = s2 / 10000.0

        # Combine 2 Sentinel-1 and 13 Sentinel-2 channels.
        image = np.concatenate([s1, s2], axis=0)

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Labels: -1 = invalid, 0 = non-water, 1 = water.
        valid_mask = (labels == 0) | (labels == 1)
        clean_labels = np.where(
            valid_mask, labels, 0
        ).astype(np.float32)

        image = torch.from_numpy(image.copy())
        labels_tensor = torch.from_numpy(clean_labels.copy())
        valid_tensor = torch.from_numpy(valid_mask.copy())

        # Data augmentation for training only.
        # Apply identical transformations to image and labels.
        if self.augment:
            # Horizontal flip.
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                labels_tensor = torch.flip(labels_tensor, dims=[1])
                valid_tensor = torch.flip(valid_tensor, dims=[1])

            # Vertical flip.
            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])
                labels_tensor = torch.flip(labels_tensor, dims=[0])
                valid_tensor = torch.flip(valid_tensor, dims=[0])

            # Rotate by 0, 90, 180, or 270 degrees.
            k = random.randint(0, 3)
            image = torch.rot90(image, k, dims=[1, 2])
            labels_tensor = torch.rot90(labels_tensor, k, dims=[0, 1])
            valid_tensor = torch.rot90(valid_tensor, k, dims=[0, 1])

        return image, labels_tensor, valid_tensor


if __name__ == "__main__":
    dataset = SatelliteFloodDataset("train", augment=True)

    print("Training samples:", len(dataset))
    print("Augmentation enabled:", dataset.augment)

    image, labels, valid_mask = dataset[0]

    print("Input tensor shape:", tuple(image.shape))
    print("Label tensor shape:", tuple(labels.shape))
    print("Valid mask shape:", tuple(valid_mask.shape))
    print("Input channels:", image.shape[0])
    print("Valid pixels:", int(valid_mask.sum()))
    print(
        "Water pixels:",
        int(((labels == 1) & valid_mask).sum()),
    )
    print(
        "Non-water pixels:",
        int(((labels == 0) & valid_mask).sum()),
    )

    val_dataset = SatelliteFloodDataset("val")
    test_dataset = SatelliteFloodDataset("test")

    print("Validation augmentation:", val_dataset.augment)
    print("Test augmentation:", test_dataset.augment)
    print("Data loader test successful.")
