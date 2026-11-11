
from pathlib import Path
import random

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

import torch
from torch.utils.data import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "sen1floods11_v1.1"
)

RASTER_DIR = DATA_DIR / "data"
SPLIT_DIR = DATA_DIR / "splits"

SAR_DIR = RASTER_DIR / "S1GRDHand"
OPTICAL_DIR = RASTER_DIR / "S2L1CHand"
LABEL_DIR = RASTER_DIR / "LabelHand"

SPLIT_FILES = {
    "train": SPLIT_DIR / "flood_train_data.txt",
    "val": SPLIT_DIR / "flood_valid_data.txt",
    "valid": SPLIT_DIR / "flood_valid_data.txt",
    "validation": SPLIT_DIR / "flood_valid_data.txt",
    "test": SPLIT_DIR / "flood_test_data.txt",
}

SUPPORTED_MODALITIES = {"sar", "optical", "fused"}


# ============================================================
# RASTER HELPERS
# ============================================================

def resolve_raster_path(directory, sample_id, suffix):
    """
    Resolve a raster filename such as:
        Ghana_103272_S1Hand.tif
        Ghana_103272_S2Hand.tif
        Ghana_103272_LabelHand.tif
    """
    path = directory / f"{sample_id}_{suffix}.tif"

    if path.exists():
        return path

    # Handle a split entry that already contains a suffix.
    possible_suffixes = (
        "_S1Hand",
        "_S2Hand",
        "_LabelHand",
        ".tif",
        ".tiff",
    )

    base = sample_id

    for ending in possible_suffixes:
        if base.endswith(ending):
            base = base[:-len(ending)]

    path = directory / f"{base}_{suffix}.tif"

    if not path.exists():
        raise FileNotFoundError(
            f"Raster not found:\n{path}\n"
            f"Sample ID: {sample_id}"
        )

    return path


def read_raster(path):
    """
    Read a raster as float32.
    Masked/nodata pixels are represented as NaN.
    """
    with rasterio.open(path) as src:
        array = src.read(masked=True).astype(np.float32).filled(np.nan)

        metadata = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
            "count": src.count,
        }

    return array, metadata


def align_raster_to_reference(source_path, reference_path,
                              resampling=Resampling.bilinear):
    """
    Reproject/resample every source band onto the exact reference grid.

    Image bands use bilinear interpolation.
    Use nearest-neighbor interpolation for categorical labels.
    """
    with rasterio.open(reference_path) as reference:
        dst_height = reference.height
        dst_width = reference.width
        dst_transform = reference.transform
        dst_crs = reference.crs

    with rasterio.open(source_path) as source:
        if source.crs is None or dst_crs is None:
            raise ValueError(
                f"Missing CRS in source or reference raster:\n"
                f"{source_path}"
            )

        source_array = (
            source.read(masked=True)
            .astype(np.float32)
            .filled(np.nan)
        )

        source_transform = source.transform
        source_crs = source.crs
        source_nodata = np.nan

        destination = np.full(
            (source.count, dst_height, dst_width),
            np.nan,
            dtype=np.float32,
        )

        for band_index in range(source.count):
            reproject(
                source=source_array[band_index],
                destination=destination[band_index],
                src_transform=source_transform,
                src_crs=source_crs,
                src_nodata=source_nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=resampling,
                init_dest_nodata=True,
            )

    return destination


# ============================================================
# PREPROCESSING
# ============================================================

def normalize_sar(sar):
    """
    Normalize each SAR band independently.

    Non-finite pixels remain NaN until the validity mask is built.
    """
    sar = sar.astype(np.float32, copy=True)

    for band_index in range(sar.shape[0]):
        band = sar[band_index]
        finite = np.isfinite(band)

        if not finite.any():
            continue

        values = band[finite]
        mean = values.mean()
        std = values.std()

        if std < 1e-6:
            std = 1.0

        band[finite] = (values - mean) / std

    return sar


def normalize_optical(optical):
    """
    Convert Sentinel-2 reflectance values to an approximate 0-1 scale.

    Sen1Floods11 Sentinel-2 data commonly uses a scale of 10000.
    Values outside the expected range are clipped after scaling.
    """
    optical = optical.astype(np.float32, copy=True)

    finite = np.isfinite(optical)
    optical[finite] = optical[finite] / 10000.0
    optical[finite] = np.clip(optical[finite], 0.0, 1.0)

    return optical


# ============================================================
# DATASET
# ============================================================

class UnifiedSatelliteDataset(Dataset):
    """
    Dataset for Sen1Floods11.

    modality="sar":
        2 Sentinel-1 radar channels

    modality="optical":
        13 Sentinel-2 optical channels

    modality="fused":
        2 Sentinel-1 + 13 Sentinel-2 = 15 channels

    Each item returns:
        image:    FloatTensor [C, H, W]
        labels:   FloatTensor [1, H, W], values 0 or 1
        valid:    BoolTensor  [1, H, W]
        modality: string
    """

    def __init__(
        self,
        split="train",
        modality="fused",
        augment=False,
    ):
        super().__init__()

        split = split.lower().strip()
        modality = modality.lower().strip()

        if split not in SPLIT_FILES:
            raise ValueError(
                f"Unsupported split '{split}'. "
                f"Choose train, val, or test."
            )

        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality '{modality}'. "
                f"Choose sar, optical, or fused."
            )

        self.split = split
        self.modality = modality
        self.augment = augment and split == "train"

        split_path = SPLIT_FILES[split]

        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found:\n{split_path}"
            )

        with open(split_path, "r", encoding="utf-8-sig") as file:
            self.sample_ids = [
                line.strip()
                for line in file
                if line.strip()
                and not line.strip().startswith("#")
            ]

        if not self.sample_ids:
            raise RuntimeError(
                f"No samples found in split file:\n{split_path}"
            )

        for directory in (SAR_DIR, OPTICAL_DIR, LABEL_DIR):
            if not directory.exists():
                raise FileNotFoundError(
                    f"Dataset directory not found:\n{directory}"
                )

        print(
            f"Loaded {len(self.sample_ids)} samples | "
            f"Split: {split} | Modality: {modality} | "
            f"Augmentation: {self.augment}"
        )

    def __len__(self):
        return len(self.sample_ids)

    def _get_paths(self, sample_id):
        sar_path = resolve_raster_path(
            SAR_DIR, sample_id, "S1Hand"
        )

        optical_path = resolve_raster_path(
            OPTICAL_DIR, sample_id, "S2Hand"
        )

        label_path = resolve_raster_path(
            LABEL_DIR, sample_id, "LabelHand"
        )

        return sar_path, optical_path, label_path

    def _load_sample(self, sample_id):
        sar_path, optical_path, label_path = self._get_paths(
            sample_id
        )

        # Labels define the reference grid so the original
        # target pixels are preserved without resampling labels.
        labels_array, label_meta = read_raster(label_path)

        if labels_array.shape[0] != 1:
            raise ValueError(
                f"Expected one label band for {sample_id}; "
                f"got {labels_array.shape[0]}."
            )

        labels = labels_array[0]

        # Read and align only the imagery required for this modality.
        if self.modality in ("sar", "fused"):
            sar = align_raster_to_reference(
                sar_path,
                label_path,
                resampling=Resampling.bilinear,
            )

            if sar.shape[0] != 2:
                raise ValueError(
                    f"Expected 2 SAR bands for {sample_id}; "
                    f"got {sar.shape[0]}."
                )

            sar = normalize_sar(sar)

        else:
            sar = None

        if self.modality in ("optical", "fused"):
            optical = align_raster_to_reference(
                optical_path,
                label_path,
                resampling=Resampling.bilinear,
            )

            if optical.shape[0] != 13:
                raise ValueError(
                    f"Expected 13 optical bands for {sample_id}; "
                    f"got {optical.shape[0]}."
                )

            optical = normalize_optical(optical)

        else:
            optical = None

        # Construct the requested model input.
        if self.modality == "sar":
            image = sar

        elif self.modality == "optical":
            image = optical

        else:
            # Channel order: Sentinel-1 first, then Sentinel-2.
            image = np.concatenate((sar, optical), axis=0)

        # Valid labels are 0 (non-water) and 1 (water).
        valid = (
            np.isfinite(labels)
            & ((labels == 0) | (labels == 1))
        )

        # Do not train on pixels where any selected input channel
        # is invalid or outside the aligned source coverage.
        valid &= np.isfinite(image).all(axis=0)

        # Convert invalid input values to zero only after creating
        # the validity mask.
        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        # Replace ignored labels with zero; valid masks exclude them.
        labels = np.where(
            valid,
            labels,
            0,
        ).astype(np.float32)

        # Optional spatial augmentation: apply the same transform
        # to imagery, labels, and validity mask.
        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, axis=2)
                labels = np.flip(labels, axis=1)
                valid = np.flip(valid, axis=1)

            if random.random() < 0.5:
                image = np.flip(image, axis=1)
                labels = np.flip(labels, axis=0)
                valid = np.flip(valid, axis=0)

            rotations = random.randint(0, 3)

            if rotations:
                image = np.rot90(
                    image, rotations, axes=(1, 2)
                )
                labels = np.rot90(
                    labels, rotations, axes=(0, 1)
                )
                valid = np.rot90(
                    valid, rotations, axes=(0, 1)
                )

        # np.flip/np.rot90 can produce negative-stride arrays.
        # Make them contiguous before converting to Torch tensors.
        image = np.ascontiguousarray(image, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.float32)
        valid = np.ascontiguousarray(valid, dtype=np.bool_)

        image_tensor = torch.from_numpy(image)
        labels_tensor = torch.from_numpy(labels).unsqueeze(0)
        valid_tensor = torch.from_numpy(valid).unsqueeze(0)

        return (
            image_tensor,
            labels_tensor,
            valid_tensor,
            self.modality,
        )

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]

        try:
            return self._load_sample(sample_id)

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sample '{sample_id}' "
                f"(split={self.split}, modality={self.modality}). "
                f"Original error: {exc}"
            ) from exc


# ============================================================
# QUICK SELF-TEST
# ============================================================

if __name__ == "__main__":
    for modality in ("sar", "optical", "fused"):
        print(f"\nTesting modality: {modality}")

        dataset = UnifiedSatelliteDataset(
            split="train",
            modality=modality,
            augment=False,
        )

        image, labels, valid, returned_modality = dataset[0]

        print("Modality:", returned_modality)
        print("Image shape:", tuple(image.shape))
        print("Label shape:", tuple(labels.shape))
        print("Valid-mask shape:", tuple(valid.shape))
        print("Image dtype:", image.dtype)
        print("Label values:", torch.unique(labels).tolist())
        print("Valid pixels:", int(valid.sum().item()))
        print("All image values finite:", bool(torch.isfinite(image).all()))

        if not valid.any().item():
            print(
                "WARNING: This sample has no valid pixels. "
                "Investigate its raster coverage and labels."
            )

    print("\nDataset self-test completed.")
