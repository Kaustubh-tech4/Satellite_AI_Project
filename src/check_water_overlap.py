
import os
import glob
import rasterio
import numpy as np

ROOT = os.path.join(
    "data", "raw", "sen1floods11_v1.1", "data"
)

LABEL_DIR = os.path.join(ROOT, "LabelHand")
JRC_DIR = os.path.join(ROOT, "JRCWaterHand")

# Allow tiny floating-point differences in raster transforms.
GRID_TOLERANCE = 1e-8

label_files = sorted(
    glob.glob(os.path.join(LABEL_DIR, "*.tif"))
)

total_valid = 0
total_label_water = 0
total_jrc_positive = 0
total_overlap = 0

matched = 0
missing = []
grid_mismatches = []

for label_path in label_files:
    label_name = os.path.basename(label_path)

    # Find the corresponding JRC file.
    base_name = label_name.removesuffix("_LabelHand.tif")
    jrc_name = base_name + "_JRCWaterHand.tif"
    jrc_path = os.path.join(JRC_DIR, jrc_name)

    if not os.path.isfile(jrc_path):
        missing.append(label_name)
        continue

    # Read label raster and its metadata.
    with rasterio.open(label_path) as src:
        labels = src.read(1)
        label_transform = src.transform
        label_crs = src.crs
        label_bounds = src.bounds

    # Read JRC raster and its metadata.
    with rasterio.open(jrc_path) as src:
        jrc = src.read(1)
        jrc_transform = src.transform
        jrc_crs = src.crs
        jrc_bounds = src.bounds

    # Check dimensions, coordinate systems, and pixel grids.
    same_grid = (
        labels.shape == jrc.shape
        and label_crs == jrc_crs
        and np.allclose(
            tuple(label_transform),
            tuple(jrc_transform),
            rtol=0,
            atol=GRID_TOLERANCE,
        )
        and np.allclose(
            [
                label_bounds.left,
                label_bounds.bottom,
                label_bounds.right,
                label_bounds.top,
            ],
            [
                jrc_bounds.left,
                jrc_bounds.bottom,
                jrc_bounds.right,
                jrc_bounds.top,
            ],
            rtol=0,
            atol=GRID_TOLERANCE,
        )
    )

    if not same_grid:
        grid_mismatches.append(label_name)
        continue

    matched += 1

    # LabelHand: -1 = invalid, 0 = non-water, 1 = water.
    valid_mask = labels != -1
    water_mask = labels == 1

    # JRCWaterHand: investigate positive values as reference water.
    jrc_water_mask = jrc == 1

    valid_label_water = water_mask & valid_mask
    valid_jrc_water = jrc_water_mask & valid_mask

    overlap_mask = valid_label_water & jrc_water_mask

    total_valid += int(valid_mask.sum())
    total_label_water += int(valid_label_water.sum())
    total_jrc_positive += int(valid_jrc_water.sum())
    total_overlap += int(overlap_mask.sum())

print("\n========== WATER REFERENCE COMPARISON ==========")
print("Label files found:", len(label_files))
print("Matched and aligned tiles:", matched)
print("Missing JRC files:", len(missing))
print("Grid mismatches:", len(grid_mismatches))

print("\n========== PIXEL STATISTICS ==========")
print("Valid label pixels:", total_valid)
print("Label water pixels:", total_label_water)
print("JRC-positive pixels on valid labels:", total_jrc_positive)
print("Overlapping water pixels:", total_overlap)

if total_jrc_positive > 0:
    percentage = 100 * total_overlap / total_jrc_positive
    print(
        "JRC-positive pixels also labeled as water:",
        f"{percentage:.2f}%"
    )
else:
    print("JRC overlap percentage: N/A")

if total_label_water > 0:
    percentage = 100 * total_overlap / total_label_water
    print(
        "Label-water pixels overlapping JRC:",
        f"{percentage:.2f}%"
    )
else:
    print("Label-water overlap percentage: N/A")

if missing:
    print("\nFirst five missing JRC files:", missing[:5])

if grid_mismatches:
    print(
        "First five grid-mismatch files:",
        grid_mismatches[:5]
    )

print("\nComparison completed.")
