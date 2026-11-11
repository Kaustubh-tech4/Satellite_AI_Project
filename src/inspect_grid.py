
import os
import rasterio

ROOT = "data/raw/sen1floods11_v1.1/data"

files = [
    ("LabelHand", "Ghana_103272_LabelHand.tif"),
    ("JRCWaterHand", "Ghana_103272_JRCWaterHand.tif"),
]

for folder, filename in files:
    path = os.path.join(ROOT, folder, filename)

    print(f"\nFILE: {path}")

    if not os.path.exists(path):
        print("File not found.")
        continue

    with rasterio.open(path) as src:
        print("Shape:", src.shape)
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Bounds:", src.bounds)
        print("Resolution:", src.res)
