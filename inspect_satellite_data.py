
from pathlib import Path
import numpy as np
import rasterio

SAMPLE_DIR = Path(
    r"D:\Satellite_AI_Project\data\raw\sen1floods11_repo"
    r"\Sen1Floods11-master\sample"
)

files = {
    "Sentinel-1 radar": SAMPLE_DIR / "S1" / "Spain_7370579_S1Hand.tif",
    "Sentinel-2 optical": SAMPLE_DIR / "S2" / "Spain_7370579_S2Hand.tif",
    "Flood reference labels": SAMPLE_DIR / "Labels" / "Spain_7370579_LabelHand.tif",
}

for description, path in files.items():
    print(f"\n--- {description} ---")
    print("File:", path)

    if not path.exists():
        print("File not found.")
        continue

    with rasterio.open(path) as src:
        print("Dimensions (bands, height, width):",
              (src.count, src.height, src.width))
        print("Data types:", src.dtypes)
        print("Coordinate system:", src.crs)

        band = src.read(1, masked=True)
        values = band.compressed()

        if values.size:
            print("Minimum:", float(values.min()))
            print("Maximum:", float(values.max()))
            print("Mean:", float(values.mean()))

        if "Labels" in str(path):
            print("Label values and counts:",
                  np.unique(values, return_counts=True))
