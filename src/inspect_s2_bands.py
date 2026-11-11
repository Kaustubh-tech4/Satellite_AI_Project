
from pathlib import Path
import rasterio

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
S2_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "sen1floods11_v1.1"
    / "data"
    / "S2L1CHand"
)

files = sorted(S2_DIR.glob("*_S2Hand.tif"))

if not files:
    raise FileNotFoundError(
        f"No Sentinel-2 TIFF files found in: {S2_DIR}"
    )

sample_path = files[0]

print("Sample file:", sample_path.name)

with rasterio.open(sample_path) as src:
    print("Image width:", src.width)
    print("Image height:", src.height)
    print("Number of bands:", src.count)
    print("Data types:", src.dtypes)
    print("CRS:", src.crs)
    print("Band descriptions:", src.descriptions)
    print("Color interpretation:", src.colorinterp)

    for index in range(1, src.count + 1):
        band = src.read(index, masked=True)
        print(
            f"Band {index}: "
            f"description={src.descriptions[index - 1]}, "
            f"minimum={band.min()}, "
            f"maximum={band.max()}"
        )
