
from pathlib import Path
import rasterio

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")

REPO_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "sen1floods11_repo"
    / "Sen1Floods11-master"
)

FILES = {
    "Sentinel-1 radar": REPO_DIR / "sample" / "S1" / "Spain_7370579_S1Hand.tif",
    "Sentinel-2 optical": REPO_DIR / "sample" / "S2" / "Spain_7370579_S2Hand.tif",
    "Flood labels": REPO_DIR / "sample" / "Labels" / "Spain_7370579_LabelHand.tif",
}

def main():
    print("Satellite AI Project - Dataset Verification")
    print("-" * 48)

    for name, path in FILES.items():
        print(f"\n{name}")

        if not path.exists():
            print("ERROR: File not found:", path)
            continue

        with rasterio.open(path) as image:
            print("Status: Found")
            print("Dimensions:", image.count, "bands x",
                  image.height, "x", image.width)
            print("Data type:", image.dtypes)
            print("Coordinate system:", image.crs)

    print("\nVerification finished.")

if __name__ == "__main__":
    main()
