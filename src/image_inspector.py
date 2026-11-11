
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image
import rasterio


PROJECT_DIR = Path(r"D:\Satellite_AI_Project")

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff"
}

STANDARD_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def inspect_image(image_path):
    """Inspect an image and recommend a possible model pathway."""

    path = Path(image_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension: {extension}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    info = {
        "file_name": path.name,
        "file_format": extension,
        "file_size_bytes": path.stat().st_size,
        "width": None,
        "height": None,
        "channels": None,
        "data_type": None,
        "color_interpretation": None,
        "crs": None,
        "metadata": {},
        "recommended_path": "UNSUPPORTED_OR_UNVERIFIED",
        "warnings": [],
    }

    # --------------------------------------------------
    # 1. Inspect PNG and JPEG images
    # --------------------------------------------------
    if extension in STANDARD_IMAGE_EXTENSIONS:
        with Image.open(path) as img:
            info["width"], info["height"] = img.size
            info["channels"] = len(img.getbands())
            info["data_type"] = str(np.asarray(img).dtype)
            info["color_interpretation"] = img.mode

            info["metadata"] = {
                str(key): str(value)
                for key, value in list(img.info.items())[:20]
                if isinstance(value, (str, int, float, bool))
            }

            if img.mode in ("RGB", "RGBA"):
                info["recommended_path"] = "RGB_MODEL_CANDIDATE"

                if img.mode == "RGBA":
                    info["warnings"].append(
                        "RGBA detected. The alpha channel must be "
                        "removed before RGB model inference."
                    )

                info["warnings"].append(
                    "Verify that this image is suitable RGB imagery. "
                    "A three-colour false-colour composite may also "
                    "have RGB channels."
                )

            elif img.mode == "P":
                info["warnings"].append(
                    "Palette image detected. Convert it to RGB "
                    "during preprocessing if appropriate."
                )

            elif img.mode in ("L", "1", "I", "F", "I;16"):
                info["warnings"].append(
                    "Grayscale or single-channel image detected. "
                    "It cannot be sent directly to a three-channel "
                    "RGB model."
                )

            else:
                info["warnings"].append(
                    f"Image mode {img.mode!r} requires compatible "
                    "preprocessing before model inference."
                )

    # --------------------------------------------------
    # 2. Inspect TIFF and GeoTIFF images
    # --------------------------------------------------
    else:
        with rasterio.open(path) as src:
            info["width"] = src.width
            info["height"] = src.height
            info["channels"] = src.count
            info["data_type"] = ", ".join(src.dtypes)
            info["crs"] = str(src.crs) if src.crs else None

            info["color_interpretation"] = [
                str(value).split(".")[-1]
                for value in src.colorinterp
            ]

            info["metadata"] = {
                str(key): str(value)
                for key, value in list(src.tags().items())[:20]
            }

            band_names = list(src.descriptions)
            info["metadata"]["band_descriptions"] = str(band_names)

            interpretations = info["color_interpretation"]

            if (
                src.count >= 3
                and interpretations[:3] == ["red", "green", "blue"]
            ):
                info["recommended_path"] = "RGB_MODEL_CANDIDATE"

                info["warnings"].append(
                    "RGB bands detected. Confirm that the image's "
                    "radiometric scaling and preprocessing are "
                    "compatible with the RGB model."
                )

                if src.count > 3:
                    info["warnings"].append(
                        "Additional bands exist. Select the RGB bands "
                        "explicitly before RGB model inference."
                    )

            elif src.count == 15:
                info["recommended_path"] = (
                    "15_CHANNEL_MODEL_CANDIDATE"
                )

                info["warnings"].append(
                    "Verify the expected input order: two Sentinel-1 "
                    "channels followed by thirteen Sentinel-2 channels. "
                    "The current model's training preprocessing must "
                    "also be reproduced."
                )

            else:
                info["warnings"].append(
                    "Band count or colour interpretation does not "
                    "identify a verified compatible model pathway."
                )

    # --------------------------------------------------
    # 3. General validation
    # --------------------------------------------------
    if info["width"] == 0 or info["height"] == 0:
        info["warnings"].append(
            "Invalid image dimensions."
        )

    if info["width"] and info["height"]:
        info["pixel_count"] = info["width"] * info["height"]

    if info["crs"] is None and extension in {".tif", ".tiff"}:
        info["warnings"].append(
            "No coordinate reference system was detected. "
            "This does not necessarily prevent image classification."
        )

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Automatically inspect satellite image formats."
    )

    parser.add_argument(
        "image",
        nargs="?",
        help="Full path to a PNG, JPG, TIFF, or GeoTIFF image."
    )

    args = parser.parse_args()

    image_path = args.image

    if not image_path:
        image_path = input(
            "Enter the full path of the image: "
        ).strip().strip('"')

    try:
        result = inspect_image(image_path)

        print("\n--- IMAGE INSPECTION REPORT ---")
        print(json.dumps(result, indent=4))
        print("\nInspection completed.")
        print("No model weights were changed.")

    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"\nInspection failed: {exc}")

    except Exception as exc:
        print(f"\nUnexpected error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
