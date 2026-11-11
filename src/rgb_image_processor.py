
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}


def load_rgb_image(image_path):
    """Load a JPG or PNG image and prepare it for RGB AI processing."""
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError("Please provide a JPG, JPEG, or PNG image.")

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        rgb_image = image.convert("RGB")
        rgb_array = np.asarray(rgb_image, dtype=np.float32) / 255.0

    return rgb_image, rgb_array


def inspect_rgb_image(image_path):
    """Display basic information about an input satellite image."""
    image, image_array = load_rgb_image(image_path)

    print("RGB IMAGE CHECK")
    print("-" * 35)
    print(f"Image size: {image.width} x {image.height}")
    print(f"Image mode: {image.mode}")
    print(f"Array shape: {image_array.shape}")
    print(f"Pixel value range: {image_array.min():.3f} to "
          f"{image_array.max():.3f}")
    print("Image loaded successfully.")

    return image, image_array


if __name__ == "__main__":
    print("RGB image processor is ready.")
    print("Supported formats: JPG, JPEG, PNG")
    print("Next: provide a satellite image to test.")
