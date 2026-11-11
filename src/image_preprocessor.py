
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def prepare_rgb_image(image_path, image_size=512):
    """
    Prepare an ordinary RGB image for an RGB model.

    Returns:
        tensor: Float32 tensor with shape (1, 3, H, W),
                with pixel values scaled to [0, 1].
        rgb_image: Resized RGB PIL image.

    Note:
        This preprocessing is intended for an RGB model.
        It is NOT compatible with the existing 15-channel
        satellite U-Net without a separate model.
    """

    image_path = Path(image_path)

    # 1. Check that the file exists.
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    # 2. Avoid silently converting scientific multispectral TIFFs.
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "This RGB preprocessor supports PNG, JPG and JPEG only. "
            "Use the multispectral/SAR processing pipeline for TIFF files."
        )

    # 3. Open image and correct its orientation when applicable.
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        rgb_image = image.convert("RGB")

    # 4. Resize to the requested model input size.
    # PIL expects (width, height).
    if isinstance(image_size, int):
        target_size = (image_size, image_size)
    else:
        target_size = tuple(image_size)

    if (
        len(target_size) != 2
        or any(not isinstance(x, int) or x <= 0 for x in target_size)
    ):
        raise ValueError(
            "image_size must be a positive integer or (width, height)."
        )

    rgb_image = rgb_image.resize(
        target_size,
        resample=Image.Resampling.BILINEAR
    )

    # 5. Convert image to a NumPy float32 array.
    image_array = np.array(
        rgb_image,
        dtype=np.float32,
        copy=True
    )

    # 6. Scale pixel values from [0, 255] to [0, 1].
    image_array /= 255.0

    # 7. Convert HWC to CHW, then add the batch dimension.
    tensor = torch.from_numpy(image_array)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()

    # Final shape: (1, 3, height, width)
    return tensor, rgb_image


def main():
    image_path = input(
        "Enter the full path of your RGB image: "
    ).strip().strip('"')

    try:
        tensor, prepared_image = prepare_rgb_image(
            image_path,
            image_size=512
        )

        print("\nRGB preprocessing successful!")
        print(f"Prepared image size: {prepared_image.size}")
        print(f"Tensor shape: {tuple(tensor.shape)}")
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"Minimum pixel value: {tensor.min().item():.4f}")
        print(f"Maximum pixel value: {tensor.max().item():.4f}")

        print("\nImage is ready as input for a compatible RGB model.")
        print("No model prediction was performed.")

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
