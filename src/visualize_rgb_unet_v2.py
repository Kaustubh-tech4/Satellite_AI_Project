
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from rgb_flood_dataset import RGBFloodDataset
from rgb_unet_model import RGBUNet


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_DIR / "models" / "rgb_unet_v2_best.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR / "results" / "rgb_v2_visualizations"
)

DEVICE = torch.device("cpu")

# Number of valid test samples to visualize
NUM_SAMPLES = 8

# Use the same validation-selected threshold as evaluation
DEFAULT_THRESHOLD = 0.30


# ============================================================
# RGB DISPLAY
# ============================================================

def prepare_rgb_image(image_tensor):
    """
    Convert normalized Sentinel-2 RGB tensor to a displayable image.

    Input channels:
        0 = B4 (Red)
        1 = B3 (Green)
        2 = B2 (Blue)
    """

    rgb = image_tensor.detach().cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Stretch each channel independently for visualization only.
    # Model input values remain unchanged.
    display_rgb = np.zeros_like(rgb, dtype=np.float32)

    for channel in range(3):
        band = rgb[:, :, channel]
        valid_values = band[np.isfinite(band)]

        if valid_values.size == 0:
            continue

        low, high = np.percentile(
            valid_values,
            [2, 98]
        )

        if high > low:
            display_rgb[:, :, channel] = np.clip(
                (band - low) / (high - low),
                0,
                1
            )
        else:
            display_rgb[:, :, channel] = np.clip(
                band,
                0,
                1
            )

    return display_rgb


# ============================================================
# MASK DISPLAY
# ============================================================

def prepare_mask(mask, valid_mask):
    """
    Show water as white, non-water as black,
    and unlabeled pixels as gray.
    """

    mask = mask.astype(np.float32)
    valid_mask = valid_mask.astype(bool)

    display = np.full(
        mask.shape,
        0.5,
        dtype=np.float32
    )

    display[valid_mask & (mask == 0)] = 0.0
    display[valid_mask & (mask == 1)] = 1.0

    return display


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("RGB U-NET V2 - PREDICTION VISUALIZATION")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unexpected checkpoint format."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    threshold = float(
        checkpoint.get(
            "validation_threshold",
            DEFAULT_THRESHOLD
        )
    )

    model = RGBUNet(
        in_channels=3,
        base_channels=8
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Threshold: {threshold:.2f}")

    # --------------------------------------------------------
    # LOAD TEST DATASET
    # --------------------------------------------------------

    dataset = RGBFloodDataset(
        split="test",
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    saved_count = 0
    skipped_count = 0

    # --------------------------------------------------------
    # GENERATE VISUALIZATIONS
    # --------------------------------------------------------

    with torch.no_grad():

        for sample_index, batch in enumerate(loader):

            image, labels, valid_mask = batch

            image = image.to(
                DEVICE,
                dtype=torch.float32
            )

            labels = labels.to(DEVICE)
            valid_mask = valid_mask.to(DEVICE).bool()

            if not valid_mask.any():
                skipped_count += 1
                continue

            # Predict water probabilities
            logits = model(image)
            probabilities = torch.sigmoid(logits)

            prediction = (
                probabilities >= threshold
            )

            # Convert tensors to 2D arrays
            rgb_image = prepare_rgb_image(image[0])

            ground_truth = labels[0].cpu().numpy()
            valid = valid_mask[0].cpu().numpy()

            # Remove a possible singleton channel
            if ground_truth.ndim == 3:
                ground_truth = ground_truth[0]

            if valid.ndim == 3:
                valid = valid[0]

            predicted_mask = prediction[0].cpu().numpy()

            if predicted_mask.ndim == 3:
                predicted_mask = predicted_mask[0]

            probability_map = (
                probabilities[0, 0].cpu().numpy()
            )

            # Mask visualization: gray means unlabeled
            gt_display = prepare_mask(
                ground_truth,
                valid
            )

            pred_display = prepare_mask(
                predicted_mask.astype(np.float32),
                valid
            )

            # Calculate errors only on valid pixels
            target_water = ground_truth == 1

            predicted_water = predicted_mask & valid
            actual_water = target_water & valid

            tp = int(
                (predicted_water & actual_water).sum()
            )

            fp = int(
                (predicted_water & ~actual_water & valid).sum()
            )

            fn = int(
                (~predicted_water & actual_water).sum()
            )

            precision = (
                tp / (tp + fp)
                if tp + fp
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn
                else 0.0
            )

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )

            # ------------------------------------------------
            # DRAW FIGURE
            # ------------------------------------------------

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(17, 5)
            )

            axes[0].imshow(rgb_image)
            axes[0].set_title(
                "Sentinel-2 RGB"
            )

            axes[1].imshow(
                gt_display,
                cmap="gray",
                vmin=0,
                vmax=1
            )
            axes[1].set_title(
                "Ground Truth"
            )

            axes[2].imshow(
                pred_display,
                cmap="gray",
                vmin=0,
                vmax=1
            )
            axes[2].set_title(
                f"V2 Prediction (T={threshold:.2f})"
            )

            im = axes[3].imshow(
                probability_map,
                cmap="viridis",
                vmin=0,
                vmax=1
            )
            axes[3].set_title(
                "Water Probability"
            )

            # Mark unlabeled areas in probability map
            axes[3].imshow(
                np.ma.masked_where(
                    valid,
                    np.ones_like(valid, dtype=float)
                ),
                cmap="gray",
                vmin=0,
                vmax=1,
                alpha=0.35
            )

            fig.colorbar(
                im,
                ax=axes[3],
                fraction=0.046,
                pad=0.04
            )

            for ax in axes:
                ax.axis("off")

            fig.suptitle(
                f"Test Sample {sample_index} | "
                f"Precision: {precision:.3f} | "
                f"Recall: {recall:.3f} | "
                f"F1: {f1:.3f}",
                fontsize=12
            )

            fig.tight_layout()

            output_path = (
                OUTPUT_DIR
                / f"rgb_v2_sample_{sample_index:03d}.png"
            )

            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close(fig)

            saved_count += 1

            print(
                f"Saved: {output_path.name} | "
                f"TP={tp:,}, FP={fp:,}, FN={fn:,}"
            )

            if saved_count >= NUM_SAMPLES:
                break

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED")
    print("=" * 60)
    print(f"Images saved: {saved_count}")
    print(f"Samples skipped: {skipped_count}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
