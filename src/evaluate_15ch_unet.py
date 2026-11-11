
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from satellite_dataset import SatelliteFloodDataset
from unet_model import UNet

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
CHECKPOINT = PROJECT_DIR / "models" / "unet_augmented_best.pth"
DEVICE = torch.device("cpu")


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

    model = UNet(in_channels=15, base_channels=16).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SatelliteFloodDataset("test", augment=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    tp = fp = fn = tn = 0
    skipped = 0
    valid_pixels = 0

    # Use the saved threshold, or 0.5 if it is absent.
    threshold = float(checkpoint.get("threshold", 0.5))

    with torch.no_grad():
        for images, labels, valid in loader:
            if not valid.any().item():
                skipped += 1
                continue

            logits = model(images.to(DEVICE))
            predictions = torch.sigmoid(logits[:, 0]) >= threshold
            targets = labels.to(DEVICE).bool()
            mask = valid.to(DEVICE).bool()

            tp += int((predictions & targets & mask).sum().item())
            fp += int((predictions & ~targets & mask).sum().item())
            fn += int((~predictions & targets & mask).sum().item())
            tn += int((~predictions & ~targets & mask).sum().item())
            valid_pixels += int(mask.sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / max(valid_pixels, 1)

    print("\nORIGINAL 15-CHANNEL U-NET TEST RESULTS")
    print("Checkpoint:", CHECKPOINT)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Threshold:", threshold)
    print("Samples:", len(dataset))
    print("Skipped empty-label samples:", skipped)
    print("Valid pixels:", f"{valid_pixels:,}")
    print("TP:", f"{tp:,}", "| FP:", f"{fp:,}")
    print("FN:", f"{fn:,}", "| TN:", f"{tn:,}")
    print(f"Water precision: {precision:.4%}")
    print(f"Water recall:    {recall:.4%}")
    print(f"Water F1-score:  {f1:.4%}")
    print(f"Water IoU:       {iou:.4%}")
    print(f"Pixel accuracy:  {accuracy:.4%}")
    print("\nCheckpoint was only read, not modified.")


if __name__ == "__main__":
    main()
