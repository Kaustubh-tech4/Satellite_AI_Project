
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from satellite_dataset import SatelliteFloodDataset
from unet_model import UNet


PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
EPOCHS = 12
BATCH_SIZE = 1
LEARNING_RATE = 5e-4
POS_WEIGHT = 5.0
PATIENCE = 4
THRESHOLD = 0.5


def calculate_loss(logits, labels, valid_mask, criterion):
    labels = labels.unsqueeze(1).float()
    valid = valid_mask.unsqueeze(1).float()

    pixel_losses = criterion(logits, labels)
    bce = (pixel_losses * valid).sum() / valid.sum().clamp_min(1.0)

    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * labels * valid).sum()
    denominator = (
        (probabilities * valid).sum()
        + (labels * valid).sum()
    )

    dice_loss = 1.0 - (
        (2.0 * intersection + 1e-6) /
        (denominator + 1e-6)
    )

    return bce + dice_loss


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    tp = fp = fn = 0

    with torch.no_grad():
        for images, labels, valid_mask in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).bool()
            valid = valid_mask.to(DEVICE).bool()

            logits = model(images)
            loss = calculate_loss(
                logits, labels, valid, criterion
            )
            total_loss += loss.item()

            predictions = torch.sigmoid(logits[:, 0]) >= THRESHOLD

            tp += int((predictions & labels & valid).sum())
            fp += int((predictions & ~labels & valid).sum())
            fn += int((~predictions & labels & valid).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return (
        total_loss / max(len(loader), 1),
        iou,
        precision,
        recall,
        f1,
    )


def main():
    torch.manual_seed(42)

    print("Loading augmented training dataset...")
    train_dataset = SatelliteFloodDataset("train", augment=True)
    val_dataset = SatelliteFloodDataset("val", augment=False)

    generator = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(in_channels=15, base_channels=16).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE),
        reduction="none",
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1,
        min_lr=1e-5,
    )

    checkpoint_path = MODEL_DIR / "unet_augmented_best.pth"
    best_f1 = -1.0
    epochs_without_improvement = 0

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Device:", DEVICE)
    print("Starting augmented training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for step, (images, labels, valid_mask) in enumerate(
            train_loader, start=1
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            valid = valid_mask.to(DEVICE).bool()

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = calculate_loss(
                logits, labels, valid, criterion
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)

        val_loss, iou, precision, recall, f1 = evaluate(
            model, val_loader, criterion
        )

        scheduler.step(f1)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch}: "
            f"Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"IoU={iou:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}, "
            f"LR={current_lr:.6f}\n"
        )

        if f1 > best_f1:
            best_f1 = f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_f1": f1,
                    "threshold": THRESHOLD,
                    "input_channels": 15,
                },
                checkpoint_path,
            )

            print("Best model saved:", checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation F1 did not improve.")
            break

    print("\nAugmented training finished.")
    print(f"Best validation F1: {best_f1:.4f}")
    print("Checkpoint:", checkpoint_path)
    print("Compare this run with your original model.")


if __name__ == "__main__":
    main()
