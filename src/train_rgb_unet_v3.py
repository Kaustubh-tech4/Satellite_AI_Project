from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from rgb_flood_dataset import RGBFloodDataset
from rgb_unet_model import RGBUNet


# ==================================================
# RGB U-NET V3 CONFIGURATION
# This experiment uses separate checkpoints; V1, V2, and
# the existing 15-channel model are not overwritten.
# ==================================================

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = MODEL_DIR / "rgb_unet_v3_best.pth"
LAST_PATH = MODEL_DIR / "rgb_unet_v3_last.pth"

DEVICE = torch.device("cpu")
EPOCHS = 15
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
BASE_CHANNELS = 8
PATIENCE = 4
SEED = 42

# Choose a threshold using VALIDATION data only.
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

# V3 experiment: give positive water pixels more weight and
# increase the Dice contribution. Compare results rather than
# assuming this change will improve performance.
POS_WEIGHT = 2.0
DICE_WEIGHT = 1.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)


class MaskedBCEDiceLoss(nn.Module):
    """BCE + Dice loss, computed only over valid labeled pixels."""

    def __init__(self, pos_weight=POS_WEIGHT, dice_weight=DICE_WEIGHT):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32),
        )
        self.dice_weight = dice_weight

    def forward(self, logits, targets, valid_mask):
        valid = valid_mask.bool()
        if not valid.any().item():
            return None

        valid_logits = logits[valid]
        valid_targets = targets[valid].float()

        bce = nn.functional.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            pos_weight=self.pos_weight,
        )

        probabilities = torch.sigmoid(valid_logits)
        intersection = (probabilities * valid_targets).sum()
        dice = (2.0 * intersection + 1e-6) / (
            probabilities.sum() + valid_targets.sum() + 1e-6
        )
        return bce + self.dice_weight * (1.0 - dice)


def calculate_metrics(tp, fp, fn, tn):
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def prepare_datasets():
    print("\nLoading training and validation datasets...")

    train_dataset = RGBFloodDataset(split="train", augment=True)
    val_dataset = RGBFloodDataset(split="val", augment=False)

    # Keep only training samples containing at least one valid labeled pixel.
    # Do not inspect or tune on the test split here.
    usable_indices = []
    for index in range(len(train_dataset)):
        _, _, valid = train_dataset[index]
        if valid.any().item():
            usable_indices.append(index)

    if not usable_indices:
        raise RuntimeError("No training images have valid labels.")

    print(f"Original training samples: {len(train_dataset)}")
    print(f"Usable training samples: {len(usable_indices)}")
    print(f"Validation samples: {len(val_dataset)}")
    return train_dataset, val_dataset, usable_indices


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    processed = 0
    skipped = 0

    for step, (images, labels, valid) in enumerate(loader, start=1):
        images = images.to(DEVICE)
        labels = labels.unsqueeze(1).to(DEVICE).float()
        valid = valid.unsqueeze(1).to(DEVICE).bool()

        if not valid.any().item():
            skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels, valid)

        if loss is None or not torch.isfinite(loss).item():
            skipped += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        processed += 1

        if step % 25 == 0:
            print(f"  Batch {step}/{len(loader)} | Loss: {loss.item():.4f}")

    if processed == 0:
        raise RuntimeError("No valid training batches were processed.")

    return total_loss / processed, processed, skipped


def validate_and_select_threshold(model, loader, criterion):
    """Evaluate each threshold on the validation set and select the best F1."""
    model.eval()
    total_loss = 0.0
    loss_batches = 0
    skipped = 0

    counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for threshold in THRESHOLDS
    }

    with torch.no_grad():
        for images, labels, valid in loader:
            images = images.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE).float()
            valid = valid.unsqueeze(1).to(DEVICE).bool()

            if not valid.any().item():
                skipped += 1
                continue

            logits = model(images)
            loss = criterion(logits, labels, valid)
            if loss is not None and torch.isfinite(loss).item():
                total_loss += loss.item()
                loss_batches += 1

            probabilities = torch.sigmoid(logits)
            targets = labels >= 0.5

            for threshold in THRESHOLDS:
                predictions = probabilities >= threshold
                counts[threshold]["tp"] += (predictions & targets & valid).sum().item()
                counts[threshold]["fp"] += (predictions & ~targets & valid).sum().item()
                counts[threshold]["fn"] += (~predictions & targets & valid).sum().item()
                counts[threshold]["tn"] += (~predictions & ~targets & valid).sum().item()

    if loss_batches == 0:
        raise RuntimeError("Validation contains no valid labeled pixels.")

    metrics_by_threshold = {
        threshold: calculate_metrics(**counts[threshold])
        for threshold in THRESHOLDS
    }

    # Deterministic tie-break: prefer the higher threshold if F1 is equal.
    best_threshold = max(
        THRESHOLDS,
        key=lambda threshold: (metrics_by_threshold[threshold]["f1"], threshold),
    )

    return (
        total_loss / loss_batches,
        best_threshold,
        metrics_by_threshold[best_threshold],
        metrics_by_threshold,
        skipped,
    )


def save_checkpoint(path, model, optimizer, epoch, val_loss, threshold, metrics):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": val_loss,
            "validation_f1": metrics["f1"],
            "validation_threshold": threshold,
            "base_channels": BASE_CHANNELS,
            "in_channels": 3,
            "input_normalization": "Sentinel-2 reflectance / 10000",
            "experiment": "RGB U-Net V3: masked BCE + stronger Dice and validation threshold sweep",
            "pos_weight": POS_WEIGHT,
            "dice_weight": DICE_WEIGHT,
            "thresholds_evaluated": THRESHOLDS,
        },
        path,
    )


def main():
    print("=" * 64)
    print("RGB U-NET V3 TRAINING")
    print("=" * 64)
    print(f"Device: {DEVICE}")
    print(f"Maximum epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Positive-class weight: {POS_WEIGHT}")
    print(f"Dice loss weight: {DICE_WEIGHT}")
    print(f"Validation thresholds: {THRESHOLDS}")
    print(f"Early stopping patience: {PATIENCE}")
    print(f"Best checkpoint: {BEST_PATH}")
    print(f"Last checkpoint: {LAST_PATH}")

    train_dataset, val_dataset, usable_indices = prepare_datasets()

    train_loader = DataLoader(
        Subset(train_dataset, usable_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = RGBUNet(in_channels=3, base_channels=BASE_CHANNELS).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = MaskedBCEDiceLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6
    )

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        print(f"\n{'=' * 64}\nEPOCH {epoch}/{EPOCHS}\n{'=' * 64}")

        train_loss, processed, skipped = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, threshold, metrics, all_metrics, val_skipped = (
            validate_and_select_threshold(model, val_loader, criterion)
        )

        scheduler.step(metrics["f1"])

        print(f"\nTraining loss:   {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Best validation threshold: {threshold:.2f}")
        print(f"Validation F1:   {metrics['f1']:.4f}")
        print(f"Water IoU:       {metrics['iou']:.4f}")
        print(f"Water precision: {metrics['precision']:.4f}")
        print(f"Water recall:    {metrics['recall']:.4f}")
        print(f"Pixel accuracy:  {metrics['accuracy']:.4f}")
        print("Validation threshold sweep:")
        for tested_threshold in THRESHOLDS:
            tested = all_metrics[tested_threshold]
            print(
                f"  T={tested_threshold:.2f} | "
                f"F1={tested['f1']:.4f} | IoU={tested['iou']:.4f} | "
                f"P={tested['precision']:.4f} | R={tested['recall']:.4f}"
            )
        print(f"Training batches processed: {processed}")
        print(f"Training batches skipped: {skipped}")
        print(f"Validation samples skipped: {val_skipped}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.7f}")
        print(f"Epoch duration: {(time.time() - start) / 60:.2f} minutes")

        save_checkpoint(
            LAST_PATH, model, optimizer, epoch, val_loss, threshold, metrics
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                BEST_PATH, model, optimizer, epoch, val_loss, threshold, metrics
            )
            print("Improved validation F1: V3 best checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(
                f"No validation F1 improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping: validation F1 did not improve.")
            break

    print("\n" + "=" * 64)
    print("RGB U-NET V3 TRAINING FINISHED")
    print("=" * 64)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Best checkpoint: {BEST_PATH}")
    print(f"Last checkpoint: {LAST_PATH}")
    print("\nV1, V2, and the existing 15-channel checkpoint were not modified.")
    print("Use the saved validation threshold when evaluating V3.")


if __name__ == "__main__":
    main()
