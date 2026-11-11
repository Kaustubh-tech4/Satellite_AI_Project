
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rgb_flood_dataset import RGBFloodDataset
from rgb_unet_model import RGBUNet


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "rgb_unet_best.pth"
LAST_MODEL_PATH = MODEL_DIR / "rgb_unet_last.pth"

DEVICE = torch.device("cpu")

EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
BASE_CHANNELS = 8
NUM_WORKERS = 0

SEED = 42

# A sample with no valid labels is excluded from training.
# Validation samples with no valid labels are skipped.


# ==================================================
# REPRODUCIBILITY
# ==================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(2)


# ==================================================
# MASKED BINARY CROSS-ENTROPY + DICE LOSS
# ==================================================

class MaskedBCEDiceLoss(nn.Module):

    def __init__(self, pos_weight=3.0, dice_weight=0.5):
        super().__init__()

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32),
        )

        self.dice_weight = dice_weight

    def forward(self, logits, targets, valid_mask):

        # logits, targets, valid_mask:
        # (N, 1, H, W)

        valid = valid_mask.bool()

        if not valid.any():
            return None

        valid_logits = logits[valid]
        valid_targets = targets[valid]

        bce = nn.functional.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            pos_weight=self.pos_weight,
        )

        probabilities = torch.sigmoid(valid_logits)

        intersection = (
            probabilities * valid_targets
        ).sum()

        dice = (
            2.0 * intersection + 1e-6
        ) / (
            probabilities.sum()
            + valid_targets.sum()
            + 1e-6
        )

        return bce + self.dice_weight * (1.0 - dice)


# ==================================================
# DATA LOADING
# ==================================================

def create_datasets():

    print("\nLoading datasets...")

    train_dataset = RGBFloodDataset(
        split="train",
        augment=True,
    )

    val_dataset = RGBFloodDataset(
        split="val",
        augment=False,
    )

    # Find training samples with at least one valid label.
    # This prevents empty-label samples from entering training.
    usable_indices = []

    print("Checking training labels...")

    for index in range(len(train_dataset)):

        _, _, valid = train_dataset[index]

        if valid.any().item():
            usable_indices.append(index)

    if not usable_indices:
        raise RuntimeError(
            "No training samples contain valid labels."
        )

    print(f"Training samples available: {len(train_dataset)}")
    print(f"Usable training samples: {len(usable_indices)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset, usable_indices


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(tp, fp, fn, tn):

    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = (
        2.0 * precision * recall
        / (precision + recall + eps)
    )

    iou = tp / (tp + fp + fn + eps)

    accuracy = (
        tp + tn
    ) / (
        tp + fp + fn + tn + eps
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }


# ==================================================
# VALIDATION
# ==================================================

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    loss_batches = 0

    tp = fp = fn = tn = 0
    skipped = 0

    with torch.no_grad():

        for images, labels, valid in loader:

            images = images.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)
            valid = valid.unsqueeze(1).to(DEVICE).bool()

            # Avoid wasting computation on entirely unlabeled images.
            if not valid.any().item():
                skipped += 1
                continue

            logits = model(images)

            loss = criterion(logits, labels, valid)

            if loss is not None:
                total_loss += loss.item()
                loss_batches += 1

            predictions = torch.sigmoid(logits) >= 0.5
            targets = labels >= 0.5

            tp += (predictions & targets & valid).sum().item()
            fp += (predictions & ~targets & valid).sum().item()
            fn += (~predictions & targets & valid).sum().item()
            tn += (~predictions & ~targets & valid).sum().item()

    if loss_batches == 0:
        raise RuntimeError(
            "Validation has no valid labeled pixels. "
            "Inspect the label files before training."
        )

    metrics = calculate_metrics(tp, fp, fn, tn)

    return (
        total_loss / loss_batches,
        metrics,
        skipped,
    )


# ==================================================
# TRAINING
# ==================================================

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0.0
    processed_batches = 0
    skipped_batches = 0

    for step, (images, labels, valid) in enumerate(loader, start=1):

        images = images.to(DEVICE)
        labels = labels.unsqueeze(1).to(DEVICE)
        valid = valid.unsqueeze(1).to(DEVICE).bool()

        if not valid.any().item():
            skipped_batches += 1
            continue

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = criterion(logits, labels, valid)

        if loss is None or not torch.isfinite(loss).item():
            skipped_batches += 1
            continue

        loss.backward()

        # Helps avoid unstable gradient updates.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        processed_batches += 1

        if step % 25 == 0:
            print(
                f"  Batch {step}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    if processed_batches == 0:
        raise RuntimeError(
            "No training batches were processed. "
            "Check your labels and dataset."
        )

    return (
        total_loss / processed_batches,
        processed_batches,
        skipped_batches,
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("RGB U-NET FLOOD SEGMENTATION TRAINING")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Base channels: {BASE_CHANNELS}")

    train_dataset, val_dataset, usable_indices = (
        create_datasets()
    )

    # Subset ensures samples with no valid labels are excluded.
    from torch.utils.data import Subset

    train_subset = Subset(
        train_dataset,
        usable_indices,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = RGBUNet(
        in_channels=3,
        base_channels=BASE_CHANNELS,
    ).to(DEVICE)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"\nModel parameters: {total_params:,}")

    # Initial class weighting. Can be tuned using training statistics.
    criterion = MaskedBCEDiceLoss(
        pos_weight=3.0,
        dice_weight=0.5,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    best_val_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):

        start_time = time.time()

        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 60)

        train_loss, processed, skipped = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, metrics, val_skipped = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(metrics["f1"])

        elapsed = time.time() - start_time

        print(f"\nTraining loss:   {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Water IoU:       {metrics['iou']:.4f}")
        print(f"Water precision: {metrics['precision']:.4f}")
        print(f"Water recall:    {metrics['recall']:.4f}")
        print(f"Water F1-score:  {metrics['f1']:.4f}")
        print(f"Pixel accuracy:  {metrics['accuracy']:.4f}")
        print(f"Training batches processed: {processed}")
        print(f"Training batches skipped: {skipped}")
        print(f"Validation samples skipped: {val_skipped}")
        print(f"Epoch duration: {elapsed / 60:.2f} minutes")

        # Save the most recent checkpoint.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_f1": metrics["f1"],
                "base_channels": BASE_CHANNELS,
                "in_channels": 3,
                "input_normalization": "Sentinel-2 reflectance / 10000",
            },
            LAST_MODEL_PATH,
        )

        # Select best checkpoint using validation water F1.
        if metrics["f1"] > best_val_f1:

            best_val_f1 = metrics["f1"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_f1": best_val_f1,
                    "base_channels": BASE_CHANNELS,
                    "in_channels": 3,
                    "input_normalization": "Sentinel-2 reflectance / 10000",
                },
                BEST_MODEL_PATH,
            )

            print("New best model saved!")

        print(f"Best validation F1 so far: {best_val_f1:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best checkpoint: {BEST_MODEL_PATH}")
    print(f"Last checkpoint: {LAST_MODEL_PATH}")
    print("\nExisting 15-channel model was not modified.")


if __name__ == "__main__":
    main()
