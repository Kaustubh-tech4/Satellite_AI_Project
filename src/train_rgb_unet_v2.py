
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
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = MODEL_DIR / "rgb_unet_v2_best.pth"
LAST_PATH = MODEL_DIR / "rgb_unet_v2_last.pth"

DEVICE = torch.device("cpu")

EPOCHS = 15
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
BASE_CHANNELS = 8

# Select checkpoints using the validation threshold
VAL_THRESHOLD = 0.3

# Stop if validation F1 fails to improve for this many epochs
PATIENCE = 4

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)


# ==================================================
# MASKED BCE + DICE LOSS
# ==================================================

class MaskedBCEDiceLoss(nn.Module):

    def __init__(self, pos_weight=1.5, dice_weight=0.5):
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
        valid_targets = targets[valid]

        bce = nn.functional.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            pos_weight=self.pos_weight,
        )

        probabilities = torch.sigmoid(valid_logits)

        intersection = (probabilities * valid_targets).sum()

        dice = (
            2.0 * intersection + 1e-6
        ) / (
            probabilities.sum()
            + valid_targets.sum()
            + 1e-6
        )

        return bce + self.dice_weight * (1.0 - dice)


# ==================================================
# DATASETS
# ==================================================

def prepare_datasets():

    print("\nLoading training and validation datasets...")

    train_dataset = RGBFloodDataset(
        split="train",
        augment=True,
    )

    val_dataset = RGBFloodDataset(
        split="val",
        augment=False,
    )

    # Exclude images with no valid labels.
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


# ==================================================
# VALIDATION METRICS
# ==================================================

def calculate_metrics(tp, fp, fn, tn):

    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }


# ==================================================
# ONE TRAINING EPOCH
# ==================================================

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0.0
    processed = 0
    skipped = 0

    for step, (images, labels, valid) in enumerate(loader, start=1):

        images = images.to(DEVICE)
        labels = labels.unsqueeze(1).to(DEVICE)
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

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        processed += 1

        if step % 25 == 0:
            print(
                f"  Batch {step}/{len(loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    if processed == 0:
        raise RuntimeError("No valid training batches were processed.")

    return total_loss / processed, processed, skipped


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

            if not valid.any().item():
                skipped += 1
                continue

            logits = model(images)
            loss = criterion(logits, labels, valid)

            if loss is not None:
                total_loss += loss.item()
                loss_batches += 1

            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= VAL_THRESHOLD
            targets = labels >= 0.5

            tp += (predictions & targets & valid).sum().item()
            fp += (predictions & ~targets & valid).sum().item()
            fn += (~predictions & targets & valid).sum().item()
            tn += (~predictions & ~targets & valid).sum().item()

    if loss_batches == 0:
        raise RuntimeError("Validation contains no valid labeled pixels.")

    metrics = calculate_metrics(tp, fp, fn, tn)

    return total_loss / loss_batches, metrics, skipped


# ==================================================
# SAVE CHECKPOINT
# ==================================================

def save_checkpoint(path, model, optimizer, epoch, val_loss, metrics):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": val_loss,
            "validation_f1": metrics["f1"],
            "validation_threshold": VAL_THRESHOLD,
            "base_channels": BASE_CHANNELS,
            "in_channels": 3,
            "input_normalization": "Sentinel-2 reflectance / 10000",
            "experiment": "RGB U-Net V2",
        },
        path,
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("RGB U-NET V2 TRAINING")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Maximum epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Validation threshold: {VAL_THRESHOLD}")
    print(f"Early stopping patience: {PATIENCE}")

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

    model = RGBUNet(
        in_channels=3,
        base_channels=BASE_CHANNELS,
    ).to(DEVICE)

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    criterion = MaskedBCEDiceLoss(
        pos_weight=1.5,
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
        min_lr=1e-6,
    )

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):

        start = time.time()

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

        print(f"\nTraining loss:   {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation F1:   {metrics['f1']:.4f}")
        print(f"Water IoU:       {metrics['iou']:.4f}")
        print(f"Water precision: {metrics['precision']:.4f}")
        print(f"Water recall:    {metrics['recall']:.4f}")
        print(f"Pixel accuracy:  {metrics['accuracy']:.4f}")
        print(f"Training batches processed: {processed}")
        print(f"Training batches skipped: {skipped}")
        print(f"Validation samples skipped: {val_skipped}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.7f}")
        print(f"Epoch duration: {(time.time() - start) / 60:.2f} minutes")

        # Always save the latest completed epoch.
        save_checkpoint(
            LAST_PATH, model, optimizer, epoch, val_loss, metrics
        )

        # Save best checkpoint based on validation F1.
        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                BEST_PATH, model, optimizer, epoch, val_loss, metrics
            )

            print("Improved validation F1: best checkpoint saved.")

        else:
            epochs_without_improvement += 1
            print(
                f"No validation F1 improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping: validation F1 did not improve.")
            break

    print("\n" + "=" * 60)
    print("RGB U-NET V2 TRAINING FINISHED")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Best checkpoint: {BEST_PATH}")
    print(f"Last checkpoint: {LAST_PATH}")
    print("\nThe original RGB and 15-channel checkpoints were not modified.")


if __name__ == "__main__":
    main()
