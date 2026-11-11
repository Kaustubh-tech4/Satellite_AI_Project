
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from unified_model import UnifiedWaterUNet
from unified_dataset import UnifiedSatelliteDataset


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_PATH = MODEL_DIR / "unified_water_unet_best.pth"
LAST_PATH = MODEL_DIR / "unified_water_unet_last.pth"

DEVICE = torch.device("cpu")

MODALITIES = ("sar", "optical", "fused")

EPOCHS = 10
LEARNING_RATE = 0.0001

# CPU/RAM-friendly initial experiment.
TRAIN_SAMPLES_PER_MODALITY = 84
VALIDATION_SAMPLES_PER_MODALITY = 15

THRESHOLD = 0.40
DICE_WEIGHT = 0.5
PATIENCE = 4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)


# ============================================================
# LOSS FUNCTION
# ============================================================

def masked_bce_dice_loss(logits, targets, valid_mask):
    """
    Calculate BCE + Dice loss using only valid label pixels.
    """
    valid = valid_mask.bool()

    if not valid.any().item():
        return None

    valid_logits = logits[valid]
    valid_targets = targets[valid]

    bce = F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
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

    return bce + DICE_WEIGHT * (1.0 - dice)


# ============================================================
# DATASETS
# ============================================================

def create_datasets():
    train_datasets = {}
    val_datasets = {}

    for modality in MODALITIES:
        print(f"\nLoading {modality} datasets...")

        train_datasets[modality] = UnifiedSatelliteDataset(
            split="train",
            modality=modality,
            augment=True,
        )

        val_datasets[modality] = UnifiedSatelliteDataset(
            split="val",
            modality=modality,
            augment=False,
        )

    return train_datasets, val_datasets


def choose_validation_indices(dataset, count):
    """
    Use a deterministic, evenly spaced subset for quick validation.
    Full validation should be run after training.
    """
    count = min(count, len(dataset))

    if count == len(dataset):
        return list(range(len(dataset)))

    return np.linspace(
        0,
        len(dataset) - 1,
        num=count,
        dtype=int,
    ).tolist()


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, train_datasets, optimizer, epoch):
    model.train()

    total_loss = 0.0
    successful_steps = 0

    for modality in MODALITIES:
        dataset = train_datasets[modality]

        sample_count = min(
            TRAIN_SAMPLES_PER_MODALITY,
            len(dataset),
        )

        indices = random.sample(
            range(len(dataset)),
            sample_count,
        )

        print(
            f"  Training {modality}: "
            f"{sample_count} samples"
        )

        for step, index in enumerate(indices, start=1):
            image, labels, valid, _ = dataset[index]

            # Add batch dimension.
            image = image.unsqueeze(0).to(DEVICE)
            labels = labels.unsqueeze(0).to(DEVICE)
            valid = valid.unsqueeze(0).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(image, modality)

            loss = masked_bce_dice_loss(
                logits,
                labels,
                valid,
            )

            if loss is None:
                continue

            if not torch.isfinite(loss).item():
                print(
                    f"WARNING: Non-finite loss; "
                    f"skipping {modality} sample {index}."
                )
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            successful_steps += 1

            if step % 20 == 0 or step == sample_count:
                print(
                    f"    {modality}: "
                    f"{step}/{sample_count} samples"
                )

    average_loss = (
        total_loss / successful_steps
        if successful_steps
        else float("nan")
    )

    return average_loss


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_modality(model, dataset, modality, indices):
    model.eval()

    total_loss = 0.0
    loss_count = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for index in indices:
        image, labels, valid, _ = dataset[index]

        image = image.unsqueeze(0).to(DEVICE)
        labels = labels.unsqueeze(0).to(DEVICE)
        valid = valid.unsqueeze(0).to(DEVICE).bool()

        logits = model(image, modality)

        loss = masked_bce_dice_loss(
            logits,
            labels,
            valid,
        )

        if loss is not None:
            total_loss += loss.item()
            loss_count += 1

        predictions = torch.sigmoid(logits) >= THRESHOLD
        targets = labels >= 0.5

        tp += int((predictions & targets & valid).sum().item())
        fp += int((predictions & ~targets & valid).sum().item())
        fn += int((~predictions & targets & valid).sum().item())
        tn += int((~predictions & ~targets & valid).sum().item())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn
        else 0.0
    )

    return {
        "loss": total_loss / loss_count if loss_count else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "valid_pixels": tp + fp + fn + tn,
    }


def validate_all(model, val_datasets):
    results = {}

    for modality in MODALITIES:
        dataset = val_datasets[modality]

        indices = choose_validation_indices(
            dataset,
            VALIDATION_SAMPLES_PER_MODALITY,
        )

        metrics = evaluate_modality(
            model,
            dataset,
            modality,
            indices,
        )

        results[modality] = metrics

        print(
            f"  {modality:8s} | "
            f"F1: {metrics['f1']:.4f} | "
            f"IoU: {metrics['iou']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Valid pixels: {metrics['valid_pixels']}"
        )

    mean_f1 = float(np.mean([
        results[m]["f1"] for m in MODALITIES
    ]))

    return results, mean_f1


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("UNIFIED SATELLITE WATER SEGMENTATION TRAINING")
    print("=" * 65)

    print("Device:", DEVICE)
    print("Modalities:", MODALITIES)
    print("Epochs:", EPOCHS)
    print("Train samples per modality per epoch:",
          TRAIN_SAMPLES_PER_MODALITY)
    print("Validation subset per modality:",
          VALIDATION_SAMPLES_PER_MODALITY)
    print("Best checkpoint:", BEST_PATH)
    print("Last checkpoint:", LAST_PATH)

    train_datasets, val_datasets = create_datasets()

    model = UnifiedWaterUNet(base_channels=16).to(DEVICE)

    print(
        "\nTrainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_mean_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        print(f"\n{'=' * 65}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 65)

        train_loss = train_one_epoch(
            model,
            train_datasets,
            optimizer,
            epoch,
        )

        print(f"\nAverage training loss: {train_loss:.4f}")
        print("Validation metrics:")

        val_results, mean_f1 = validate_all(
            model,
            val_datasets,
        )

        print(f"\nMean validation F1: {mean_f1:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mean_validation_f1": mean_f1,
            "validation_results": val_results,
            "modalities": list(MODALITIES),
            "threshold": THRESHOLD,
            "base_channels": 16,
            "seed": SEED,
        }

        # Always preserve the latest training state.
        torch.save(checkpoint, LAST_PATH)

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            epochs_without_improvement = 0

            torch.save(checkpoint, BEST_PATH)

            print("Best checkpoint updated:", BEST_PATH)

        else:
            epochs_without_improvement += 1

        elapsed = time.time() - start_time

        print(f"Epoch duration: {elapsed / 60:.1f} minutes")
        print(f"Best mean validation F1: {best_mean_f1:.4f}")

        if epochs_without_improvement >= PATIENCE:
            print(
                "\nEarly stopping: validation F1 did not improve "
                f"for {PATIENCE} consecutive epochs."
            )
            break

    print("\n" + "=" * 65)
    print("TRAINING FINISHED")
    print("=" * 65)
    print("Best checkpoint:", BEST_PATH)
    print("Latest checkpoint:", LAST_PATH)
    print(f"Best mean validation F1: {best_mean_f1:.4f}")
    print(
        "\nNote: validation uses a subset of samples. "
        "Run a separate full validation and held-out test "
        "before reporting final performance."
    )


if __name__ == "__main__":
    main()
