
import os
import csv
import numpy as np
import torch

from satellite_dataset import SatelliteFloodDataset
from unet_model import UNet


PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR, "models", "unet_augmented_best.pth"
)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "mission_priority.csv")

THRESHOLD = 0.40
HIGH_UNCERTAINTY_THRESHOLD = 0.80
DEVICE = torch.device("cpu")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    print("Loading trained model...")
    dataset = SatelliteFloodDataset("test", augment=False)

    model = UNet(in_channels=15, base_channels=16).to(DEVICE)
    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []

    for index in range(len(dataset)):
        image, labels, valid_mask = dataset[index]
        valid = valid_mask.cpu().numpy().astype(bool)

        if not valid.any():
            print(f"Skipping sample {index}: no valid pixels")
            continue

        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(DEVICE))
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

        predictions = probabilities >= THRESHOLD

        # Normalized binary predictive entropy (0 to 1)
        p = np.clip(probabilities, 1e-7, 1 - 1e-7)
        entropy = -(
            p * np.log2(p) + (1 - p) * np.log2(1 - p)
        )

        valid_count = int(valid.sum())
        predicted_water_pct = float(
            predictions[valid].mean() * 100
        )
        mean_entropy = float(entropy[valid].mean())
        high_uncertainty_pct = float(
            (entropy[valid] >= HIGH_UNCERTAINTY_THRESHOLD).mean() * 100
        )

        # Preliminary heuristic score, not a validated operational score.
        # Water coverage and uncertainty each contribute 50%.
        priority_score = (
            0.5 * predicted_water_pct
            + 0.5 * (mean_entropy * 100)
        )

        sample_ids = getattr(dataset, "sample_ids", None)
        sample_id = (
            str(sample_ids[index])
            if sample_ids is not None
            else f"test_sample_{index + 1:03d}"
        )

        results.append({
            "rank": 0,
            "sample_id": sample_id,
            "predicted_water_pct": round(predicted_water_pct, 4),
            "mean_entropy": round(mean_entropy, 6),
            "high_uncertainty_pct": round(high_uncertainty_pct, 4),
            "valid_pixels": valid_count,
            "priority_score": round(priority_score, 4),
        })

        print(
            f"[{index + 1}/{len(dataset)}] {sample_id} | "
            f"Water: {predicted_water_pct:.2f}% | "
            f"Entropy: {mean_entropy:.3f}"
        )

    results.sort(
        key=lambda item: item["priority_score"],
        reverse=True
    )

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    fieldnames = [
        "rank",
        "sample_id",
        "predicted_water_pct",
        "mean_entropy",
        "high_uncertainty_pct",
        "valid_pixels",
        "priority_score",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n--- Mission Priority Summary ---")
    print(f"Images ranked: {len(results)}")
    print(f"CSV saved to: {CSV_PATH}")

    print("\nTop 10 images to inspect first:")
    for item in results[:10]:
        print(
            f"Rank {item['rank']:>2}: {item['sample_id']} | "
            f"Score {item['priority_score']:.2f} | "
            f"Water {item['predicted_water_pct']:.2f}% | "
            f"Uncertainty {item['mean_entropy']:.3f}"
        )

    print("\nMission-priority analysis completed.")


if __name__ == "__main__":
    main()
