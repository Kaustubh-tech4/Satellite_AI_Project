
import os
import csv
import random

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_CSV = os.path.join(
    PROJECT_DIR, "results", "mission_priority.csv"
)

OUTPUT_CSV = os.path.join(
    PROJECT_DIR, "results", "downlink_selection.csv"
)

# Simulation settings
DOWNLINK_CAPACITY = 20
RANDOM_SEED = 42


def load_images(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Priority file not found: {path}"
        )

    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        images = list(reader)

    required_columns = {
        "sample_id",
        "priority_score",
        "predicted_water_pct",
        "mean_entropy",
    }

    if not images:
        raise ValueError("The priority CSV contains no image records.")

    missing = required_columns - set(images[0].keys())
    if missing:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing)}"
        )

    for image in images:
        image["priority_score"] = float(image["priority_score"])
        image["predicted_water_pct"] = float(
            image["predicted_water_pct"]
        )
        image["mean_entropy"] = float(image["mean_entropy"])

    return images


def select_images(images, capacity, strategy):
    capacity = min(capacity, len(images))

    if strategy == "priority":
        ordered = sorted(
            images,
            key=lambda item: item["priority_score"],
            reverse=True,
        )
        return ordered[:capacity]

    if strategy == "random":
        rng = random.Random(RANDOM_SEED)
        return rng.sample(images, capacity)

    raise ValueError(f"Unknown selection strategy: {strategy}")


def summarize(selected):
    count = len(selected)

    if count == 0:
        return {
            "images_transmitted": 0,
            "total_priority_score": 0.0,
            "mean_priority_score": 0.0,
            "mean_predicted_water_pct": 0.0,
            "mean_entropy": 0.0,
        }

    return {
        "images_transmitted": count,
        "total_priority_score": round(
            sum(x["priority_score"] for x in selected), 4
        ),
        "mean_priority_score": round(
            sum(x["priority_score"] for x in selected) / count, 4
        ),
        "mean_predicted_water_pct": round(
            sum(x["predicted_water_pct"] for x in selected) / count, 4
        ),
        "mean_entropy": round(
            sum(x["mean_entropy"] for x in selected) / count, 6
        ),
    }


def save_selection(priority_selected, random_selected, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fields = [
        "strategy",
        "rank_within_selection",
        "sample_id",
        "priority_score",
        "predicted_water_pct",
        "mean_entropy",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for strategy, selected in [
            ("priority", priority_selected),
            ("random_baseline", random_selected),
        ]:
            for rank, item in enumerate(selected, start=1):
                writer.writerow({
                    "strategy": strategy,
                    "rank_within_selection": rank,
                    "sample_id": item["sample_id"],
                    "priority_score": item["priority_score"],
                    "predicted_water_pct": item["predicted_water_pct"],
                    "mean_entropy": item["mean_entropy"],
                })


def print_summary(title, summary):
    print(f"\n{title}")
    for key, value in summary.items():
        print(f"{key}: {value}")


def main():
    images = load_images(INPUT_CSV)
    capacity = min(DOWNLINK_CAPACITY, len(images))

    priority_selected = select_images(
        images, capacity, "priority"
    )

    random_selected = select_images(
        images, capacity, "random"
    )

    priority_summary = summarize(priority_selected)
    random_summary = summarize(random_selected)

    print("\n========== DOWNLINK SIMULATION ==========")
    print(f"Available images: {len(images)}")
    print(f"Downlink capacity: {capacity} equal-sized images")
    print(f"Random seed: {RANDOM_SEED}")

    print_summary(
        "PRIORITY-BASED SELECTION",
        priority_summary
    )

    print_summary(
        "RANDOM-SELECTION BASELINE",
        random_summary
    )

    score_difference = (
        priority_summary["total_priority_score"]
        - random_summary["total_priority_score"]
    )

    print("\n========== COMPARISON ==========")
    print(
        "Priority-score difference "
        "(priority minus random): "
        f"{score_difference:.4f}"
    )

    if random_summary["total_priority_score"] > 0:
        improvement = (
            score_difference
            / random_summary["total_priority_score"]
        ) * 100
        print(f"Relative score improvement: {improvement:.2f}%")
    else:
        print("Relative score improvement: not defined")

    priority_ids = {
        item["sample_id"] for item in priority_selected
    }
    random_ids = {
        item["sample_id"] for item in random_selected
    }

    overlap = len(priority_ids & random_ids)
    print(f"Images selected by both strategies: {overlap}/{capacity}")

    save_selection(
        priority_selected,
        random_selected,
        OUTPUT_CSV
    )

    print(f"\nSelection details saved to: {OUTPUT_CSV}")
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()
