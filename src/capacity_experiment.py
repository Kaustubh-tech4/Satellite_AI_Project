
import os
import csv
import random
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_CSV = os.path.join(
    PROJECT_DIR, "results", "mission_priority.csv"
)

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CSV_OUTPUT = os.path.join(
    RESULTS_DIR, "capacity_experiment.csv"
)
PLOT_OUTPUT = os.path.join(
    RESULTS_DIR, "capacity_comparison.png"
)

CAPACITIES = [10, 20, 30, 40, 50]
RANDOM_TRIALS = 100
RANDOM_SEED = 42


def load_images():
    with open(
        INPUT_CSV, "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("mission_priority.csv has no records.")

    for row in rows:
        row["priority_score"] = float(row["priority_score"])

    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    images = load_images()

    # Ensure priority-based selection is sorted by score.
    ranked = sorted(
        images,
        key=lambda row: row["priority_score"],
        reverse=True,
    )

    output_rows = []

    print("\n========== CAPACITY EXPERIMENT ==========")
    print(f"Available images: {len(images)}")
    print(f"Random trials per capacity: {RANDOM_TRIALS}")

    for capacity_requested in CAPACITIES:
        capacity = min(capacity_requested, len(images))

        # Deterministic priority-based selection.
        priority_selected = ranked[:capacity]
        priority_total = sum(
            row["priority_score"] for row in priority_selected
        )
        priority_mean = priority_total / capacity

        random_totals = []

        # Use independent, reproducible random trials.
        for trial in range(RANDOM_TRIALS):
            rng = random.Random(
                RANDOM_SEED + capacity_requested * 1000 + trial
            )
            selected = rng.sample(images, capacity)
            random_totals.append(
                sum(row["priority_score"] for row in selected)
            )

        random_mean_total = sum(random_totals) / RANDOM_TRIALS
        random_std_total = (
            sum(
                (score - random_mean_total) ** 2
                for score in random_totals
            ) / RANDOM_TRIALS
        ) ** 0.5

        improvement_pct = (
            (priority_total - random_mean_total)
            / random_mean_total * 100
            if random_mean_total else 0.0
        )

        output_rows.append({
            "capacity": capacity,
            "priority_total_score": round(priority_total, 4),
            "priority_mean_score": round(priority_mean, 4),
            "random_mean_total_score": round(random_mean_total, 4),
            "random_std_total_score": round(random_std_total, 4),
            "relative_score_improvement_pct": round(
                improvement_pct, 4
            ),
            "random_trials": RANDOM_TRIALS,
        })

        print(
            f"Capacity {capacity:2d} | "
            f"Priority total: {priority_total:.2f} | "
            f"Random mean: {random_mean_total:.2f} | "
            f"Random SD: {random_std_total:.2f} | "
            f"Score difference: {improvement_pct:.2f}%"
        )

    with open(
        CSV_OUTPUT, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    capacities = [row["capacity"] for row in output_rows]
    priority_scores = [
        row["priority_total_score"] for row in output_rows
    ]
    random_scores = [
        row["random_mean_total_score"] for row in output_rows
    ]
    random_std = [
        row["random_std_total_score"] for row in output_rows
    ]

    plt.figure(figsize=(9, 5))
    plt.plot(
        capacities, priority_scores,
        marker="o", label="Priority-based"
    )
    plt.plot(
        capacities, random_scores,
        marker="s", label="Random baseline (mean of 100 trials)"
    )
    plt.fill_between(
        capacities,
        [m - s for m, s in zip(random_scores, random_std)],
        [m + s for m, s in zip(random_scores, random_std)],
        alpha=0.2,
        label="Random baseline ± 1 SD",
    )

    plt.xlabel("Downlink capacity (number of images)")
    plt.ylabel("Total selected priority score")
    plt.title("Priority-Based Selection vs Random Baseline")
    plt.xticks(capacities)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=200)
    plt.close()

    print(f"\nResults CSV: {CSV_OUTPUT}")
    print(f"Comparison chart: {PLOT_OUTPUT}")
    print("Capacity experiment completed.")


if __name__ == "__main__":
    main()
