
import os
import sys
import subprocess

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(PROJECT_DIR, "src")

PIPELINE_STEPS = [
    ("AI analysis and mission prioritization",
     "mission_priority.py"),
    ("Bandwidth-limited downlink simulation",
     "downlink_simulator.py"),
]


def run_step(title, script_name):
    script_path = os.path.join(SRC_DIR, script_name)

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    if not os.path.isfile(script_path):
        raise FileNotFoundError(
            f"Required script not found: {script_path}"
        )

    subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_DIR,
        check=True,
    )


def main():
    print("AI SATELLITE INTELLIGENCE SIMULATION")
    print(f"Project directory: {PROJECT_DIR}")

    try:
        for title, script_name in PIPELINE_STEPS:
            run_step(title, script_name)

        priority_csv = os.path.join(
            PROJECT_DIR, "results", "mission_priority.csv"
        )
        downlink_csv = os.path.join(
            PROJECT_DIR, "results", "downlink_selection.csv"
        )

        print("\n" + "=" * 65)
        print("MISSION PIPELINE SUMMARY")
        print("=" * 65)
        print(f"Priority records: {priority_csv}")
        print(f"Downlink records: {downlink_csv}")
        print("All pipeline stages completed successfully.")

    except subprocess.CalledProcessError as error:
        print(
            f"\nPipeline stopped because "
            f"{error.cmd[-1]} failed."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
