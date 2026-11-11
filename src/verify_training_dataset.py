
from pathlib import Path

PROJECT_DIR = Path(r"D:\Satellite_AI_Project")
BASE_DIR = PROJECT_DIR / "data" / "raw" / "sen1floods11_v1.1"
DATA_DIR = BASE_DIR / "data"
SPLIT_DIR = BASE_DIR / "splits"

FOLDERS = {
    "S1": DATA_DIR / "S1GRDHand",
    "S2": DATA_DIR / "S2L1CHand",
    "Labels": DATA_DIR / "LabelHand",
}

SPLITS = {
    "Training": "flood_train_data.txt",
    "Validation": "flood_valid_data.txt",
    "Testing": "flood_test_data.txt",
}


def main():
    total = 0
    overall_missing = 0

    for split_name, filename in SPLITS.items():
        split_path = SPLIT_DIR / filename
        sample_ids = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        missing_count = 0
        print(f"\n{split_name}: {len(sample_ids)} samples")

        for sample_id in sample_ids:
            for data_type, folder in FOLDERS.items():
                expected = folder / f"{sample_id}_{data_type if data_type != 'Labels' else 'Label'}Hand.tif"

                if not expected.is_file():
                    print(f"MISSING: {expected}")
                    missing_count += 1

        total += len(sample_ids)
        overall_missing += missing_count

        print(f"Missing files: {missing_count}")

    print("\n" + "=" * 40)
    print("Total split entries:", total)
    print("Total missing files:", overall_missing)

    if overall_missing == 0:
        print("SUCCESS: All listed samples have matching files.")
    else:
        print("CHECK REQUIRED: Some files are missing.")


if __name__ == "__main__":
    main()
