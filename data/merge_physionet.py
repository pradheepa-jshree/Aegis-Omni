"""
data/merge_physionet.py

If you have the REAL PhysioNet 2019 Challenge dataset:
  https://physionet.org/content/challenge-2019/1.0.0/

1. Download training_setA.zip and training_setB.zip
2. Unzip both into  data/raw/training_setA/  and  data/raw/training_setB/
3. Run: python data/merge_physionet.py
4. Output: data/sepsis_dataset.csv  (pipe-delimited, ready for the pipeline)

Each .psv file = one patient.  This script:
  - Reads all .psv files from both sets
  - Adds a patient_id column
  - Concatenates into one big CSV
  - Saves as pipe-delimited (matches generate_physionet_synthetic.py format)
"""

import os
import glob
import pandas as pd

RAW_DIRS    = ["data/raw/training_setA", "data/raw/training_setB"]
OUTPUT_PATH = "data/sepsis_dataset.csv"


def merge():
    all_frames = []
    pid = 0

    for raw_dir in RAW_DIRS:
        if not os.path.exists(raw_dir):
            print(f"[Merge] Skipping {raw_dir} — not found")
            continue

        psv_files = sorted(glob.glob(os.path.join(raw_dir, "*.psv")))
        print(f"[Merge] {raw_dir}: found {len(psv_files)} patient files")

        for fpath in psv_files:
            df = pd.read_csv(fpath, sep="|")
            df.insert(0, "patient_id", pid)
            df["ICULOS"] = range(len(df))
            all_frames.append(df)
            pid += 1

    if not all_frames:
        raise FileNotFoundError(
            "No .psv files found. Download the dataset first:\n"
            "https://physionet.org/content/challenge-2019/1.0.0/"
        )

    merged = pd.concat(all_frames, ignore_index=True)
    merged.to_csv(OUTPUT_PATH, sep="|", index=False)
    print(f"[Merge] Done. {pid} patients, {len(merged):,} rows → {OUTPUT_PATH}")
    print(f"[Merge] Sepsis rate: {merged['SepsisLabel'].mean():.3f}")


if __name__ == "__main__":
    merge()
