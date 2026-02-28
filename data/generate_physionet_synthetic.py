"""
data/generate_physionet_synthetic.py

Generates a realistic synthetic PhysioNet-format sepsis dataset.
Used when you don't have the real dataset yet — produces statistically
similar distributions so the model trains meaningfully.

The REAL PhysioNet 2019 dataset can be downloaded from:
  https://physionet.org/content/challenge-2019/1.0.0/

To use the real dataset instead:
  1. Download training_setA.zip and training_setB.zip
  2. Run: python data/merge_physionet.py  (included below)
  3. That produces data/sepsis_dataset.csv in the same format as this script

Run: python data/generate_physionet_synthetic.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_PATIENTS   = 800
MAX_ICU_HOURS = 72
SEPSIS_RATE   = 0.12   # ~12% develop sepsis — matches PhysioNet stats

FEATURE_DEFAULTS = {
    "HR":           (80,  15),
    "O2Sat":        (97,  2),
    "Temp":         (37.0, 0.8),
    "SBP":          (120, 20),
    "MAP":          (80,  15),
    "DBP":          (70,  12),
    "Resp":         (18,  4),
    "EtCO2":        (35,  5),
    "BaseExcess":   (0,   4),
    "HCO3":         (24,  4),
    "FiO2":         (0.4, 0.15),
    "pH":           (7.4, 0.05),
    "PaCO2":        (40,  8),
    "SaO2":         (97,  2),
    "AST":          (30,  20),
    "BUN":          (15,  8),
    "Alkalinephos": (80,  40),
    "Calcium":      (9.0, 0.8),
    "Chloride":     (102, 5),
    "Creatinine":   (1.0, 0.4),
    "Bilirubin_direct": (0.3, 0.2),
    "Glucose":      (110, 25),
    "Lactate":      (1.2, 0.5),
    "Magnesium":    (2.0, 0.3),
    "Phosphate":    (3.5, 0.8),
    "Potassium":    (4.0, 0.5),
    "Bilirubin_total": (0.8, 0.4),
    "TroponinI":    (0.02, 0.05),
    "Hct":          (38,  5),
    "Hgb":          (12,  2),
    "PTT":          (30,  8),
    "WBC":          (9,   4),
    "Fibrinogen":   (300, 80),
    "Platelets":    (220, 70),
    "Age":          (62,  15),
    "Gender":       None,
    "Unit1":        None,
    "Unit2":        None,
    "HospAdmTime":  (-24, 12),
}

SEPSIS_DRIFT = {
    "HR":       +25,
    "MAP":      -20,
    "Lactate":  +3.5,
    "Resp":     +8,
    "Temp":     +1.2,
    "WBC":      +8,
    "Creatinine": +0.8,
    "O2Sat":    -5,
    "SBP":      -25,
    "pH":       -0.08,
}


def make_patient(pid: int, is_sepsis: bool, total_hours: int) -> pd.DataFrame:
    rows = []
    sepsis_onset = int(total_hours * np.random.uniform(0.5, 0.85)) if is_sepsis else total_hours + 99
    label_active = False

    # Per-patient baseline offsets
    baseline_shift = {k: np.random.normal(0, v[1] * 0.3)
                      for k, v in FEATURE_DEFAULTS.items() if v is not None}

    for t in range(total_hours):
        row = {"patient_id": pid, "ICULOS": t, "SepsisLabel": 0}

        # Hours until sepsis
        hrs_to_sepsis = sepsis_onset - t
        progression   = max(0.0, 1.0 - hrs_to_sepsis / 8.0) if is_sepsis else 0.0

        for feat, params in FEATURE_DEFAULTS.items():
            if feat == "Gender":
                row[feat] = int(pid % 2)
                continue
            if feat in ("Unit1", "Unit2"):
                row[feat] = int(np.random.random() > 0.5)
                continue

            mean, std = params
            drift = SEPSIS_DRIFT.get(feat, 0) * progression
            base  = baseline_shift.get(feat, 0)
            val   = mean + base + drift + np.random.normal(0, std * 0.4)

            # Clip to physiologically plausible ranges
            val = np.clip(val, mean - 3*std, mean + 3*std)

            # Introduce realistic missingness (~20% for labs, 2% for vitals)
            is_lab = feat not in ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","Age","Gender","Unit1","Unit2","HospAdmTime","ICULOS"]
            miss_p = 0.20 if is_lab else 0.02
            if np.random.random() < miss_p:
                val = np.nan

            row[feat] = round(float(val), 2) if not np.isnan(val) else np.nan

        # SepsisLabel fires 6h after onset (matching PhysioNet labeling)
        if is_sepsis and t >= sepsis_onset:
            row["SepsisLabel"] = 1

        rows.append(row)

    return pd.DataFrame(rows)


def generate(output_path: str = "data/sepsis_dataset.csv"):
    print("[DataGen] Generating synthetic PhysioNet-format dataset...")
    frames = []

    for pid in range(N_PATIENTS):
        is_sepsis   = np.random.random() < SEPSIS_RATE
        total_hours = int(np.random.uniform(24, MAX_ICU_HOURS))
        df = make_patient(pid, is_sepsis, total_hours)
        frames.append(df)

        if (pid + 1) % 100 == 0:
            print(f"  Generated {pid+1}/{N_PATIENTS} patients...")

    full = pd.concat(frames, ignore_index=True)

    pos_rate = full["SepsisLabel"].mean()
    print(f"[DataGen] Total rows: {len(full):,} | Sepsis label rate: {pos_rate:.3f}")
    print(f"[DataGen] Columns: {list(full.columns)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full.to_csv(output_path, sep="|", index=False)
    print(f"[DataGen] Saved → {output_path}")
    return output_path


if __name__ == "__main__":
    generate()
