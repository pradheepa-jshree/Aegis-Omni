"""
setup_and_train.py — ONE COMMAND to go from zero to trained model.

What this does:
  1. Checks if data/sepsis_dataset.csv exists
     → If not, generates a realistic synthetic PhysioNet-format dataset
     → If real data exists, uses it as-is
  2. Runs the full preprocessing pipeline → artifacts/
  3. Trains the BiLSTM+Attention model → artifacts/best_model.pt
  4. Calibrates threshold (FDR < 5%) → artifacts/threshold.json
  5. Prints final metrics and run instructions

Run: python setup_and_train.py
  OR: python setup_and_train.py --data path/to/your/sepsis_dataset.csv
  OR: python setup_and_train.py --synthetic   (force synthetic even if real exists)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS   = "artifacts"
DEFAULT_CSV = "data/sepsis_dataset.csv"

BOLD  = "\033[1m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
CYAN  = "\033[96m"; RESET = "\033[0m"

def banner(msg):
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")

def ok(msg):   print(f"{GREEN}  ✓ {msg}{RESET}")
def info(msg): print(f"  → {msg}")


def step1_data(csv_path: str, force_synthetic: bool):
    banner("STEP 1 / 4  —  Dataset")

    if force_synthetic or not os.path.exists(csv_path):
        if not force_synthetic:
            print(f"{YELLOW}  ⚠ {csv_path} not found — generating synthetic dataset{RESET}")
            print(f"  (To use the real PhysioNet data, see data/merge_physionet.py)\n")
        else:
            info("Forcing synthetic dataset generation")

        from data.generate_physionet_synthetic import generate
        generate(csv_path)
        ok(f"Synthetic dataset saved → {csv_path}")
    else:
        import pandas as pd
        df = pd.read_csv(csv_path, sep="|", nrows=5)
        ok(f"Real dataset found: {csv_path}  ({df.shape[1]} columns)")
        info("Using real PhysioNet data")

    return csv_path


def step2_pipeline(csv_path: str):
    banner("STEP 2 / 4  —  Feature Engineering & Preprocessing")
    from data_pipeline.data_pipeline import run_pipeline
    t0 = time.time()
    result = run_pipeline(csv_path, ARTIFACTS)
    ok(f"Pipeline complete in {time.time()-t0:.1f}s")
    X_tr = result[0]
    info(f"Train sequences: {X_tr.shape} | Features: {X_tr.shape[2]}")
    return result


def step3_train():
    banner("STEP 3 / 4  —  Training BiLSTM + Attention")
    import train as train_module
    t0 = time.time()
    train_module.train()
    ok(f"Training complete in {time.time()-t0:.1f}s")
    ok(f"Best model → {ARTIFACTS}/best_model.pt")


def step4_calibrate():
    banner("STEP 4 / 4  —  Threshold Calibration (FDR < 5%)")
    import calibration as cal_module
    threshold, _ = cal_module.calibrate()
    ok(f"Calibration complete | threshold={threshold:.4f}")
    return threshold


def print_summary(threshold: float):
    import json
    banner("✅  AEGIS-OMNI IS READY")

    with open(f"{ARTIFACTS}/threshold.json") as f:
        m = json.load(f)

    print(f"""
  {BOLD}Model metrics (test set):{RESET}
    AUROC     : {m.get('auroc',   '?')}
    Precision : {m.get('precision','?')}  (target > 0.95)
    Recall    : {m.get('recall',  '?')}
    F1        : {m.get('f1',      '?')}
    FDR       : {m.get('fdr',     '?')}  (target < 0.05)
    Threshold : {m.get('threshold','?')}

  {BOLD}To start the system:{RESET}
    {CYAN}# Terminal 1 — API{RESET}
    uvicorn api.main:app --reload --port 8000

    {CYAN}# Terminal 2 — Dashboard{RESET}
    streamlit run streamlit_app.py

    {CYAN}# Terminal 3 — Live debug monitor (real model){RESET}
    python debug_run.py

    {CYAN}# (Optional) Federated simulation{RESET}
    python federated/federated_simulation.py

  {BOLD}API docs:{RESET} http://localhost:8000/docs
  {BOLD}Dashboard:{RESET} http://localhost:8501
""")


def main():
    parser = argparse.ArgumentParser(description="Aegis-Omni — One-shot setup & train")
    parser.add_argument("--data",      default=DEFAULT_CSV,
                        help=f"Path to sepsis CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic data generation even if real CSV exists")
    args = parser.parse_args()

    os.makedirs(ARTIFACTS, exist_ok=True)
    os.makedirs("data",     exist_ok=True)

    print(f"\n{BOLD}  AEGIS-OMNI — Setup & Train Pipeline{RESET}")
    print(f"  Artifacts → {ARTIFACTS}/\n")

    csv_path  = step1_data(args.data, args.synthetic)
    step2_pipeline(csv_path)
    step3_train()
    threshold = step4_calibrate()
    print_summary(threshold)


if __name__ == "__main__":
    main()
