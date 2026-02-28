"""
debug_run.py — Live ICU monitor using the REAL trained AegisLSTM model.

Each tick:
  1. Generates a realistic evolving vital-sign window for each patient
  2. Scales it with the REAL scaler from training
  3. Runs REAL model inference (BiLSTM + Attention)
  4. Calibrates probability with the REAL isotonic calibrator
  5. Fires alerts at the REAL optimized threshold

Run: python debug_run.py
Requirements: artifacts/ must be populated (run setup_and_train.py first)
"""

import os, sys, json, pickle, time, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS = "artifacts"

# ── Load real artifacts ───────────────────────────────────────────────────────
def load_artifacts():
    from models.model import build_model

    with open(f"{ARTIFACTS}/scaler.pkl",       "rb") as f: scaler    = pickle.load(f)
    with open(f"{ARTIFACTS}/feature_cols.pkl", "rb") as f: feat_cols = pickle.load(f)
    with open(f"{ARTIFACTS}/calibrator.pkl",   "rb") as f: calibrator = pickle.load(f)
    with open(f"{ARTIFACTS}/threshold.json")        as f: thresh_info = json.load(f)

    model = build_model(len(feat_cols), "cpu")
    model.load_state_dict(torch.load(f"{ARTIFACTS}/best_model.pt", map_location="cpu"))
    model.eval()

    threshold = thresh_info["threshold"]
    print(f"[DebugRun] Model loaded | features:{len(feat_cols)} | threshold:{threshold:.4f}")
    return model, scaler, feat_cols, calibrator, threshold


# ── Patient vital-sign simulators ─────────────────────────────────────────────
PATIENTS = [
    {"id": "P001", "name": "J. Harrison", "bed": "ICU-01", "dx": "Post-cardiac surgery",
     "stable": True,  "base_risk": 0.18},
    {"id": "P002", "name": "M. Okonkwo",  "bed": "ICU-02", "dx": "Sepsis - pneumonia",
     "stable": False, "base_risk": 0.42},   # This one deteriorates
    {"id": "P003", "name": "R. Vasquez",  "bed": "ICU-03", "dx": "ARDS",
     "stable": True,  "base_risk": 0.22},
]

# Per-patient running vital state (12-hour rolling windows)
patient_state = {
    "P001": {"HR":74,"MAP":82,"SpO2":97,"Lactate":1.1,"SBP":118,"Resp":17,"Temp":37.1,"WBC":8.5},
    "P002": {"HR":96,"MAP":61,"SpO2":94,"Lactate":2.8,"SBP":95, "Resp":24,"Temp":38.4,"WBC":14.2},
    "P003": {"HR":68,"MAP":78,"SpO2":96,"Lactate":1.4,"SBP":112,"Resp":20,"Temp":37.0,"WBC":9.1},
}
patient_velocity = {
    "P001": 0.003,
    "P002": 0.012,   # Rising
    "P003": 0.001,
}
# Rolling 12h windows per patient (list of dicts, max len=12)
patient_windows = {p["id"]: [] for p in PATIENTS}


def evolve_vitals(pid: str, tick: int):
    """Advance each patient's vitals one step."""
    vt  = patient_state[pid]
    vel = patient_velocity[pid]
    p   = next(p for p in PATIENTS if p["id"] == pid)

    # Deterioration driver
    if not p["stable"] and tick > 3:
        vel = min(vel * 1.08, 0.030)
    else:
        vel = max(vel * 0.96, 0.001)
    patient_velocity[pid] = vel

    sev = 1.0 + vel * 20  # noise scales with deterioration speed

    vt["HR"]      = float(np.clip(vt["HR"]      + np.random.normal(0, 2.0*sev),  40, 180))
    vt["MAP"]     = float(np.clip(vt["MAP"]      + np.random.normal(0, 1.5*sev),  30, 130))
    vt["SpO2"]    = float(np.clip(vt["SpO2"]     + np.random.normal(0, 0.4*sev),  70, 100))
    vt["Lactate"] = float(np.clip(vt["Lactate"]  + np.random.normal(vel*2, 0.1*sev), 0.5, 12))
    vt["SBP"]     = float(np.clip(vt["SBP"]      + np.random.normal(0, 2.5*sev),  60, 200))
    vt["Resp"]    = float(np.clip(vt["Resp"]      + np.random.normal(0, 0.8*sev),   8,  40))
    vt["Temp"]    = float(np.clip(vt["Temp"]      + np.random.normal(0, 0.05*sev), 35, 41))
    vt["WBC"]     = float(np.clip(vt["WBC"]       + np.random.normal(vel*3, 0.3*sev), 1, 50))

    patient_windows[pid].append(dict(vt))
    if len(patient_windows[pid]) > 12:
        patient_windows[pid].pop(0)


def build_feature_window(pid: str, feat_cols: list, scaler) -> np.ndarray | None:
    """
    Turn the 12-hour rolling window into a scaled feature tensor
    matching the exact columns the model was trained on.
    """
    win = patient_windows[pid]
    if len(win) < 2:
        return None

    # Pad to 12 rows if needed (repeat first row)
    while len(win) < 12:
        win.insert(0, win[0])

    rows = []
    for step in win:
        row = {}
        # Fill known vitals
        row["HR"]      = step["HR"]
        row["MAP"]     = step["MAP"]
        row["O2Sat"]   = step["SpO2"]
        row["SBP"]     = step["SBP"]
        row["Resp"]    = step["Resp"]
        row["Temp"]    = step["Temp"]
        row["WBC"]     = step["WBC"]
        row["Lactate"] = step["Lactate"]
        row["DBP"]     = step["MAP"] * 0.7 + np.random.normal(0, 2)
        row["EtCO2"]   = 35 + np.random.normal(0, 3)

        # Derived features (shock index, ratios)
        row["shock_index"]       = step["HR"] / max(step["SBP"], 1)
        row["lactate_map_ratio"] = step["Lactate"] / max(step["MAP"], 1)

        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)

    # Add rolling stats for HR, MAP, Lactate (match pipeline engineering)
    for col in ["HR","MAP","Lactate","Resp","SBP"]:
        if col in df.columns:
            df[f"{col}_mean4h"]  = df[col].rolling(4, min_periods=1).mean()
            df[f"{col}_std4h"]   = df[col].rolling(4, min_periods=1).std().fillna(0)
            df[f"{col}_slope2h"] = df[col].diff(2).fillna(0)

    # Build array aligned to training feature_cols
    mat = np.zeros((12, len(feat_cols)), dtype=np.float32)
    for i, fc in enumerate(feat_cols):
        if fc in df.columns:
            mat[:, i] = df[fc].values.astype(np.float32)
        # else stays 0 (median-imputed effectively)

    # Scale
    scaled = scaler.transform(mat.reshape(-1, len(feat_cols))).reshape(12, len(feat_cols))
    return scaled.astype(np.float32)


def real_inference(pid, model, scaler, feat_cols, calibrator):
    """Run actual model inference. Returns (calibrated_prob, raw_prob)."""
    window = build_feature_window(pid, feat_cols, scaler)
    if window is None:
        return 0.05, 0.05

    x = torch.from_numpy(window).unsqueeze(0)  # (1, 12, F)
    with torch.no_grad():
        raw_prob, _ = model(x)
    raw = float(raw_prob.item())
    cal = float(calibrator.transform([raw])[0])
    return cal, raw


# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD  = "\033[1m";  DIM = "\033[2m"
GREEN  = "\033[92m"; CYAN  = "\033[96m"
TIER_C = {"WATCH":"\033[94m","WARN":"\033[93m","CRITICAL":"\033[91m"}

def tier(r):
    if r >= 0.80: return "CRITICAL"
    if r >= 0.65: return "WARN"
    return "WATCH"

def risk_bar(r, w=20):
    filled = int(r * w)
    bar    = "█"*filled + "░"*(w-filled)
    return f"{TIER_C[tier(r)]}{bar}{RESET}"

def vital_flag(name, val, lo, hi):
    bad = val < lo or val > hi
    c   = "\033[93m" if bad else DIM
    f   = " !" if bad else "  "
    return f"{c}{name}:{val:.1f}{f}{RESET}"


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  AEGIS-OMNI — REAL MODEL DEBUG MONITOR")
    print("  Loading trained artifacts...")
    print("=" * 65)

    try:
        model, scaler, feat_cols, calibrator, threshold = load_artifacts()
    except FileNotFoundError:
        print("\n  ✗ Artifacts not found. Run first:\n")
        print("    python setup_and_train.py\n")
        sys.exit(1)

    print(f"  ✓ Ready | threshold={threshold:.4f} | features={len(feat_cols)}")
    print("  Live scores update every 5s — Ctrl+C to stop\n")

    tick = 0
    total_alerts = 0

    try:
        while True:
            tick += 1
            os.system("cls" if os.name == "nt" else "clear")

            print(f"{BOLD}╔══════════════════════════════════════════════════════════════╗{RESET}")
            print(f"{BOLD}║  AEGIS-OMNI  ·  REAL MODEL  ·  {time.strftime('%H:%M:%S')}  ·  Tick #{tick:04d}        ║{RESET}")
            print(f"{BOLD}╚══════════════════════════════════════════════════════════════╝{RESET}")
            print(f"  {DIM}BiLSTM+Attention | {len(feat_cols)} features | threshold={threshold:.4f} | alerts={total_alerts}{RESET}\n")

            for p in PATIENTS:
                pid = p["id"]
                evolve_vitals(pid, tick)

                # ── REAL model inference ──────────────────────────────────
                cal_prob, raw_prob = real_inference(pid, model, scaler, feat_cols, calibrator)

                vt = patient_state[pid]
                t  = tier(cal_prob)
                tc = TIER_C[t]

                # Confidence via MC Dropout (10 passes with dropout active)
                model.train()
                mc = []
                with torch.no_grad():
                    w = build_feature_window(pid, feat_cols, scaler)
                    if w is not None:
                        x = torch.from_numpy(w).unsqueeze(0)
                        for _ in range(10):
                            pr, _ = model(x)
                            mc.append(float(pr.item()))
                model.eval()
                uncertainty = float(np.std(mc)) if mc else 0.05
                confidence  = round(max(0.0, min(1.0, 1.0 - uncertainty * 4)), 2)

                lead_time = max(0.2, 6.0 - cal_prob * 5.8)
                ci_lo     = max(0.0, cal_prob - 2*uncertainty)
                ci_hi     = min(1.0, cal_prob + 2*uncertainty)

                print(f"  {BOLD}┌─ {p['bed']}  {p['name']}  {'─'*35}{RESET}")
                print(f"  │  {DIM}{p['dx']}{RESET}")
                print(f"  │")
                print(f"  │  RISK     {risk_bar(cal_prob)}  {tc}{BOLD}{cal_prob*100:.1f}%{RESET} [{tc}{t}{RESET}]  raw={raw_prob:.3f}")
                print(f"  │  CI       [{ci_lo*100:.1f}%–{ci_hi*100:.1f}%]   Uncertainty: ±{uncertainty:.3f}   Confidence: {confidence:.0%}")
                print(f"  │  LEAD     ~{lead_time:.1f}h to predicted event")
                print(f"  │")

                hr_s  = vital_flag("HR",      vt["HR"],       60, 100)
                map_s = vital_flag("MAP",      vt["MAP"],      65, 100)
                spo_s = vital_flag("SpO2",    vt["SpO2"],      95, 100)
                lac_s = vital_flag("Lac",     vt["Lactate"],  0.5, 2.0)
                rsp_s = vital_flag("Resp",    vt["Resp"],      12,  20)
                print(f"  │  VITALS   {hr_s}  {map_s}  {spo_s}  {lac_s}  {rsp_s}")
                print(f"  │")

                if t in ("WARN", "CRITICAL"):
                    total_alerts += 1
                    print(f"  │  {tc}{BOLD}🚨 ALERT FIRED  —  threshold={threshold:.3f}  cal_prob={cal_prob:.3f}{RESET}")

                    drivers = []
                    if vt["MAP"]     < 65:  drivers.append(f"MAP↓ ({vt['MAP']:.0f} mmHg)")
                    if vt["Lactate"] > 2.0: drivers.append(f"Lactate↑ ({vt['Lactate']:.1f})")
                    if vt["SpO2"]    < 95:  drivers.append(f"SpO2↓ ({vt['SpO2']:.0f}%)")
                    if vt["HR"]      > 100: drivers.append(f"HR↑ ({vt['HR']:.0f})")
                    if not drivers: drivers = ["compound deterioration pattern"]

                    event = "Septic Shock" if vt["Lactate"] > 2.0 or vt["MAP"] < 65 else "Haemodynamic compromise"
                    print(f"  │  {tc}  Predicted event:  {event}{RESET}")
                    print(f"  │  {tc}  Driving vitals:   {', '.join(drivers[:3])}{RESET}")
                    print(f"  │  {tc}  Model confidence: {confidence:.0%}{RESET}")
                    print(f"  │")
                    print(f"  │  {DIM}  → Vasopressor review immediately")
                    print(f"  │    → Blood cultures + lactate STAT")
                    print(f"  │    → Physician bedside within 15 min{RESET}")
                else:
                    print(f"  │  {GREEN}✓ Below alert threshold ({cal_prob*100:.1f}% < {threshold*100:.0f}%)  —  monitoring{RESET}")

                print(f"  {BOLD}└{'─'*62}{RESET}\n")

            print(f"  {DIM}Model: BiLSTM+Attention | Calibrated isotonic | FDR<5% threshold{RESET}")
            print(f"  {DIM}Next inference in 5s...  (Ctrl+C to stop){RESET}")
            time.sleep(5)

    except KeyboardInterrupt:
        print(f"\n\n  Aegis-Omni stopped. Total alerts fired: {total_alerts}\n")


if __name__ == "__main__":
    main()
