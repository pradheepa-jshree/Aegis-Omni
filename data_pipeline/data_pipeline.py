"""
data_pipeline/data_pipeline.py
Loads the PhysioNet sepsis CSV → engineers features → builds LSTM sequences.
Works with both the real dataset and the synthetic generator.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── Column definitions (PhysioNet 2019 standard) ─────────────────────────────
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LAB_COLS   = [
    "BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN",
    "Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct",
    "Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total",
    "TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets",
]
DEMO_COLS  = ["Age","Gender","Unit1","Unit2","HospAdmTime","ICULOS"]
LABEL_COL  = "SepsisLabel"

WINDOW_SIZE    = 12   # 12-hour look-back
PREDICTION_GAP = 6    # predict event 6 hours ahead


# ── 1. Load ──────────────────────────────────────────────────────────────────

def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="|")
    if "patient_id" not in df.columns:
        # Auto-assign patient_id by detecting resets in ICULOS
        df["patient_id"] = (df["ICULOS"].diff() <= 0).cumsum()
    print(f"[Pipeline] Loaded  {len(df):,} rows | "
          f"{df['patient_id'].nunique()} patients | "
          f"sepsis rate: {df[LABEL_COL].mean():.3f}")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values("ICULOS").copy()
        for col in VITAL_COLS:
            if col not in g.columns:
                continue
            g[f"{col}_mean4h"]  = g[col].rolling(4, min_periods=1).mean()
            g[f"{col}_std4h"]   = g[col].rolling(4, min_periods=1).std().fillna(0)
            g[f"{col}_slope2h"] = g[col].diff(2).fillna(0)

        # Clinical composite indices
        if "HR" in g.columns and "SBP" in g.columns:
            g["shock_index"] = (g["HR"] / (g["SBP"].clip(lower=1))).clip(0, 5)
        if "Lactate" in g.columns and "MAP" in g.columns:
            g["lactate_map_ratio"] = (g["Lactate"] / (g["MAP"].clip(lower=1))).clip(0, 10)
        if "Resp" in g.columns and "PaCO2" in g.columns:
            g["resp_paco2"] = (g["Resp"] * g["PaCO2"]).clip(0, 1000)

        frames.append(g)

    out = pd.concat(frames).reset_index(drop=True)
    print(f"[Pipeline] Engineered → {out.shape[1]} columns")
    return out


# ── 3. Imputation ─────────────────────────────────────────────────────────────

def impute(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = (
        df.groupby("patient_id")[feature_cols]
          .transform(lambda x: x.ffill().bfill())
    )
    medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)
    return df


# ── 4. Sequence Builder ───────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, feature_cols: list):
    """
    Sliding window: X (N, WINDOW_SIZE, F)  y (N,)
    Label = 1 if sepsis occurs in the next PREDICTION_GAP hours.
    """
    X_list, y_list = [], []

    for pid, g in df.groupby("patient_id"):
        g    = g.sort_values("ICULOS").reset_index(drop=True)
        vals = g[feature_cols].values.astype(np.float32)
        lbls = g[LABEL_COL].values

        end = len(g) - WINDOW_SIZE - PREDICTION_GAP + 1
        for i in range(max(0, end)):
            window       = vals[i : i + WINDOW_SIZE]
            future_label = int(lbls[i + WINDOW_SIZE : i + WINDOW_SIZE + PREDICTION_GAP].max())
            X_list.append(window)
            y_list.append(future_label)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"[Pipeline] Sequences: {X.shape} | Positive rate: {y.mean():.4f}")
    return X, y


# ── 5. Helper ─────────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list:
    skip = {"patient_id", LABEL_COL}
    return [c for c in df.columns if c not in skip]


# ── 6. Full Pipeline ──────────────────────────────────────────────────────────

def run_pipeline(csv_path: str, save_dir: str = "artifacts"):
    os.makedirs(save_dir, exist_ok=True)

    df_raw = load_raw(csv_path)
    df_eng = engineer_features(df_raw)
    feat_cols = get_feature_cols(df_eng)
    df_imp = impute(df_eng, feat_cols)

    X, y = build_sequences(df_imp, feat_cols)

    # ── Scale ──
    N, T, F  = X.shape
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)

    # ── Split: 70 / 15 / 15 ──
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.15,
                                               random_state=42, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.176,
                                               random_state=42, stratify=y_tr)

    # ── Save ──
    for name, arr in [("X_train",X_tr),("X_val",X_va),("X_test",X_te),
                      ("y_train",y_tr),("y_val",y_va),("y_test",y_te)]:
        np.save(f"{save_dir}/{name}.npy", arr)

    with open(f"{save_dir}/scaler.pkl",      "wb") as f: pickle.dump(scaler,    f)
    with open(f"{save_dir}/feature_cols.pkl","wb") as f: pickle.dump(feat_cols, f)

    print(f"[Pipeline] Saved → train:{X_tr.shape} val:{X_va.shape} test:{X_te.shape}")
    return X_tr, X_va, X_te, y_tr, y_va, y_te, scaler, feat_cols


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sepsis_dataset.csv"
    run_pipeline(path)
