# 🏥 Aegis-Omni — ICU Early Warning System

> Predicts septic shock **6 hours in advance** using time-series ICU vitals.
> BiLSTM + Attention · SHAP Explainability · Featherless AI Narratives · FastAPI · Streamlit

---

## Architecture

```
PhysioNet CSV
    ↓
data_pipeline/data_pipeline.py   ← rolling features, sliding windows, scaling
    ↓
models/model.py                  ← BiLSTM + Multi-Head Attention (PyTorch)
    ↓
train.py                         ← focal loss, early stopping, checkpoint
    ↓
calibration.py                   ← isotonic calibration, threshold @ FDR < 5%
    ↓
explainability/explain.py        ← SHAP + Featherless AI narrative
    ↓
api/main.py                      ← FastAPI: POST /predict
    ↓
streamlit_app.py                 ← Patient dashboard
    ↓
Docker / Render deployment
```

---

## Quickstart (local, no Docker)

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare dataset
Download the PhysioNet Challenge 2019 sepsis dataset and merge into a single CSV:
```
data/sepsis_dataset.csv
```
The CSV must have a `|`-delimited header with columns including `HR`, `MAP`, `Lactate`,
`SepsisLabel`, etc. (standard PhysioNet format). Add a `patient_id` column or let the
pipeline auto-generate one.

### 3. Run the pipeline
```bash
# Step 1 — Preprocessing
python data_pipeline/data_pipeline.py data/sepsis_dataset.csv

# Step 2 — Train
python train.py

# Step 3 — Calibrate threshold
python calibration.py

# (Optional) Step 4 — Federated simulation
python federated/federated_simulation.py
```
All artefacts are saved to `artifacts/`.

### 4. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Start the dashboard
```bash
streamlit run streamlit_app.py
```
Open http://localhost:8501

---

## Docker (recommended)

```bash
cp .env.example .env
# Edit .env with your FEATHERLESS_API_KEY

docker compose up --build
```

| Service    | URL                    |
|------------|------------------------|
| FastAPI    | http://localhost:8000  |
| Streamlit  | http://localhost:8501  |
| Postgres   | localhost:5432         |
| API docs   | http://localhost:8000/docs |

---

## API Reference

### `POST /predict`
```json
{
  "patient_id": "P001",
  "window": {
    "HR":      [96, 98, 102, ...],   // 12 hourly readings
    "MAP":     [72, 70, 68, ...],
    "Lactate": [1.2, 1.4, 1.6, ...],
    "SpO2":    [97, 97, 96, ...],
    "Resp":    [18, 18, 19, ...]
  },
  "vitals_snapshot": { "HR": 104, "MAP": 52, "Lactate": 3.3 }
}
```

**Response:**
```json
{
  "patient_id":   "P001",
  "risk_score":   0.7821,
  "risk_percent": "78.2%",
  "confidence":   0.891,
  "alert_tier":   "WARN",
  "top_features": ["Lactate", "MAP", "shock_index", ...],
  "shap_values":  [0.31, 0.24, 0.18, ...],
  "ai_narrative": "Aegis-Omni assigns HIGH risk (78.2%) ...",
  "threshold":    0.624,
  "alert":        true
}
```

---

## Featherless AI Setup

1. Get your API key from [featherless.ai](https://featherless.ai)
2. Add to `.env`:
   ```
   FEATHERLESS_API_KEY=sk-...
   FEATHERLESS_MODEL=meta-llama/Llama-3.1-8B-Instruct
   ```
3. If key is absent, the system falls back to a deterministic template narrative.

---

## Deployment on Render

1. Push to GitHub
2. Create a **Web Service** on [render.com](https://render.com):
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
3. Add environment variables from `.env`
4. Mount a persistent disk at `/app/artifacts` and pre-upload trained artifacts

---

## Project Structure

```
aegis-omni/
├── data_pipeline/
│   └── data_pipeline.py     ← preprocessing & sequence builder
├── models/
│   └── model.py             ← BiLSTM + Attention architecture
├── explainability/
│   └── explain.py           ← SHAP + Featherless AI narrative
├── federated/
│   └── federated_simulation.py  ← FedAvg across 3 simulated hospitals
├── api/
│   └── main.py              ← FastAPI backend
├── database/
│   └── schema.sql           ← PostgreSQL schema
├── train.py                 ← training loop (focal loss)
├── calibration.py           ← threshold optimisation (FDR < 5%)
├── streamlit_app.py         ← Streamlit dashboard
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example

