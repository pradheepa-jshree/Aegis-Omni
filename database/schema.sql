-- database/schema.sql
-- PostgreSQL schema for Aegis-Omni ICU Early Warning System

-- ── Extensions ──────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;  -- optional: time-series optimisation

-- ── Patients ──────────────────────────────────────────────────────────────
CREATE TABLE patients (
    patient_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id     VARCHAR(64) UNIQUE NOT NULL,   -- hospital MRN
    first_name      VARCHAR(64),
    last_name       VARCHAR(64),
    date_of_birth   DATE,
    sex             CHAR(1) CHECK (sex IN ('M','F','O')),
    admission_ts    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    discharge_ts    TIMESTAMPTZ,
    unit            VARCHAR(32) DEFAULT 'ICU',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Vital Signs (time-series) ──────────────────────────────────────────────
CREATE TABLE vitals (
    id          BIGSERIAL PRIMARY KEY,
    patient_id  UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hr          FLOAT,    -- Heart Rate (bpm)
    map         FLOAT,    -- Mean Arterial Pressure (mmHg)
    sbp         FLOAT,    -- Systolic BP (mmHg)
    dbp         FLOAT,    -- Diastolic BP (mmHg)
    spo2        FLOAT,    -- SpO2 (%)
    resp_rate   FLOAT,    -- Respiratory Rate (bpm)
    temp        FLOAT,    -- Temperature (°C)
    etco2       FLOAT     -- End-tidal CO2 (mmHg)
);
CREATE INDEX idx_vitals_patient_ts ON vitals(patient_id, recorded_at DESC);

-- ── Lab Results ────────────────────────────────────────────────────────────
CREATE TABLE labs (
    id          BIGSERIAL PRIMARY KEY,
    patient_id  UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    collected_at TIMESTAMPTZ NOT NULL,
    lactate     FLOAT,
    wbc         FLOAT,
    hgb         FLOAT,
    hct         FLOAT,
    platelets   FLOAT,
    creatinine  FLOAT,
    bun         FLOAT,
    glucose     FLOAT,
    sodium      FLOAT,
    potassium   FLOAT,
    chloride    FLOAT,
    bicarbonate FLOAT,
    ph          FLOAT,
    pao2        FLOAT,
    paco2       FLOAT,
    fio2        FLOAT,
    bilirubin   FLOAT,
    troponin    FLOAT,
    fibrinogen  FLOAT,
    procalcitonin FLOAT
);
CREATE INDEX idx_labs_patient_ts ON labs(patient_id, collected_at DESC);

-- ── Aegis Predictions ─────────────────────────────────────────────────────
CREATE TABLE predictions (
    id              BIGSERIAL PRIMARY KEY,
    patient_id      UUID NOT NULL REFERENCES patients(patient_id),
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   VARCHAR(32) DEFAULT '1.0.0',
    risk_score      FLOAT NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    calibrated_prob FLOAT NOT NULL CHECK (calibrated_prob BETWEEN 0 AND 1),
    confidence      FLOAT CHECK (confidence BETWEEN 0 AND 1),
    alert_tier      VARCHAR(16) CHECK (alert_tier IN ('WATCH','WARN','CRITICAL')),
    alert_fired     BOOLEAN DEFAULT FALSE,
    top_features    JSONB,       -- [{"feature": "Lactate", "shap": 0.312}, ...]
    ai_narrative    TEXT,
    window_start    TIMESTAMPTZ,
    window_end      TIMESTAMPTZ
);
CREATE INDEX idx_predictions_patient ON predictions(patient_id, predicted_at DESC);
CREATE INDEX idx_predictions_alert   ON predictions(alert_fired, alert_tier)
    WHERE alert_fired = TRUE;

-- ── Alert Log ─────────────────────────────────────────────────────────────
CREATE TABLE alerts (
    id              BIGSERIAL PRIMARY KEY,
    prediction_id   BIGINT REFERENCES predictions(id),
    patient_id      UUID NOT NULL REFERENCES patients(patient_id),
    fired_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tier            VARCHAR(16) NOT NULL,
    acknowledged    BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(64),
    acknowledged_at TIMESTAMPTZ,
    outcome         VARCHAR(32),   -- 'TRUE_POSITIVE','FALSE_POSITIVE','UNKNOWN'
    notes           TEXT
);

-- ── Outcomes (Ground Truth for Model Monitoring) ──────────────────────────
CREATE TABLE outcomes (
    id              BIGSERIAL PRIMARY KEY,
    patient_id      UUID NOT NULL REFERENCES patients(patient_id),
    event_type      VARCHAR(64) NOT NULL,  -- 'septic_shock','cardiac_arrest','death','discharge'
    event_ts        TIMESTAMPTZ NOT NULL,
    sofa_score      FLOAT,
    apache_ii       FLOAT,
    icu_los_hours   FLOAT,
    hospital_mortality BOOLEAN,
    notes           TEXT
);

-- ── Model Versions ────────────────────────────────────────────────────────
CREATE TABLE model_versions (
    id              SERIAL PRIMARY KEY,
    version         VARCHAR(32) UNIQUE NOT NULL,
    trained_at      TIMESTAMPTZ DEFAULT NOW(),
    auroc           FLOAT,
    auprc           FLOAT,
    threshold       FLOAT,
    precision_val   FLOAT,
    recall_val      FLOAT,
    fdr             FLOAT,
    training_samples INT,
    notes           TEXT,
    is_active       BOOLEAN DEFAULT FALSE
);

-- ── Drift Log ─────────────────────────────────────────────────────────────
CREATE TABLE drift_events (
    id              SERIAL PRIMARY KEY,
    detected_at     TIMESTAMPTZ DEFAULT NOW(),
    feature_name    VARCHAR(64),
    severity        VARCHAR(16) CHECK (severity IN ('low','medium','high')),
    ks_statistic    FLOAT,
    p_value         FLOAT,
    action_taken    VARCHAR(128)
);

-- ── Views ─────────────────────────────────────────────────────────────────

-- Latest prediction per patient
CREATE OR REPLACE VIEW v_latest_predictions AS
SELECT DISTINCT ON (p.patient_id)
    p.patient_id,
    pa.external_id,
    pa.unit,
    p.predicted_at,
    p.risk_score,
    p.calibrated_prob,
    p.alert_tier,
    p.alert_fired,
    p.ai_narrative
FROM predictions p
JOIN patients pa ON pa.patient_id = p.patient_id
ORDER BY p.patient_id, p.predicted_at DESC;

-- Active alerts (unacknowledged)
CREATE OR REPLACE VIEW v_active_alerts AS
SELECT
    a.id AS alert_id,
    a.fired_at,
    a.tier,
    pa.external_id AS patient_mrn,
    pa.unit,
    p.risk_score,
    p.ai_narrative
FROM alerts a
JOIN patients pa ON pa.patient_id = a.patient_id
JOIN predictions p ON p.id = a.prediction_id
WHERE a.acknowledged = FALSE
ORDER BY a.fired_at DESC;
