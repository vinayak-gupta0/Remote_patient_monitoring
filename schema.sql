BEGIN;

-- Patients 
CREATE TABLE IF NOT EXISTS patients (
  id TEXT PRIMARY KEY,               
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Vital snapshots (time-series for charts)
CREATE TABLE IF NOT EXISTS vital_snapshots (
  id BIGSERIAL PRIMARY KEY,
  patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  hr REAL,
  rr REAL,
  temp REAL,
  bp_sys REAL,
  bp_dia REAL
);

CREATE INDEX IF NOT EXISTS idx_vitals_patient_ts
ON vital_snapshots (patient_id, ts DESC);

-- Minute logs
CREATE TABLE IF NOT EXISTS minute_logs (
  id BIGSERIAL PRIMARY KEY,
  patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
  ts_minute TIMESTAMPTZ NOT NULL,      

  hr REAL,
  rr REAL,
  temp REAL,
  bp_sys REAL,
  bp_dia REAL,

  ecg_snippet JSONB,             
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- prevent duplicates from Streamlit reruns:
  CONSTRAINT uq_minute_log UNIQUE (patient_id, ts_minute)
);

CREATE INDEX IF NOT EXISTS idx_minute_logs_patient_ts
ON minute_logs (patient_id, ts_minute DESC);

-- Daily report 
CREATE TABLE IF NOT EXISTS events (
  id BIGSERIAL PRIMARY KEY,
  patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,

  ts_minute TIMESTAMPTZ NOT NULL,        
  total_score INTEGER NOT NULL,
  risk_level TEXT NOT NULL,              
  trigger TEXT NOT NULL,            

  subscores JSONB NOT NULL,              
  vitals JSONB NOT NULL,              

  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT uq_event UNIQUE (patient_id, ts_minute, trigger)
);

CREATE INDEX IF NOT EXISTS idx_events_patient_ts
ON events (patient_id, ts_minute DESC);

-- 5) ECG segments 
CREATE TABLE IF NOT EXISTS ecg_segments (
  id BIGSERIAL PRIMARY KEY,
  patient_id TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,

  ts_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  duration_sec REAL NOT NULL,
  fs INTEGER NOT NULL DEFAULT 250,       

  lead TEXT NOT NULL DEFAULT 'single',   
  data JSONB NOT NULL,         

  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ecg_patient_ts
ON ecg_segments (patient_id, ts_start DESC);

COMMIT;
