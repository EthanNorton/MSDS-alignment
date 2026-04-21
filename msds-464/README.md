# RL Data Platform Starter (Platform + Analytics Engineering Focus)

This project reframes RL coursework as an analytics platform workflow that an AI/analytics team could own in production.

It is built around a medallion-style pipeline and experimentation lifecycle:

1. **Ingest (Bronze)** raw public data with Spark.
2. **Transform (Silver)** clean and standardize feature-ready state data.
3. **Curate (Gold)** produce aggregated decision signals for training/evaluation.
4. **Train + Evaluate** RL baseline artifacts for reproducible experimentation.
5. **CI/CD** validate pipeline quality and smoke test code on every pull request.

## Why This Fits Platform / Analytics Engineering Roles

- Spark-first ingestion and transformation code (`pyspark`).
- Databricks-ready folder + workflow scaffolding.
- Data quality and schema checks built into transformation.
- Reproducible outputs written as Parquet/JSON artifacts.
- CI pipeline to enforce reliability and consistency.

## Public Dataset

Default ingestion target:

- NYC TLC Yellow Taxi Trips (public parquet, no auth)
- Example file:
  - `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet`

This dataset supports realistic operations questions:
- dispatch/load balancing,
- surge-aware policy decisions,
- queueing and routing proxies.

## Multi-Source Ingestion (No Code Changes)

You can switch ingestion sources with environment variables:

- `DATA_SOURCE=nyc_tlc` (default): read public TLC parquet from URL
- `DATA_SOURCE=inventory_synth`: generate synthetic operations data with the same normalized schema

Optional:
- `SOURCE_URL` to point to a different parquet file
- `BRONZE_PATH` to override output path for the bronze layer

### PowerShell Examples

```powershell
# Default: NYC TLC parquet source
$env:DATA_SOURCE="nyc_tlc"
python src/ingest_spark.py

# Alternate: synthetic inventory-like source
$env:DATA_SOURCE="inventory_synth"
python src/ingest_spark.py

# Custom URL example
$env:DATA_SOURCE="nyc_tlc"
$env:SOURCE_URL="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet"
python src/ingest_spark.py
```

## Project Structure

```text
.
├── .github/workflows/ci.yml
├── databricks/
│   └── workflows/
│       └── rl_data_platform_job.yml
├── data/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── artifacts/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── config.py
│   ├── ingest_spark.py
│   ├── transform_spark.py
│   ├── train_rl.py
│   └── evaluate.py
├── tests/
│   └── test_data.py
├── requirements.txt
└── README.md
```

## Quick Start (Local)

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run pipeline:
   - `python src/ingest_spark.py` (supports env-based source switching)
   - `python src/transform_spark.py`
   - `python src/train_rl.py`
   - `python src/evaluate.py`
4. Run tests:
   - `pytest -q`

## Databricks Integration

Use `databricks/workflows/rl_data_platform_job.yml` as a starter workflow spec.

Recommended deployment pattern:

1. Keep pipeline code in repo (`src/`).
2. Use Databricks Repos or Asset Bundles for deployment.
3. Configure a job with task chaining:
   - `ingest_spark` -> `transform_spark` -> `train_rl` -> `evaluate`.
4. Store outputs in a Unity Catalog volume or cloud object storage.

## CI/CD Philosophy

The CI workflow runs:
- dependency install,
- unit tests,
- pipeline lint/smoke checks (extendable),
- gatekeeping before merge.

This gives you an interview-ready narrative:

> "I built an RL project as a platform system with Spark ingestion, medallion-style transformations, reproducible experiments, and CI-driven quality controls, designed for Databricks execution."
