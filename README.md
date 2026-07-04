# F1 Race Winner Prediction

A local end-to-end ML platform that predicts Formula 1 race winners from pre-race
information. The Core ML Platform (data processing through API/dashboard serving)
is complete; deployment, automated ingestion, and monitoring are future work.

## Architecture

```text
Ergast CSVs -> validated parquet -> master dataset -> temporal features
  -> model comparison/calibration -> MLflow registry -> prediction layer
  -> FastAPI -> Streamlit dashboard
```

The current model is an OOF-isotonic-calibrated tuned logistic regression,
registered as `f1-winner` version 2 at alias `Staging`. Historical predictions are
served through 2024. The dashboard calls the API over HTTP and does not import model
or feature code.

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest tests/
uvicorn app.api:app
streamlit run app/dashboard.py
```

The API requires `data/processed/features.parquet` and the local MLflow registry.
To rebuild datasets:

```bash
python -m src.data.build_interim --target all
python -m src.pipelines.build_dataset
python -m src.features.pipeline
```

See [the user guide](docs/user_guide.md) and
[API reference](docs/api_reference.md). Engineering design and evidence are in
`reports/`; `context/` is internal project memory.

## Current baseline

- Validation 2022–2023: 68.2% top-1, 88.6% top-3.
- Final test 2024: 45.8% top-1 (pole-baseline parity), 75.0% top-3.
- 285 automated tests at the milestone.
- No public-production security, CI/CD, containers, automated ETL, or monitoring.

Next recommended milestone: complete loader tests and measured ≥80% `src/` coverage,
establish Git tracking, then add CI.
