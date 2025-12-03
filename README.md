# Cancer Survival Prediction System

A Streamlit-based clinical decision support app for survival prediction, treatment recommendations, and risk assessment.

## Quick Start

Prerequisites: Python 3.11 (local) or Docker.

Local (dev):
```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
# open http://localhost:8501
```

Docker (recommended for consistent environment):
```bash
# from project root
docker compose build
docker compose up -d
# open http://localhost:8501
```
Stop:
```bash
docker compose down
```

## Notes
- Ensure `models/` and `data/` exist in the project root or are mounted into the container so the app can load the .pkl files.
- `docker-compose.yml` mounts the project for live development; remove the bind mount for production images.
- Rebuild the image after dependency changes: `docker compose build --no-cache`.

## Minimal project layout
```
├── data/         # datasets and encoders
├── models/       # trained .pkl files
├── streamlit_app/ # app entry and requirements
└── src/          # training & analysis scripts
```

## Recommended repository additions
- `.env.example` for configurable settings
- `scripts/run_dev.sh` to simplify local Docker startup
- a small `/health` endpoint or healthcheck in `docker-compose.yml`
- CI to build and test images (optional)

## Disclaimer
This tool supports clinical decision-making and is not a substitute for professional judgment. Validate outputs before clinical use.
