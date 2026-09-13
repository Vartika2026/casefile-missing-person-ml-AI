# CASEFILE: AI-Powered Missing Person Investigation and Probable Location Prediction System

> **Academic simulation only.** This application uses synthetic case identities and synthetic movement data. It is not intended for real-world missing-person decisions.

## What this project implements
- Data collection/source documentation
- Data preprocessing and validation
- Movement feature engineering
- K-Means movement clustering
- Isolation Forest anomaly detection
- Random Forest location prediction
- Markov-chain route prediction
- Search-priority scoring
- Feature-importance explainability
- Folium/GeoPandas-ready interactive map
- Streamlit dashboard
- Model evaluation and comparison
- Synthetic case investigation

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python generate_data.py
streamlit run app/app.py
```

The dashboard will normally open at `http://localhost:8501`.

## Project structure
See the supplied Advanced ML project guide. This repository follows the requested `data/`, `notebooks/`, `models/`, `src/`, `app/`, and `reports/` organization.

## Data
The packaged demo uses synthetic GPS trajectories and synthetic case records so that no personal data is used. The project guide recommends Microsoft GeoLife, OpenStreetMap, and data.gov.in for public-data extensions.

## Safety and ethics
Predictions are probabilistic and are not proof of a person's location. Anomalies are not evidence of criminal or suspicious behavior. False positives, bias, data sparsity, and distribution shift can materially affect results.
