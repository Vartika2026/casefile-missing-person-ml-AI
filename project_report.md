# CASEFILE Investigation Report

## Abstract
This project demonstrates an end-to-end machine-learning investigation-support workflow using synthetic movement trajectories and fictional case information.

## Introduction / Problem Statement
The goal is to analyze movement patterns and produce a ranked list of probable geographical areas and routes for an academic simulation.

## Objectives
The implementation follows the supplied project guide: preprocessing, feature engineering, clustering, anomaly detection, location prediction, route prediction, search priority, explainability, mapping, and dashboard delivery.

## Dataset Description
The demo uses synthetic GPS trajectories and synthetic case records. Public GeoLife/OpenStreetMap/data.gov.in sources are documented for future extension.

## Methodology
- K-Means: movement clustering
- Isolation Forest: anomaly detection
- Random Forest: supervised area prediction
- Markov Chain: route prediction
- Feature importance: explainability
- Weighted search-priority score

## Evaluation
The dashboard reports accuracy, precision, recall, F1, Top-1, Top-3 and Top-5 metrics for the synthetic location-prediction task. Anomaly results should be interpreted through false-positive/false-negative analysis and threshold choice.

## Limitations
Synthetic data do not reproduce real-world mobility complexity. Prediction quality can degrade with sparse data, changed routines, weather/context shifts, biased sampling, or geographic distribution shift.

## Ethical Considerations
All identities are fictional. Predictions are probabilistic and do not prove a person's location. Anomalies are not evidence of criminality or suspicious behavior.

## Future Scope
Add a real public GeoLife preprocessing pipeline, OSM road-network routing, calibrated probabilities, SHAP explanations, temporal models, graph-based route prediction, and formal uncertainty reporting.

## Conclusion
The project integrates supervised learning, unsupervised learning, anomaly detection, geospatial visualization and explainability in a complete Streamlit application.
