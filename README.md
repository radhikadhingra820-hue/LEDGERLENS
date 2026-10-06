# 🔎 LEDGERLENS

**Public Development Project Anomaly Detection**

LEDGERLENS is a Streamlit machine-learning prototype that uses exploratory analysis, engineered project metrics and Isolation Forest to flag unusual public development project behaviour.

## Live Demo

Deploy from this repository with Streamlit Community Cloud and add the generated `streamlit.app` link here.

## Problem

Public development projects produce information about project cost, spending, fund release, utilisation, progress and timelines. Unusual combinations of these values can help identify projects for further review.

## What the prototype does

- Overview and EDA dashboard
- Isolation Forest anomaly detection
- Review queue for flagged projects
- Project-level anomaly explanations
- Sector-based peer comparisons

## ML Pipeline

Project data → Feature engineering → Standardization → Isolation Forest → Anomaly score → Review

## Baseline Features

- Cost overrun
- Fund utilisation
- Funds released ratio
- Time elapsed
- Progress vs time
- Spending vs progress

## Dataset

The repository includes a compact seed CSV for reproducibility. The baseline expands it deterministically to a 260-project development portfolio when needed, so the anomaly detector has enough observations for a meaningful baseline. The generated portfolio is synthetic and should not be presented as an official government dataset.

## Tech Stack

Python • Pandas • NumPy • scikit-learn • Isolation Forest • Streamlit

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Competition Challenge

Participants receive a working anomaly-detection baseline and can improve it through GitHub Issues.

Suggested directions:

1. Improve anomaly detection
2. Improve feature engineering
3. Improve anomaly explanations
4. Compare anomaly-detection approaches

## Project Structure

```
LEDGERLENS/
├── app.py
├── model.py
├── sample_projects.csv
├── requirements.txt
├── CONTRIBUTING.md
├── PRESENTATION.md
├── LICENSE
├── .gitignore
└── tests/
    └── test_model.py
```
