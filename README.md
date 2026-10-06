# 🔎 LEDGERLENS

**Public Development Project Anomaly Detection**

LEDGERLENS is a Streamlit machine-learning prototype that uses exploratory analysis, engineered project metrics and Isolation Forest to flag unusual public development project behaviour.

## Live Demo

Add the deployed Streamlit link here after deployment.

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

The starter prototype uses a controlled sample public-works dataset so the baseline is reproducible. It is intentionally suitable for demonstrating anomaly-detection improvements; it should not be presented as an official government dataset.

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
