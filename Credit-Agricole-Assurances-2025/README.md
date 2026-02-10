#  AgriRisk: Trinity Capped Architecture

### Top 4.6% Solution - Insurance Claim Prediction Challenge

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red) ![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)

##  Overview
This repository contains the source code for the **"Trinity Capped"** model, which ranked in the **Top 13** (out of 284 teams) in the Crédit Agricole Assurances data challenge (2025).

The goal was to predict the **claim cost (Charge)** for agricultural insurance contracts based on heterogeneous data (Meteorological, Structural, and Economic features).

##  The "Trinity" Architecture
To handle the high variance and Zero-Inflated nature of the data, I designed a 3-pillar ensemble strategy:

| Component | Model | Weight | Role |
| :--- | :--- | :--- | :--- |
| **1. Physics** | **LightGBM (Tweedie)** | **50%** | Captures the physical causality (Climate $\times$ Structure). Uses `tweedie_variance_power=1.5` to model the compound Poisson-Gamma distribution. |
| **2. Deep** | **ZINB Neural Net** | **30%** | A custom PyTorch network with **Entity Embeddings** for categorical variables. It learns non-linear interactions that trees miss. |
| **3. Safety** | **XGBoost** | **20%** | A robust baseline to stabilize predictions and prevent overfitting from the Deep Learning component. |

## Feature Engineering Highlights
Instead of relying on raw data, I engineered domain-specific features:
* **Capital Density:** Total insured capital divided by surface area. High-density farms have higher claim severity potential.
* **Climate Stress Index:** Interaction between temperature duration and drought indicators.

##  Robustness Strategy: "Capping"
The evaluation metric (RMSE) is highly sensitive to outliers.
I implemented a **Statistical Capping Strategy** at the 99.5th percentile on the predictions. This acts as a "circuit breaker," preventing the model from predicting unrealistically high values that would severely penalize the score on the Private Leaderboard.
