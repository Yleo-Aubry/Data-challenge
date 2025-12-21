#  QRT 2025 - Leukemia Survival Prediction (AML Prognosis)

### Top 3.3% Solution (Rank 21st / 635)

![Python](https://img.shields.io/badge/Python-3.10-blue) ![scikit-survival](https://img.shields.io/badge/Lib-scikit--survival-red) ![XGBoost](https://img.shields.io/badge/Model-XGBoost%20AFT-green)

##  Overview
This repository contains the solution developed for the **"Overall Survival Prediction in AML"** challenge hosted on Challenge Data (ENS).

**The Goal:** Predict the survival time (time-to-event) of patients suffering from Acute Myeloid Leukemia (AML).
**The Complexity:**
* **High-Dimensional Data:** A mix of clinical attributes and complex molecular/genomic features.
* **Right-Censored Data:** Many patients were still alive at the end of the study, requiring specialized loss functions (Cox PH, AFT) rather than standard regression.

##  Results
* **Rank:** **21st** out of 635 participants.
* **Percentile:** **Top 3.3%**.
* **Metric:** Optimized the **Concordance Index (C-index)**, measuring the model's ability to correctly order patient risks.

##  Methodology

### 1. Survival Analysis Architecture
Unlike standard classification, this problem requires modeling the *probability of survival over time*. I implemented a voting ensemble of three distinct mathematical approaches:
* **XGBoost AFT (Accelerated Failure Time):** A gradient boosting model optimized to predict the log of the survival time directly.
* **Random Survival Forests:** To capture non-linear interactions between genetic mutations without assuming proportional hazards.
* **Cox Proportional Hazards (Regularized):** A linear baseline with ElasticNet penalty to select the most predictive genes.

### 2. Handling High-Dimensionality
The dataset contained extensive molecular data relative to the number of samples ($p \gg n$ regime).
* **Feature Selection:** Applied recursive feature elimination and correlation filtering to identify key biomarkers.
* **Imputation:** Used iterative imputation for missing clinical values to preserve statistical power.

### 3. Stack Used
* **Python:** Core logic.
* **scikit-survival:** The industry-standard library for survival analysis in Python.
* **XGBoost / LightGBM:** For gradient boosting implementations.

