# District Flood Risk Assessment & Decision Support System

## Overview
This project is an AI-assisted decision support system designed for classifying historical district flood-risk patterns across India. It provides a data-driven approach to understanding flood vulnerabilities at the district level by analyzing historical flood inventories and physical exposure metrics.

## Problem Statement
Understanding flood risk at a granular, district level is critical for effective disaster management and resource allocation. Traditional methods often struggle to synthesize diverse historical data (e.g., duration, affected population, animal fatalities, and flooded areas) into a cohesive risk assessment. This project aims to bridge that gap by using machine learning to classify historical flood severity, benchmarked against the District Flood Severity Index (DFSI).

## What the System Does
The system processes historical flood data to classify districts into four risk quartiles: **Low**, **Moderate**, **High**, and **Very High**. It provides an interactive Streamlit dashboard that visualizes the predicted risk class, model confidence, historical indicators, and feature contributions for any selected district in India. 

## Key Features
- **Historical Risk Classification**: Classifies 731 Indian districts into historical risk quartiles.
- **Interactive Dashboard**: A clean, professional Streamlit interface for exploring district-level data.
- **Contextual Rainfall**: Displays current rainfall context alongside historical indicators (rainfall is excluded from the ML model to prevent target leakage and maintain historical focus).
- **Explainable AI (XAI)**: Provides global and district-specific local explanations for the model's classifications.

## AI / ML Approach
- **Algorithm**: The core classification is performed using a **Random Forest Classifier**.
- **Model Explainability**: The project leverages **SHAP (Shapley Additive exPlanations)** and **Permutation Importance** to interpret the model's decisions globally and locally.
- **AI & Development context**: **IBM BOB** was used during the ideation phase to explore and refine the project concept and its sustainability-oriented direction. The final machine-learning prototype, data processing pipeline, and dashboard were implemented independently using Python and open-source data-science libraries. IBM BOB was NOT used as part of the Random Forest training or prediction pipeline.

## Technologies Used
- **Python 3**
- **Scikit-Learn** (Machine Learning & Pipelines)
- **Pandas & NumPy** (Data Manipulation)
- **SHAP** (Model Explainability)
- **Streamlit** (Web Application / Dashboard)
- **Matplotlib & Seaborn** (Data Visualization)

## Model Evaluation
The model was evaluated using 5-fold Stratified Cross-Validation on the historical dataset. 
- **Random Forest (Historical Flood Characteristics)**
  - Macro F1: `0.5823 ± 0.0334`
  - Accuracy: `0.5855`
- **Baseline (Physical Exposure Only)**
  - Macro F1: `0.3162 ± 0.0462`

*Note: The reported metrics measure performance on the historical district-classification task. They do not represent future predictive accuracy.*

## Explainability Approach
The system uses SHAP `TreeExplainer` to calculate both global feature importance across the entire dataset and local feature contributions for individual districts. This ensures transparency, allowing users to see exactly which historical factors (e.g., maximum flood duration, animal fatalities) pushed a specific classification toward or away from a given risk tier.

## How to Run the Project Locally
1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit dashboard:
   ```bash
   python -m streamlit run app.py
   ```
4. Access the dashboard at `http://localhost:8501`.

## Project Structure
- `app.py`: The main Streamlit dashboard application.
- `predict.py`: Reusable inference layer exposing the trained model and dataset retrieval functions.
- `train_models.py` / `evaluate_models_cv.py`: Model training and cross-validation scripts.
- `explain_model.py`: Script to generate SHAP and permutation importance results.
- `build_dataset.py`: ETL pipeline script to preprocess raw CSVs into the final dataset.
- `data/`: Directory containing raw and processed CSV datasets.
- `outputs/`: Directory containing generated SHAP and explainability artifacts.
- `requirements.txt`: Python dependencies.

## Data Sources & Attribution
This project relies on several external datasets. The data remains the property of their respective providers. We do not claim ownership or creation of these source datasets.
- **HydroSense Lab, IIT Delhi / Zenodo**: India Flood Inventory and DFSI-related flood datasets. Used for historical modeling and DFSI target derivation.
  - Source: [https://zenodo.org/records/16994648](https://zenodo.org/records/16994648)
- **Government of India Open Government Data (OGD)**: Local Government Directory (LGD) district data. Used for district and state mapping.
  - Source: [https://data.gov.in/resource/local-government-directory-lgd-districts](https://data.gov.in/resource/local-government-directory-lgd-districts)
- **India Meteorological Department (IMD) / NWDP**: Rainfall data accessed through the National Water Data Portal. Used **only** as contextual current rainfall information in the dashboard, and **NOT** as an input to the machine learning model.
  - Source: [https://www.nwdp.nwic.gov.in/en/dataset/rainfall-daily-imd/resource/8752174f-1d17-4aaf-8058-2eb396f50157](https://www.nwdp.nwic.gov.in/en/dataset/rainfall-daily-imd/resource/8752174f-1d17-4aaf-8058-2eb396f50157)

## Important Limitations & Responsible AI
- **Historical Analysis Only**: This system analyzes historical flood-risk patterns and is **not a future flood prediction or life-critical early-warning system**.
- **No Causal Claims**: Predictive feature importance does not imply causation.
- **Variable Data Quality**: Historical reporting completeness varies significantly across different districts and states.
- **Expert Support Tool**: Model outputs should support, not replace, expert decision-making and domain knowledge.
- **DFSI Independence**: The reference DFSI target is strictly excluded from model inputs to prevent target leakage.

## Live Demo
[Live Demo Link Placeholder]

## Acknowledgements
Project developed as a sustainability-focused AI initiative during the **1M1B AI for Sustainability Virtual Internship**.
