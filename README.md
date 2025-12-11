# Predictive Maintenance using Random Forest & Monte-Carlo Simulation

## Project Overview
Modern industrial machines generate continuous IoT sensor data such as vibration, temperature, humidity, and pressure. Unexpected failures cause downtime, safety risks, and increased maintenance costs.

This project develops a Predictive Maintenance System that:
- Classifies machine health into Normal, Warning, and Failure.
- Forecasts the number of machines likely to fail in the next 10 days using Monte-Carlo Simulation.

---

## Objectives

### 1. Machine Health Classification
Predict the machine state using a Random Forest Classifier:
- Normal  
- Warning  
- Failure  

### 2. Failure Forecasting
Use Monte-Carlo Simulation to model sensor drift and future uncertainty to estimate:
- Expected machine failures for each of the next 10 days.

---

## Dataset Description

| Component | Details |
|----------|---------|
| Source | Internal Industrial Maintenance Dataset |
| Files Used | Temperature, Humidity, Pressure, Vibration, PLC Logs, Fault Annotations |
| Final Rows After Merge | 144 |
| Sampling Rate | Every 10 minutes |
| Sensors Used | Vibration, Temperature, Pressure, Humidity |

### Class Distribution Before Balancing

| Label | Count |
|-------|-------|
| Normal | 121 |
| Warning | 17 |
| Failure | 6 |

<img width="1750" height="522" alt="image" src="https://github.com/user-attachments/assets/f99cee4c-de76-4211-83f2-747c22ee38d9" />


A significant class imbalance was present and addressed using SMOTE oversampling.

---

## Methodology

| Stage | Method | Purpose |
|--------|--------|---------|
| Data Pre-processing | Timestamp merge, feature selection | Align and clean sensor data |
| Scaling | StandardScaler | Improves model stability and supports SMOTE |
| Imbalance Handling | SMOTE oversampling | Helps the model learn from limited failure samples |
| ML Model | Random Forest Classifier | Robust for small datasets and noisy signals |
| Forecasting | Monte-Carlo Simulation | Models uncertainty and future sensor degradation |

---

## Model Training Results

### Train-Test Split
Stratified 80% training and 20% testing split.

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|----------|--------|----------|---------|
| Failure | 1.00 | 0.83 | 0.91 | 6 |
| Normal | 0.96 | 0.99 | 0.98 | 121 |
| Warning | 0.93 | 0.76 | 0.84 | 17 |

Overall Accuracy: **95.83%**

---

## Confusion Matrix



<img width="447" height="289" alt="image" src="https://github.com/user-attachments/assets/4bf8ac99-a430-4ad0-b6d2-865922b8737e" />

---
## Machine-Level 10-Day Failure Forecast

<img width="1779" height="787" alt="Screenshot 2025-12-11 120929" src="https://github.com/user-attachments/assets/b8c7ed0a-5f8d-4cf3-8cf3-01fe9f93ee94" />
Here we can forecast the machine failures for each 10 days

---
##  10-Day Failure Risk Summary

<img width="1726" height="605" alt="image" src="https://github.com/user-attachments/assets/b445ee2e-ace0-4a12-9425-55bfef4d86ba" />

Here's the Graphical representation of 10-Day Failure Risk Summary

<img width="1764" height="449" alt="Screenshot 2025-12-11 121447" src="https://github.com/user-attachments/assets/4d07de06-9431-428a-8374-c62ba25e9600" />



