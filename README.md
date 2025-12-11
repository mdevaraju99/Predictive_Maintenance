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

Insert confusion matrix image:

```md
![Confusion Matrix](results/confusion_matrix.png)
