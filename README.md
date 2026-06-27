# Restaurant Resource Planning System

## Self-Learning Forecasting and Resource Optimization Platform


## Overview

The Restaurant Resource Planning System is an AI-based application that predicts customer demand, optimizes staff scheduling, calculates ingredient requirements, and improves automatically using manager feedback.

The system reduces:

- Food waste
- Labor cost
- Staff shortage
- Ingredient shortage


---

# Features

## 1. Customer Demand Forecasting

Predicts hourly customer covers using an XGBoost regression model.

Input features:

- Historical sales
- Hour
- Day of week
- Weekend
- Holiday
- Weather
- Promotion
- Previous demand
- Rolling average demand


Output:
Hourly customer prediction
Daily expected covers
Peak customer hour



---

## 2. Staff Scheduling Engine

Generates optimized staff requirements based on predicted customers.


Roles:

- Kitchen Staff
- Waiters
- Cashiers
- Managers


Output:


data/staff_schedule.csv



Generated information:

- Required staff per hour
- Total staff hours
- Estimated labor cost


---

## 3. Inventory Forecasting

Calculates ingredient requirements using:

Customer Forecast
+
Recipe Usage
+
Current Inventory

Purchase Requirement



Output:


data/inventory_order.csv



---

## 4. Self-Learning Feedback System

Managers can submit corrections:

Example:


Predicted Customers: 120

Actual Customers: 85

Reason: Heavy Rain



Feedback is stored and used for model retraining.

Learning process:


Historical Data
+
Manager Feedback
|
v
Updated Forecast Model



---

# Machine Learning Model

Algorithm:


XGBoost Regression



Why XGBoost:

- Good performance on tabular data
- Handles complex relationships
- Fast training
- Supports feature importance


---

# Project Structure



restaurant-resource-planning/

│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│ ├── historical_sales.csv
│ ├── weather.csv
│ ├── holidays.csv
│ ├── recipes.csv
│ ├── ingredients.csv
│ ├── inventory.csv
│ ├── feedback.csv
│ ├── forecast.csv
│ └── staff_schedule.csv
│
├── models/
│ ├── cover_model.pkl
│ └── cover_features.pkl
│
└── src/
├── data_generator.py
├── preprocess.py
├── train_cover.py
├── predict.py
├── staff_scheduler.py
├── inventory.py
├── feedback.py
└── retrain.py



---

# Installation


Create environment:


```bash
python3.10 -m venv .venv

Activate:

source .venv/bin/activate

Install packages:

pip install -r requirements.txt
Requirements
pandas
numpy
scikit-learn
xgboost
joblib
Running the Application

Start dashboard:

python3.10 app.py

Menu:

1. Generate Customer Forecast
2. Generate Staff Schedule
3. Generate Inventory Order
4. Submit Feedback
5. Retrain Model
6. Exit
Training Model

Train forecasting model:

python3.10 src/train_cover.py

Generated models:

models/cover_model.pkl

models/cover_features.pkl
Prediction

Run:

python3.10 src/predict.py

Example:

Hour    Predicted Covers

11          13
12          17
13          19
19          24
20          22
Staff Scheduling

Run:

python3.10 src/staff_scheduler.py

Example:

Hour   Covers   Kitchen   Waiters

19       24        3        3
20       22        1        2
Retraining

Run:

python3.10 src/retrain.py

Process:

Historical Data
+
Feedback Data
=
New Improved Model
Model Performance

Model:

XGBoost Regression

Dataset:

8760 hourly records

Evaluation:

MAE   : 3.04

RMSE  : 3.78

SMAPE : 54.21%

Important features:

Weekend
Rolling demand
Weather
Holiday
Previous day demand
Future Improvements
Real-time POS integration
Weather API integration
Automatic supplier ordering
Mobile dashboard
Cloud deployment
Advanced deep learning models
Conclusion

The Restaurant Resource Planning System provides an intelligent solution for restaurant operations by combining:

Machine learning forecasting
Staff optimization
Inventory planning
Continuous learning

The system helps restaurants reduce waste, control costs, and improve customer service.

