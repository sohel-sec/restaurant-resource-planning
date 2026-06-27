
from pathlib import Path

import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

import joblib


# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# ======================================================
# Load Training Data
# ======================================================

X_train = pd.read_csv(
    DATA_DIR / "X_train.csv"
)


X_test = pd.read_csv(
    DATA_DIR / "X_test.csv"
)


y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
).values.ravel()


y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).values.ravel()



print("=" * 50)
print("Dataset Loaded")
print("=" * 50)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "Features:",
    X_train.shape[1]
)



# ======================================================
# XGBoost Model
# ======================================================

model = XGBRegressor(

    n_estimators=500,

    learning_rate=0.05,

    max_depth=6,

    min_child_weight=3,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1

)



# ======================================================
# Train Model
# ======================================================

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train
)



# ======================================================
# Prediction
# ======================================================

predictions = model.predict(
    X_test
)


# Customer count cannot be negative

predictions = np.maximum(
    predictions,
    0
)



# ======================================================
# Evaluation Metrics
# ======================================================

mae = mean_absolute_error(
    y_test,
    predictions
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)



# Safe MAPE
# Ignore rows where actual covers = 0

mask = y_test != 0


if mask.sum() > 0:

    mape = np.mean(
        np.abs(
            (
                y_test[mask]
                -
                predictions[mask]
            )
            /
            y_test[mask]
        )
    )

else:

    mape = 0



# ======================================================
# Display Results
# ======================================================

print("\n" + "=" * 50)

print(
    "Customer Forecast Model Evaluation"
)

print("=" * 50)


print(
    f"MAE  : {mae:.2f}"
)


print(
    f"RMSE : {rmse:.2f}"
)


print(
    f"MAPE : {mape * 100:.2f}%"
)

smape = np.mean(
    2 * np.abs(predictions - y_test)
    /
    (
        np.abs(y_test)
        +
        np.abs(predictions)
        +
        1e-8
    )
)


print(
    f"SMAPE: {smape * 100:.2f}%"
)


print("=" * 50)



# ======================================================
# Save Model
# ======================================================

model_path = MODEL_DIR / "cover_model.pkl"


joblib.dump(
    model,
    model_path
)



# ======================================================
# Save Feature Names
# ======================================================

feature_path = MODEL_DIR / "cover_features.pkl"


joblib.dump(
    list(X_train.columns),
    feature_path
)



print("\nModel saved:")
print(model_path)


print("\nFeatures saved:")
print(feature_path)



# ======================================================
# Feature Importance
# ======================================================

importance = pd.DataFrame(
    {
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }
)


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


importance.to_csv(
    MODEL_DIR / "feature_importance.csv",
    index=False
)


print("\nTop Features:")

print(
    importance.head(10)
)

