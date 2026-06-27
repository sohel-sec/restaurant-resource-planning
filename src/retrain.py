from pathlib import Path

import pandas as pd
import joblib

from xgboost import XGBRegressor



# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"


SALES_PATH = DATA_DIR / "historical_sales.csv"

FEEDBACK_PATH = DATA_DIR / "feedback.csv"


MODEL_PATH = MODEL_DIR / "cover_model.pkl"

FEATURE_PATH = MODEL_DIR / "cover_features.pkl"



# ======================================================
# Load Dataset
# ======================================================

sales = pd.read_csv(
    SALES_PATH
)


feedback = pd.read_csv(
    FEEDBACK_PATH
)



print("=" * 50)

print("Retraining Model")

print("=" * 50)



print(
    "Historical samples:",
    len(sales)
)


print(
    "Feedback samples:",
    len(feedback)
)



# ======================================================
# Prepare Feedback Data
# ======================================================


feedback = feedback.rename(

    columns={

        "Actual_Covers":
            "Covers"

    }

)



# Add missing columns

required_columns = [

    "Date",
    "Hour",
    "Covers",
    "Weather",
    "Reservations",
    "WalkIns",
    "Promotion",
    "Holiday"

]



for col in required_columns:


    if col not in feedback.columns:

        if col == "Weather":

            feedback[col] = "Sunny"

        else:

            feedback[col] = 0



feedback = feedback[required_columns]



# ======================================================
# Prepare Historical Data
# ======================================================


sales["Date"] = pd.to_datetime(

    sales["Date"]

)



feedback["Date"] = pd.to_datetime(

    feedback["Date"]

)



# Combine

data = pd.concat(

    [

        sales,

        feedback

    ],

    ignore_index=True

)



print(

    "Combined samples:",

    len(data)

)



# ======================================================
# Feature Engineering
# ======================================================


data["Hour"] = data["Hour"].astype(int)



data["DayOfWeek"] = (

    data["Date"]

    .dt.day_name()

)



data["Weekend"] = (

    data["DayOfWeek"]

    .isin(

        [

            "Friday",

            "Saturday"

        ]

    )

    .astype(int)

)



data["Year"] = (

    data["Date"]

    .dt.year

)



data["Month"] = (

    data["Date"]

    .dt.month

)



data["Day"] = (

    data["Date"]

    .dt.day

)



# ======================================================
# Weather Encoding
# ======================================================


if "Weather" in data.columns:


    data = pd.get_dummies(

        data,

        columns=[
            "Weather"
        ]

    )



# ======================================================
# Day Encoding
# ======================================================


data = pd.get_dummies(

    data,

    columns=[

        "DayOfWeek"

    ]

)



# ======================================================
# Lag Features
# ======================================================


if "Covers" in data.columns:


    data["Lag_1_Day"] = (

        data["Covers"]

        .shift(24)

    )


    data["Lag_7_Days"] = (

        data["Covers"]

        .shift(168)

    )


    data["Rolling_7"] = (

        data["Covers"]

        .rolling(7)

        .mean()

    )



# Fill missing lag values

data = data.fillna(0)



# ======================================================
# Training Data
# ======================================================


X = data.drop(

    columns=[

        "Covers",

        "Date"

    ]

)



y = data["Covers"]



# Remove object columns

object_columns = X.select_dtypes(

    include=[
        "object"
    ]

).columns



if len(object_columns) > 0:


    print(
        "Removing object columns:",
        list(object_columns)
    )


    X = X.drop(

        columns=object_columns

    )



# Replace invalid values

X = X.replace(

    [

        float("inf"),

        float("-inf")

    ],

    0

)



X = X.fillna(0)



# Convert bool

for col in X.columns:


    if X[col].dtype == "bool":

        X[col] = X[col].astype(int)



X = X.astype(float)



print()

print(
    "Training features:",
    len(X.columns)
)



# ======================================================
# Train XGBoost
# ======================================================


model = XGBRegressor(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42

)



model.fit(

    X,

    y

)



# ======================================================
# Save Model
# ======================================================


MODEL_DIR.mkdir(

    exist_ok=True

)



joblib.dump(

    model,

    MODEL_PATH

)



joblib.dump(

    X.columns.tolist(),

    FEATURE_PATH

)



print()

print("=" * 50)

print(
    "Model retrained successfully"
)

print("=" * 50)



print(
    "Model:",
    MODEL_PATH
)


print(
    "Features:",
    FEATURE_PATH
)