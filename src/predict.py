
from pathlib import Path

import pandas as pd
import numpy as np

import joblib



# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

DATA_DIR = BASE_DIR / "data"


MODEL_PATH = MODEL_DIR / "cover_model.pkl"

FEATURE_PATH = MODEL_DIR / "cover_features.pkl"

SALES_PATH = DATA_DIR / "historical_sales.csv"

FORECAST_PATH = DATA_DIR / "forecast.csv"



# ======================================================
# Load Model
# ======================================================

model = joblib.load(
    MODEL_PATH
)


features = joblib.load(
    FEATURE_PATH
)



# ======================================================
# Load Historical Data
# ======================================================

history = pd.read_csv(
    SALES_PATH
)


history["Date"] = pd.to_datetime(
    history["Date"]
)


if "DayOfWeek" not in history.columns:

    history["DayOfWeek"] = (
        history["Date"]
        .dt.day_name()
    )



# ======================================================
# Prediction Function
# ======================================================

def predict_covers(
        date,
        weather="Sunny",
        holiday=0,
        promotion=0,
        reservations=50
):

    date = pd.to_datetime(
        date
    )


    day_name = date.day_name()


    rows = []



    for hour in range(11, 23):


        # ----------------------------------------------
        # Historical same day/hour pattern
        # ----------------------------------------------

        hour_history = history[
            (history["Hour"] == hour)
            &
            (
                history["DayOfWeek"]
                ==
                day_name
            )
        ]



        if len(hour_history) > 0:

            rolling_7 = (
                hour_history
                .tail(7)["Covers"]
                .mean()
            )


            lag_1_day = (
                hour_history
                .tail(1)["Covers"]
                .iloc[0]
            )


        else:

            fallback = history[
                history["Hour"] == hour
            ]


            if len(fallback) > 0:

                rolling_7 = (
                    fallback
                    .tail(7)["Covers"]
                    .mean()
                )


                lag_1_day = (
                    fallback
                    .tail(1)["Covers"]
                    .iloc[0]
                )

            else:

                rolling_7 = 0

                lag_1_day = 0



        row = {


            "Hour": hour,


            "Weekend": int(
                day_name
                in [
                    "Friday",
                    "Saturday"
                ]
            ),


            "Holiday": holiday,


            "Promotion": promotion,


            "Reservations": reservations,


            "Year": date.year,


            "Month": date.month,


            "Day": date.day,


            "DayOfYear": date.dayofyear,


            "WeekOfYear":
                int(date.isocalendar().week),


            "Lag_1_Day": lag_1_day,


            "Lag_7_Days": rolling_7,


            "Rolling_7": rolling_7,


            "Weather_Rainy":
                int(weather == "Rainy"),


            "Weather_Sunny":
                int(weather == "Sunny")

        }



        row[
            f"DayOfWeek_{day_name}"
        ] = 1



        rows.append(row)



    future = pd.DataFrame(
        rows
    )



    # ==================================================
    # Match Training Features
    # ==================================================

    for col in features:

        if col not in future.columns:

            future[col] = 0



    future = future[features]



    # ==================================================
    # Predict
    # ==================================================

    predictions = model.predict(
        future
    )


    predictions = np.maximum(
        predictions,
        0
    )



    # ==================================================
    # Smooth Hourly Demand
    # ==================================================

    predictions = (
        pd.Series(predictions)
        .rolling(
            window=3,
            center=True,
            min_periods=1
        )
        .mean()
        .round()
        .astype(int)
        .values
    )



    forecast = pd.DataFrame(
        {

            "Hour":
                range(11, 23),


            "Predicted_Covers":
                predictions

        }
    )


    return forecast



# ======================================================
# Run Prediction
# ======================================================

if __name__ == "__main__":


    forecast = predict_covers(

        date="2026-06-27",

        weather="Sunny",

        holiday=0,

        promotion=1,

        reservations=80

    )



    print("\n")

    print("=" * 40)

    print(
        "Customer Forecast"
    )

    print("=" * 40)


    print(
        forecast.to_string(
            index=False
        )
    )



    # ==================================================
    # Summary
    # ==================================================

    total_covers = (
        forecast["Predicted_Covers"]
        .sum()
    )


    peak = forecast.loc[
        forecast["Predicted_Covers"]
        .idxmax()
    ]



    print("\n")

    print(
        "Daily Summary"
    )

    print("=" * 40)


    print(
        f"Expected Covers : {total_covers}"
    )


    print(
        f"Peak Hour      : {peak['Hour']}:00"
    )


    print(
        f"Peak Covers    : {peak['Predicted_Covers']}"
    )



    # ==================================================
    # Save Forecast
    # ==================================================

    forecast.to_csv(
        FORECAST_PATH,
        index=False
    )


    print("\nForecast saved:")

    print(
        FORECAST_PATH
    )

