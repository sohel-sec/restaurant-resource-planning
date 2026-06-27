
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================
# Load Data
# ======================================================

sales = pd.read_csv(
    DATA_DIR / "historical_sales.csv"
)

weather = pd.read_csv(
    DATA_DIR / "weather.csv"
)

holidays = pd.read_csv(
    DATA_DIR / "holidays.csv"
)


print("Sales Columns:")
print(sales.columns.tolist())



# ======================================================
# Merge Weather
# ======================================================

if "Weather" not in sales.columns:

    weather = weather.drop_duplicates(
        subset=["Date"]
    )

    df = sales.merge(
        weather,
        on="Date",
        how="left"
    )

else:

    df = sales.copy()



# ======================================================
# Merge Holiday
# ======================================================

if "Holiday" not in df.columns:

    holidays = holidays.drop_duplicates(
        subset=["Date"]
    )

    df = df.merge(
        holidays,
        on="Date",
        how="left"
    )



# ======================================================
# Clean Columns
# ======================================================


# Remove duplicate columns created by pandas

duplicate_columns = [
    col for col in df.columns
    if col.endswith("_x")
]


for col in duplicate_columns:

    original = col.replace("_x", "")

    if original not in df.columns:

        df.rename(
            columns={
                col: original
            },
            inplace=True
        )

    else:

        df.drop(
            columns=[col],
            inplace=True
        )



duplicate_columns = [
    col for col in df.columns
    if col.endswith("_y")
]


df.drop(
    columns=duplicate_columns,
    inplace=True,
    errors="ignore"
)



# ======================================================
# Missing Value Handling
# ======================================================


if "Weather" not in df.columns:

    df["Weather"] = "Unknown"


df["Weather"] = (
    df["Weather"]
    .fillna("Unknown")
)



if "Holiday" not in df.columns:

    df["Holiday"] = 0


df["Holiday"] = (
    df["Holiday"]
    .fillna(0)
    .astype(int)
)



# ======================================================
# Date Feature Engineering
# ======================================================

df["Date"] = pd.to_datetime(
    df["Date"]
)


df["Year"] = (
    df["Date"]
    .dt.year
)


df["Month"] = (
    df["Date"]
    .dt.month
)


df["Day"] = (
    df["Date"]
    .dt.day
)


df["DayOfYear"] = (
    df["Date"]
    .dt.dayofyear
)


df["WeekOfYear"] = (
    df["Date"]
    .dt.isocalendar()
    .week
    .astype(int)
)



# ======================================================
# Weekend Feature
# ======================================================

# df["IsWeekend"] = (
#     df["DayOfWeek"]
#     .isin(
#         [
#             "Friday",
#             "Saturday"
#         ]
#     )
#     .astype(int)
# )



# ======================================================
# Sort Data
# ======================================================

df = df.sort_values(
    [
        "Date",
        "Hour"
    ]
)



# ======================================================
# Lag Features
# ======================================================


df["Lag_1_Day"] = (
    df.groupby("Hour")["Covers"]
    .shift(1)
)



df["Lag_7_Days"] = (
    df.groupby("Hour")["Covers"]
    .shift(7)
)



df["Rolling_7"] = (
    df.groupby("Hour")["Covers"]
    .transform(
        lambda x:
        x.rolling(
            window=7,
            min_periods=1
        )
        .mean()
    )
)



df.fillna(
    0,
    inplace=True
)



# ======================================================
# Encode Categorical Variables
# ======================================================


categorical_columns = [
    "DayOfWeek",
    "Weather"
]


df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)



# ======================================================
# Remove Date
# ======================================================

df.drop(
    columns=["Date"],
    inplace=True
)



# ======================================================
# Split Features / Target
# ======================================================


TARGET = "Covers"

X = df.drop(
    columns=[
        TARGET,
        "WalkIns"
    ],
    errors="ignore"
)


y = df[TARGET]



# ======================================================
# Convert Remaining Objects
# For XGBoost Compatibility
# ======================================================


for col in X.columns:

    if X[col].dtype == "object":

        X[col] = pd.factorize(
            X[col]
        )[0]



# ======================================================
# Train Test Split
# ======================================================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)



# ======================================================
# Save Processed Data
# ======================================================


X_train.to_csv(
    PROCESSED_DIR / "X_train.csv",
    index=False
)


X_test.to_csv(
    PROCESSED_DIR / "X_test.csv",
    index=False
)


y_train.to_csv(
    PROCESSED_DIR / "y_train.csv",
    index=False
)


y_test.to_csv(
    PROCESSED_DIR / "y_test.csv",
    index=False
)



# ======================================================
# Validation
# ======================================================


print("\n" + "=" * 60)

print(
    "Preprocessing Completed Successfully"
)

print("=" * 60)


print(
    f"Training Samples : {len(X_train)}"
)


print(
    f"Testing Samples  : {len(X_test)}"
)


print(
    f"Features         : {X_train.shape[1]}"
)


print("\nData Types:")

print(
    X_train.dtypes.value_counts()
)


print("=" * 60)

