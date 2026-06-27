import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
from pathlib import Path 


NUM_DAYS = 730  # 2 years of data
START_DATE = datetime(2023, 1, 1)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


random.seed(42)
np.random.seed(42)

# -----------------------------
# Restaurant Hours
# -----------------------------
HOURS = list(range(11, 23))  # 11 AM - 10 PM

# Peak demand multipliers
HOURLY_WEIGHTS = {
    11: 0.40,
    12: 0.80,
    13: 1.00,
    14: 0.70,
    15: 0.40,
    16: 0.35,
    17: 0.60,
    18: 0.90,
    19: 1.20,
    20: 1.10,
    21: 0.70,
    22: 0.30
}

WEATHER = ["Sunny", "Cloudy", "Rainy"]

INGREDIENTS = {
    "Chicken": {
        "ShelfLife": 3,
        "LeadTime": 2,
        "UsagePerCover": 0.22
    },
    "Rice": {
        "ShelfLife": 180,
        "LeadTime": 7,
        "UsagePerCover": 0.12
    },
    "Tomato": {
        "ShelfLife": 6,
        "LeadTime": 2,
        "UsagePerCover": 0.05
    },
    "Onion": {
        "ShelfLife": 15,
        "LeadTime": 4,
        "UsagePerCover": 0.03
    },
    "Oil": {
        "ShelfLife": 365,
        "LeadTime": 10,
        "UsagePerCover": 0.01
    }
}

RECIPES = [
    {
        "Dish": "Chicken Rice",
        "Chicken": 0.25,
        "Rice": 0.15,
        "Tomato": 0.05,
        "Onion": 0.03,
        "Oil": 0.01
    },
    {
        "Dish": "Grilled Chicken",
        "Chicken": 0.30,
        "Rice": 0.05,
        "Tomato": 0.04,
        "Onion": 0.02,
        "Oil": 0.01
    },
    {
        "Dish": "Fried Rice",
        "Chicken": 0.12,
        "Rice": 0.20,
        "Tomato": 0.03,
        "Onion": 0.03,
        "Oil": 0.02
    }
]


# -----------------------------
# Holidays
# -----------------------------
holiday_dates = []

for i in range(NUM_DAYS):

    date = START_DATE + timedelta(days=i)

    if random.random() < 0.08:
        holiday_dates.append(date.date())


# -----------------------------
# Generate Sales
# -----------------------------
rows = []

for i in range(NUM_DAYS):

    date = START_DATE + timedelta(days=i)

    weekday = date.strftime("%A")

    weekend = weekday in ["Friday", "Saturday"]

    weather = random.choices(
        WEATHER,
        weights=[0.55, 0.25, 0.20]
    )[0]

    holiday = date.date() in holiday_dates

    promotion = random.random() < 0.15

    reservations = random.randint(20, 90)

    base_customers = 90

    if weekend:
        base_customers += 45

    if holiday:
        base_customers += 35

    if promotion:
        base_customers += 20

    if weather == "Rainy":
        base_customers -= 30

    base_customers += np.random.randint(-10, 11)

    base_customers = max(base_customers, 20)

    for hour in HOURS:

        weight = HOURLY_WEIGHTS[hour]

        expected = int(base_customers * weight / 8)

        walkins = max(0, expected - reservations // len(HOURS))

        covers = max(
            0,
            int(np.random.normal(expected, 4))
        )

        rows.append({
            "Date": date.date(),
            "Hour": hour,
            "DayOfWeek": weekday,
            "Month": date.month,
            "Weekend": int(weekend),
            "Weather": weather,
            "Holiday": int(holiday),
            "Promotion": int(promotion),
            "Reservations": reservations,
            "WalkIns": walkins,
            "Covers": covers
        })


sales_df = pd.DataFrame(rows)

sales_df.to_csv(
    os.path.join(DATA_DIR / "historical_sales.csv"),
    index=False
)

# -----------------------------
# Weather
# -----------------------------
weather_df = sales_df[["Date", "Weather"]].drop_duplicates()

weather_df.to_csv(
    os.path.join(DATA_DIR / "weather.csv"),
    index=False
)

# -----------------------------
# Holidays
# -----------------------------
holiday_df = pd.DataFrame({
    "Date": holiday_dates,
    "Holiday": 1
})

holiday_df.to_csv(
    os.path.join(DATA_DIR / "holidays.csv"),
    index=False
)

# -----------------------------
# Ingredients
# -----------------------------
ingredient_rows = []

for item, values in INGREDIENTS.items():

    ingredient_rows.append({
        "Ingredient": item,
        **values
    })

ingredient_df = pd.DataFrame(ingredient_rows)

ingredient_df.to_csv(
    os.path.join(DATA_DIR / "ingredients.csv"),
    index=False
)

# -----------------------------
# Recipes
# -----------------------------
recipe_df = pd.DataFrame(RECIPES)

recipe_df.to_csv(
    os.path.join(DATA_DIR / "recipes.csv"),
    index=False
)

# -----------------------------
# Inventory
# -----------------------------
inventory = []

for ingredient in INGREDIENTS:

    inventory.append({
        "Ingredient": ingredient,
        "CurrentStock": random.randint(15, 120)
    })

inventory_df = pd.DataFrame(inventory)

inventory_df.to_csv(
    os.path.join(DATA_DIR / "inventory.csv"),
    index=False
)

# -----------------------------
# Feedback
# -----------------------------
feedback_df = pd.DataFrame(columns=[
    "Date",
    "PredictedCovers",
    "ActualCovers",
    "Difference",
    "Reason"
])

feedback_df.to_csv(
    os.path.join(DATA_DIR / "feedback.csv"),
    index=False
)

print("=" * 60)
print("Restaurant Dataset Generated Successfully")
print("=" * 60)
print(f"Sales Records      : {len(sales_df)}")
print(f"Weather Records   : {len(weather_df)}")
print(f"Holidays          : {len(holiday_df)}")
print(f"Ingredients       : {len(ingredient_df)}")
print(f"Recipes           : {len(recipe_df)}")
print(f"Inventory Items   : {len(inventory_df)}")
print("=" * 60)
