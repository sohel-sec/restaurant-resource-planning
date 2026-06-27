from pathlib import Path

import pandas as pd



# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"


FORECAST_PATH = DATA_DIR / "forecast.csv"

OUTPUT_PATH = DATA_DIR / "staff_schedule.csv"



# ======================================================
# Load Forecast
# ======================================================

forecast = pd.read_csv(

    FORECAST_PATH

)



if "Predicted_Covers" not in forecast.columns:

    raise Exception(
        "forecast.csv must contain Predicted_Covers column"
    )



print("=" * 50)

print("Staff Scheduling Engine")

print("=" * 50)



print(

    "Forecast loaded:",

    len(forecast),

    "hours"

)



# ======================================================
# Staffing Rules
# ======================================================

def calculate_staff(covers):

    """
    Staffing calculation rules

    """

    manager = 1


    # One waiter for every 10 customers

    waiters = max(

        1,

        round(covers / 10)

    )


    # One kitchen staff for every 15 customers

    kitchen = max(

        1,

        round(covers / 15)

    )


    # Cashier requirement

    if covers > 50:

        cashier = 2

    else:

        cashier = 1



    return (

        kitchen,

        waiters,

        cashier,

        manager

    )



# ======================================================
# Generate Schedule
# ======================================================

schedule = []



for _, row in forecast.iterrows():


    hour = int(row["Hour"])

    covers = int(row["Predicted_Covers"])



    kitchen, waiters, cashier, manager = calculate_staff(

        covers

    )



    schedule.append(

        {

            "Hour":

                hour,


            "Expected_Covers":

                covers,


            "Kitchen_Staff":

                kitchen,


            "Waiters":

                waiters,


            "Cashiers":

                cashier,


            "Managers":

                manager

        }

    )



schedule = pd.DataFrame(

    schedule

)



# ======================================================
# Peak Hour Adjustment
# ======================================================

peak_index = schedule[

    "Expected_Covers"

].idxmax()



peak_hour = schedule.loc[

    peak_index,

    "Hour"

]



print()

print(

    "Peak hour:",

    peak_hour

)



# Add extra resources during peak

schedule.loc[

    peak_index,

    "Kitchen_Staff"

] += 1



schedule.loc[

    peak_index,

    "Waiters"

] += 1



# ======================================================
# Labor Cost Calculation
# ======================================================


STAFF_COST = {

    "Kitchen_Staff": 12,

    "Waiters": 10,

    "Cashiers": 11,

    "Managers": 20

}



schedule["Total_Staff"] = (

    schedule["Kitchen_Staff"]

    +

    schedule["Waiters"]

    +

    schedule["Cashiers"]

    +

    schedule["Managers"]

)



schedule["Labor_Cost"] = (

    schedule["Kitchen_Staff"]

    *

    STAFF_COST["Kitchen_Staff"]

    +

    schedule["Waiters"]

    *

    STAFF_COST["Waiters"]

    +

    schedule["Cashiers"]

    *

    STAFF_COST["Cashiers"]

    +

    schedule["Managers"]

    *

    STAFF_COST["Managers"]

)



# ======================================================
# Staffing Status
# ======================================================

def utilization(row):

    ratio = (

        row["Total_Staff"]

        /

        max(row["Expected_Covers"], 1)

    )


    if ratio > 0.5:

        return "Over Staffed"


    elif ratio < 0.15:

        return "Under Staffed"


    else:

        return "Optimal"



schedule["Status"] = schedule.apply(

    utilization,

    axis=1

)



# ======================================================
# Display
# ======================================================

print()

print("=" * 60)

print("STAFF REQUIREMENT")

print("=" * 60)



print(

    schedule.to_string(

        index=False

    )

)



print()

print("=" * 60)

print("Daily Summary")

print("=" * 60)



print(

    "Peak Hour:",

    peak_hour

)


print(

    "Maximum Covers:",

    schedule["Expected_Covers"].max()

)


print(

    "Total Staff Hours:",

    schedule["Total_Staff"].sum()

)


print(

    "Estimated Labor Cost:",

    schedule["Labor_Cost"].sum()

)



# ======================================================
# Save
# ======================================================


schedule.to_csv(

    OUTPUT_PATH,

    index=False

)



print()

print("Saved:")

print(

    OUTPUT_PATH

)