from pathlib import Path
import subprocess
import pandas as pd



# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent


SRC_DIR = BASE_DIR / "src"

DATA_DIR = BASE_DIR / "data"



# ======================================================
# Run Module
# ======================================================

def run_module(filename):

    path = SRC_DIR / filename


    if not path.exists():

        print(
            "Missing:",
            path
        )

        return False


    subprocess.run(

        [
            "python3.10",
            str(path)

        ]

    )

    return True



# ======================================================
# View CSV
# ======================================================

def show_file(filename):

    path = DATA_DIR / filename


    if not path.exists():

        print(
            "\nNo data available:"
            ,
            filename
        )

        return


    df = pd.read_csv(path)


    print("\n")

    print("="*60)

    print(filename)

    print("="*60)


    print(

        df.to_string(

            index=False

        )

    )



# ======================================================
# Dashboard Summary
# ======================================================

def dashboard():


    print("\n")

    print("="*60)

    print(
        "RESTAURANT RESOURCE PLANNING DASHBOARD"
    )

    print("="*60)



    forecast = DATA_DIR / "forecast.csv"


    if forecast.exists():


        df = pd.read_csv(
            forecast
        )


        total = df["Predicted_Covers"].sum()


        peak = df.loc[

            df["Predicted_Covers"].idxmax()

        ]



        print(

            f"""
Customer Forecast

Expected Covers : {total}
Peak Hour       : {peak['Hour']}:00
Peak Covers     : {peak['Predicted_Covers']}
"""

        )



    staff = DATA_DIR / "staff_schedule.csv"


    if staff.exists():


        df = pd.read_csv(
            staff
        )


        print(

            f"""
Staff Planning

Total Staff Hours : {df['Total_Staff'].sum()}
Labor Cost        : {df['Labor_Cost'].sum()}
"""

        )



    inventory = DATA_DIR / "inventory_order.csv"


    if inventory.exists():


        df = pd.read_csv(
            inventory
        )


        orders = (

            df["Order_Quantity"]

            >0

        ).sum()



        print(

            f"""
Inventory

Items To Order : {orders}
"""

        )



# ======================================================
# Feedback Input
# ======================================================

def feedback_input():


    print("\nManager Feedback")


    date = input(
        "Date (YYYY-MM-DD): "
    )


    hour = int(

        input(
            "Hour: "
        )

    )


    predicted = int(

        input(
            "Predicted covers: "
        )

    )


    actual = int(

        input(
            "Actual covers: "
        )

    )


    weather = input(

        "Weather: "

    )



    subprocess.run(

        [

            "python3.10",

            str(
                SRC_DIR / "feedback.py"
            ),

        ]

    )



# ======================================================
# Menu
# ======================================================

def menu():


    while True:


        dashboard()


        print(
"""
1. Generate Customer Forecast
2. Generate Staff Schedule
3. Generate Inventory Order
4. View Forecast
5. View Staff Schedule
6. View Inventory
7. Submit Feedback
8. Retrain Model
9. Exit
"""
        )


        choice = input(
            "Select option: "
        )



        if choice == "1":

            run_module(
                "predict.py"
            )


        elif choice == "2":

            run_module(
                "staff_scheduler.py"
            )


        elif choice == "3":

            run_module(
                "inventory.py"
            )


        elif choice == "4":

            show_file(
                "forecast.csv"
            )


        elif choice == "5":

            show_file(
                "staff_schedule.csv"
            )


        elif choice == "6":

            show_file(
                "inventory_order.csv"
            )


        elif choice == "7":

            feedback_input()



        elif choice == "8":

            run_module(
                "retrain.py"
            )


        elif choice == "9":

            print(
                "System closed"
            )

            break


        else:

            print(
                "Invalid option"
            )



if __name__ == "__main__":

    menu()