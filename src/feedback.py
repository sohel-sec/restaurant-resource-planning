from pathlib import Path
import pandas as pd
from datetime import datetime



# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"


FEEDBACK_PATH = DATA_DIR / "feedback.csv"



# ======================================================
# Add Manager Feedback
# ======================================================

def save_feedback(
        date,
        hour,
        predicted,
        actual,
        weather="Sunny",
        reason=""
):


    new_feedback = pd.DataFrame(

        [
            {

                "Date": date,

                "Hour": hour,

                "Predicted_Covers": predicted,

                "Actual_Covers": actual,

                "Weather": weather,

                "Reason": reason,

                "Created_At":
                    datetime.now()

            }
        ]

    )


    if FEEDBACK_PATH.exists():

        old = pd.read_csv(
            FEEDBACK_PATH
        )


        data = pd.concat(

            [
                old,
                new_feedback
            ],

            ignore_index=True

        )


    else:

        data = new_feedback



    data.to_csv(

        FEEDBACK_PATH,

        index=False

    )


    print(
        "Feedback saved:"
    )

    print(
        FEEDBACK_PATH
    )



# ======================================================
# Test
# ======================================================

if __name__ == "__main__":


    save_feedback(

        date="2026-06-27",

        hour=20,

        predicted=26,

        actual=18,

        weather="Rainy",

        reason="Heavy rain"

    )