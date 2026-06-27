from pathlib import Path
import pandas as pd



# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"


FORECAST_PATH = DATA_DIR / "forecast.csv"
RECIPES_PATH = DATA_DIR / "recipes.csv"
INVENTORY_PATH = DATA_DIR / "inventory.csv"

OUTPUT_PATH = DATA_DIR / "inventory_order.csv"



# ======================================================
# Helper
# ======================================================

def normalize_columns(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    return df



def title_case(x):

    return (
        str(x)
        .strip()
        .title()
    )



# ======================================================
# Load Files
# ======================================================

forecast = pd.read_csv(
    FORECAST_PATH
)


recipes = pd.read_csv(
    RECIPES_PATH
)


inventory = pd.read_csv(
    INVENTORY_PATH
)



recipes = normalize_columns(
    recipes
)


inventory = normalize_columns(
    inventory
)



print("Recipe columns:")
print(recipes.columns.tolist())


print("\nInventory columns:")
print(inventory.columns.tolist())



# ======================================================
# Recipe Conversion
# ======================================================

if "ingredient" not in recipes.columns:


    print(
        "\nWide recipe format detected..."
    )


    if "dish" not in recipes.columns:

        raise Exception(
            "recipes.csv requires Dish column"
        )


    ingredient_columns = [

        c

        for c in recipes.columns

        if c != "dish"

    ]


    recipes = recipes.melt(

        id_vars=[
            "dish"
        ],

        value_vars=ingredient_columns,

        var_name="ingredient",

        value_name="quantity"

    )



# rename recipe columns

recipes = recipes.rename(

    columns={

        "ingredient":
            "Ingredient",

        "quantity":
            "Quantity"

    }

)



recipes["Ingredient"] = (

    recipes["Ingredient"]

    .apply(title_case)

)


recipes["Quantity"] = pd.to_numeric(

    recipes["Quantity"],

    errors="coerce"

)



recipes = recipes.dropna()



# ======================================================
# Inventory Conversion
# ======================================================


if "ingredient" not in inventory.columns:

    raise Exception(
        f"Inventory needs ingredient column. Found {inventory.columns.tolist()}"
    )



inventory = inventory.rename(

    columns={

        "ingredient":
            "Ingredient"

    }

)



inventory["Ingredient"] = (

    inventory["Ingredient"]

    .apply(title_case)

)



# Find stock column automatically

possible_stock = [

    "stock",

    "quantity",

    "qty",

    "available",

    "current",

    "amount"

]



stock_column = None



for col in possible_stock:

    if col in inventory.columns:

        stock_column = col

        break



if stock_column is None:


    # If only 2 columns exist,
    # assume second column is stock

    if len(inventory.columns) == 2:

        stock_column = (
            inventory.columns[1]
        )

    else:

        raise Exception(
            f"Cannot find stock column. Found {inventory.columns.tolist()}"
        )



inventory = inventory.rename(

    columns={

        stock_column:
            "Stock"

    }

)



inventory["Stock"] = pd.to_numeric(

    inventory["Stock"],

    errors="coerce"

).fillna(0)



# ======================================================
# Forecast Demand
# ======================================================

total_covers = int(

    forecast["Predicted_Covers"]

    .sum()

)


print(
    f"\nExpected customers: {total_covers}"
)



# ======================================================
# Ingredient Requirement
# ======================================================

requirements = {}



for _, row in recipes.iterrows():

    ingredient = row["Ingredient"]

    qty = row["Quantity"]


    requirements[ingredient] = (

        requirements.get(
            ingredient,
            0
        )

        +

        qty * total_covers

    )



# ======================================================
# Inventory Calculation
# ======================================================

result = []



for ingredient, required in requirements.items():


    current = inventory.loc[

        inventory["Ingredient"]

        ==
        ingredient,

        "Stock"

    ]



    available = (

        current.iloc[0]

        if len(current) > 0

        else 0

    )


    order = max(

        required - available,

        0

    )


    result.append(

        {

            "Ingredient":
                ingredient,

            "Required":
                round(required,2),

            "Available":
                round(available,2),

            "Order_Quantity":
                round(order,2)

        }

    )



output = pd.DataFrame(
    result
)



# ======================================================
# Display
# ======================================================

print("\n")

print("="*55)

print(
    "INGREDIENT ORDER FORECAST"
)

print("="*55)


print(
    output.to_string(
        index=False
    )
)



# ======================================================
# Save
# ======================================================

output.to_csv(

    OUTPUT_PATH,

    index=False

)


print("\nSaved:")

print(
    OUTPUT_PATH
)