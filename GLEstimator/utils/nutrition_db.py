import pandas as pd

NUTRITION_DB_PATH = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\GLEstimator\utils\ingredients_metadata.csv"
INGREDIENT_COLUMN_NAME = "ingr"
GLYCEMIC_INDEX_COLUMN_NAME = "Glycemic Index"
CARBS_COLUMN_NAME = "carb(g)"

def get_nutrition(meal_info: dict) -> float:
    # Receive input such as {"rice": 73}
    gl = 0
    for ingredient in meal_info.keys():
        gl += get_glycemic_index(ingredient.lower()) * get_carbs_per_gram(ingredient.lower()) * meal_info[ingredient]
    return gl/100

def get_glycemic_index(ingredient_name: str) -> float:
    try:
        df = pd.read_csv(NUTRITION_DB_PATH)
        row = df[df[INGREDIENT_COLUMN_NAME] == ingredient_name]
        if not row.empty:
            return row.iloc[0][GLYCEMIC_INDEX_COLUMN_NAME]
        print(f"{ingredient_name} has no glycemic index data")
        return 0
    except Exception as e:
        raise e

def get_carbs_per_gram(ingredient_name: str) -> float:
    try:
        df = pd.read_csv(NUTRITION_DB_PATH)
        row = df[df[INGREDIENT_COLUMN_NAME] == ingredient_name]
        if not row.empty:
            return row.iloc[0][CARBS_COLUMN_NAME]
        print(f"{ingredient_name} has no carbs data")
        return 0
    except Exception as e:
        raise e


if __name__ == "__main__":
    meal_data = {"rice": 0, "chicken": 0, "broccoli": 100}
    gl = get_nutrition(meal_data)
    print(gl)