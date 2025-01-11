import pandas as pd
import os
from tqdm import tqdm


def extract_ingredients(row, sauces) -> list:
    """
    This function extracts ingredients from a row from the csv file (Nutrtion5k dataset).
    :param row: contains ingredients from a row.
    :return: ingredients in the row.
    """
    row_values = row.tolist()
    ingredients = []
    for i in range(1, len(row_values) - 1):  # First column is dish id
        current_value = str(row_values[i])
        next_value = str(row_values[i + 1])

        if current_value.startswith('ingr_'):
            if next_value not in sauces:
                ingredients.append(next_value)
    return ingredients


def add_unique_items(list1, list2) -> list:
    """
    This function adds unique items from two lists.
    :param list1: list of items.
    :param list2: list of items.
    :return: list that contains unique items.
    """
    for item in list2:
        if item not in list1:
            list1.append(item)
    return list1


def extract_ingredient_content(row, ingredient, mode) -> str or None:
    """
    This function extracts the nutritional content of an ingredient from a row.
    :param row: contains ingredients from a row.
    :param ingredient: name of the ingredient.
    :param mode: mass or carbs.
    :return: nutritional content of the ingredient.
    """
    add = 0
    match mode:
        case 'mass':
            add = 1
        case 'carbs':
            add = 4
    ls = row.tolist()
    index = ls.index(ingredient)
    if index + add < len(ls):
        return ls[index + add]


def preprocess_ingredients_dataset(annotations_file, metadata_file) -> None:
    """
    This function preprocesses the ingredients dataset.
    :param annotations_file: the input file that contains all dishes metadata from Nutrition5k dataset.
    :param metadata_file: the input file that contains all ingredients metadata from Nutrition5k dataset.
    """
    annotations = pd.read_csv(annotations_file, header=None)
    metadata = pd.read_csv(metadata_file)
    metadata_filtered = metadata.dropna(subset=['Glycemic Index']) # Remove ingredients without glycemic index.
    sauces = metadata_filtered[metadata_filtered['IsSauce'] == 'Yes']['ingr'].tolist()
    remained_ingredients = metadata_filtered['ingr'].tolist() # Lists the ingredients that were left after removing the
    # ones without glycemic index.
    res = pd.DataFrame(
        columns=["Dish ID", "Ingredient Name", "Mass (g)", "Carbs (g)", "Glycemic Index", "Glycemic Load"])
    for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
        dish_id = row.iloc[0]
        ingredients = extract_ingredients(row, sauces)
        for ingredient in ingredients:
            if ingredient in remained_ingredients and ingredient not in sauces:
                mass = extract_ingredient_content(row, ingredient, mode='mass')
                carbs = extract_ingredient_content(row, ingredient, mode='carbs')
                glycemic_index = metadata_filtered[metadata_filtered['ingr'] == ingredient]['Glycemic Index'].iloc[0]
                glycemic_load = carbs * glycemic_index / 100
                new_row_data = {'Dish ID': dish_id, 'Ingredient Name': ingredient, "Mass (g)": mass, "Carbs (g)": carbs,
                                "Glycemic Index": glycemic_index, "Glycemic Load": glycemic_load}
                res.loc[len(res)] = new_row_data
    res.to_csv(os.path.join(os.path.dirname(os.getcwd()), r'data/ingredients-110124.csv'), index=False)

if __name__ == '__main__':
    dirname = os.path.dirname(os.getcwd())
    input_filepath = os.path.join(dirname, r"data/raw/nutrition5k_dataset_metadata_dish_metadata_cafe1.csv")
    sauces_filepath = os.path.join(dirname, r"data/raw/nutrition5k_dataset_metadata_ingredients_metadata.csv")
    preprocess_ingredients_dataset(input_filepath, sauces_filepath)



