import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm


def extract_ingredients(row, sauces) -> list:
    """
    This function extracts ingredients from a row from the csv file (Nutrtion5k dataset).
    :param sauces: A list of ignored ingredients, that cannot be seen through the image.
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
    This function preprocesses the ingredients' dataset.
    :param annotations_file: the input file that contains all dishes metadata from Nutrition5k dataset.
    :param metadata_file: the input file that contains all ingredients metadata from Nutrition5k dataset.
    """
    annotations = pd.read_csv(annotations_file, header=None, on_bad_lines='skip')
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
        is_all_exist = all(ingredient in remained_ingredients for ingredient in ingredients) # Returns true if all
        # ingredients exist in remained_ingredients, are ingredients that they all have glycemic indexes.
        if is_all_exist: # Continue only if all ingredients have glycemic index value in metadata file.
            for ingredient in ingredients:
                if ingredient not in sauces:
                    mass = extract_ingredient_content(row, ingredient, mode='mass')
                    carbs = extract_ingredient_content(row, ingredient, mode='carbs')
                    glycemic_index = metadata_filtered[metadata_filtered['ingr'] == ingredient]['Glycemic Index'].iloc[0]
                    glycemic_load = carbs * glycemic_index / 100
                    new_row_data = {'Dish ID': dish_id, 'Ingredient Name': ingredient, "Mass (g)": mass, "Carbs (g)": carbs,
                                    "Glycemic Index": glycemic_index, "Glycemic Load": glycemic_load}
                    res.loc[len(res)] = new_row_data
    res.to_csv(os.path.join(os.path.dirname(os.getcwd()), r'data/ingredients-cafe2.csv'), index=False)

def classify_ingredients(input_filepath: str):
    """
    Adds column "Class" to the input file, according to glycemic load of the dish.
    If the dish if below 10, the class is 1, if between 11 and 19, the class is 2, and if above 20, the class is 3.
    :param input_filepath: A path to the input file.
    """
    df = pd.read_csv(input_filepath)
    grouped_dishes = df.groupby('Dish ID')
    for group in grouped_dishes.groups:
        gl = grouped_dishes.get_group(group)['Glycemic Load'].sum()
        if gl <= 10:
            df.loc[df['Dish ID'] == group, 'Class'] = 1
        elif 10 < gl <= 19:
            df.loc[df['Dish ID'] == group, 'Class'] = 2
        elif gl > 19:
            df.loc[df['Dish ID'] == group, 'Class'] = 3
    df.to_csv(input_filepath, index=False)

def prepare_dataset_single_ingredient(annotations_filepath, sauces_filepath):
    annotations = pd.read_csv(annotations_filepath, header=None)
    sauces_df = pd.read_csv(sauces_filepath)
    res = pd.DataFrame(columns=["Dish ID", "Mass (g)"])
    sauces = sauces_df[sauces_df['IsSauce'] == 'Yes']['ingr'].tolist()
    for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
        dish_id = row.iloc[0]
        dish_ingredients = extract_ingredients(row, sauces)
        if len(dish_ingredients) == 1:
            dish_mass = extract_ingredient_content(row, dish_ingredients[0], mode='mass')
            new_row_data = {'Dish ID': dish_id, "Mass (g)": dish_mass}
            res.loc[len(res)] = new_row_data
    res.to_csv("SingleDishDatasetCafe2.csv", index=False)


if __name__ == '__main__':
    # dirname = os.path.dirname(os.getcwd())
    # input_filepath = os.path.join(dirname, r"data/raw/nutrition5k_dataset_metadata_dish_metadata_cafe2.csv")
    # sauces_filepath = os.path.join(dirname, r"data/raw/nutrition5k_dataset_metadata_ingredients_metadata.csv")
    folder_path = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\data\single_ingredient_images"
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files_no_extension = [Path(f).stem for f in files]
    csv_filepath = r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\Portions Estimation\data\ingredients\SingleDishDataset.csv"
    df = pd.read_csv(csv_filepath)
    df_copy = df
    all_dishes = df['Dish ID'].tolist()
    for dish_id in all_dishes:
        if dish_id not in files_no_extension:
            df_copy = df_copy.drop(df_copy[df_copy['Dish ID']==dish_id].index)
    df_copy.to_csv('SingleDishDatasetFixed.csv', index=False)









