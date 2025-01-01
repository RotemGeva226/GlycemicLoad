import pandas as pd
import re


def extract_num_of_ingredients(path_dish_metadata: str) -> None:
    """
    This function receives dish metadata of Nutrition5k dataset and extract for each dish the number of ingredients.
    This includes ingredients that fall under sauce category.
    :param path_dish_metadata: A path to the csv file that contains the dish metadata.
    """
    data = pd.read_csv(path_dish_metadata, header=None)
    data['ingr_count'] = data.iloc[:, 1:].apply(lambda row: sum(row.astype(str).str.startswith('ingr')), axis=1)
    result_df = pd.DataFrame({
        'dish_id': data.iloc[:, 0],  # The first column is the dish ID
        'ingr_count': data['ingr_count']  # The count of 'ingr' prefixes
    })
    result_df.to_csv('dish_ingr_count.csv', index=False)
    print("Results saved to 'dish_ingr_count.csv'")


def extract_num_of_ingredients_without_sauce_dishes(path_ingr: str, path_dish_metadata: str):
    """
    This function receives dish metadata of Nutrition5k dataset and extract for each dish the number of ingredients.
    This does not include ingredients that fall under sauce category.
    :param path_ingr: A path to the csv that contains all the ingredients.
    :param path_dish_metadata: A path to the csv that contains the dish metadata.
    :return: Creates a csv with two columns: DishID and NumOfIngredients
    """
    df_ingr = pd.read_csv(path_ingr)
    df_dish_metadata = pd.read_csv(path_dish_metadata, header=None)
    sauces = df_ingr[df_ingr['IsSauce'] == 'Yes']
    sauces = sauces['ingr'].tolist()

    # Function to count "ingr" cells and apply the condition
    def count_ingredients(row):
        ingr_count = 0
        # Convert row to a list for easier handling
        row_values = row.tolist()

        # Skip the first column (dish_id) and check until second-to-last element
        for i in range(1, len(row_values) - 1):
            current_value = str(row_values[i])
            next_value = str(row_values[i + 1])

            if current_value.startswith('ingr_'):
                ingr_count += 1
                # Check if the next value is in special_list
                if next_value in sauces:
                    ingr_count -= 1

        return ingr_count

    # Apply the function to count ingredients per row
    result_df = pd.DataFrame({
        'dish_id': df_dish_metadata.iloc[:, 0],
        'ingr_count': df_dish_metadata.apply(count_ingredients, axis=1)
    })

    # Print results for verification
    print("\nInput DataFrame:")
    print(df_dish_metadata)
    print("\nResult DataFrame:")
    print(result_df)

    # Save the result to a new CSV file
    result_df.to_csv('dish_ingr_count_without_sauce.csv', index=False)
    print("\nResults saved to 'dish_ingr_count_without_sauce.csv'")


def is_contained(ls1: list, ls2: list) -> tuple:
    """
    This function checks if each element of ls1 is partially contained in ls2.
    Also, it checks if there are elements in ls1 that do not exist in ls2.
    :param ls1: The narrowed down list.
    :param ls2: The extended list.
    """

    def preprocess_string(s):
        # Convert to lowercase for case-insensitive comparison
        return re.sub(r'[^\w\s]', '', s.lower())  # To remove commas from items

    match_results = []
    not_in_ls2 = []  # The ingredients the models missed.
    not_in_ls1 = []  # The ingredients that are in ls2 but not in ls1.

    # Preprocess and tokenize ls2 items
    ls2_tokenized = [preprocess_string(item).split() for item in ls2]

    for item1 in ls1:
        # Preprocess and tokenize item1
        item1_tokens = preprocess_string(item1).split()

        # Check if any token in item1_tokens is present in any tokenized item from ls2
        match_found = any(any(token in item2 for token in item1_tokens) for item2 in ls2_tokenized)
        match_results.append((item1, match_found))

        # If no match found, add to not_in_ls2
        if not match_found:
            not_in_ls2.append(item1)

    matched_items = [item1 for item1, match in match_results if match]

    ls1_tokenized = [preprocess_string(item).split() for item in ls1]
    for item2 in ls2:
        item2_tokens = preprocess_string(item2).split()
        match_found = any(any(token in item1 for token in item2_tokens) for item1 in ls1_tokenized)
        if not match_found:
            not_in_ls1.append(item2)

    print(f"The matching items are: {matched_items}.")
    print(f"The extra items are: {not_in_ls2}.")
    print(f"The missing items are: {not_in_ls1} ")
    return matched_items, not_in_ls2, not_in_ls1


def calculate_iis(ingr_actual: list = None, ingr_predicted: list = None, mode: str = 'manual') -> float:
    """
    This function calculates Ingredients Identification Score (IIS), similar to F1 score.
    :param mode: Automatic uses lists, while manual uses manually inserted tp, fn and fp.
    :param ingr_actual: All the ingredients the models should identify.
    :param ingr_predicted: All the ingredients the models identified.
    :return: IIS
    """
    global tp, fp, fn
    print(f"Calculating IIS using mode: {mode}...")
    shared_ingr, extra_ingr, missing_ingr = is_contained(ingr_predicted, ingr_actual)
    match mode:
        case 'automatic':
            tp = len(shared_ingr)  # How many ingr overlap
            fn = len(missing_ingr)  # How many ingr in actual and not in predicted
            fp = len(extra_ingr)  # How many ingredients appear in predicted but not in actual?
        case 'manual':
            tp = int(input("Enter TP (num of matching items)."))
            fn = int(input("Enter FN (items found in actual and not in predicted)."))
            fp = int(input("Enter FP (items in predicted that not in actual)."))
    if tp == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    iis = float("{:.2f}".format((2 * precision * recall) / (precision + recall)))  # Ingredients Identification Score
    print(f"IIS is: {iis}")
    return iis


def calc_actual_and_predicted_ingredients(actual_path: str, predicted_path: str, ingredients_path: str,
                                          dish_id: str) -> tuple:
    """
    This function calculates the actual ingredients as they are listed in Nutrition5k dataset.
    Also, the function calculates the predicted ingredients from the responses of the models.
    :param actual_path: Path to Nutrition5k dish data csv.
    :param predicted_path: Path to models results csv.
    :param ingredients_path: Path to ingredients metadata of Nutrition5k.
    :param dish_id: Dish id as it described in Nutrition5k dataset.
    :return: A tuple contains two lists, one contains the actual ingr and the other the ingr the models predicted.
    """
    print(f"Calculating ingredients for dish: {dish_id}...")
    actual_df = pd.read_csv(actual_path, header=None)
    predicted_df = pd.read_csv(predicted_path)
    ingredients_df = pd.read_csv(ingredients_path)
    sauces = ingredients_df[ingredients_df['IsSauce'] == 'Yes']['ingr'].to_numpy()

    # Calculating the predicted ingredients of the dish
    predicted_ingredients = str(predicted_df[predicted_df.iloc[:, 0] == dish_id]['Predicted Ingredients'].to_numpy())
    ls_predicted_ingredients = re.findall(r"'([^']*)'", predicted_ingredients)
    print(f"The predicted ingredients are: {ls_predicted_ingredients}, total: {len(ls_predicted_ingredients)}")

    # Calculating the actual ingredients of the dish
    row = actual_df[actual_df.iloc[:, 0] == dish_id]
    actual_ingredients = []
    for i in range(row.size - 1):
        if isinstance(row[i], pd.Series):
            value = row[i].values[0]  # Access the first element explicitly
        else:
            value = row[i]
        if type(value) == str and pd.isna(value) != True:  # If the cell contains a string
            curr_val = ' '.join(row[i].values)
            if curr_val.startswith('ingr') and row[i + 1].values[0] not in sauces:
                actual_ingredients.append(row[i + 1].values[0])
    print(f"The actual ingredients are: {actual_ingredients}, total: {len(actual_ingredients)}")

    for ingredient in actual_ingredients:  # Remove ingredients that were defined as sauces
        if is_sauce(ingredients_path, ingredient):
            print(f"Removed {ingredient} because it is a sauce.")
            actual_ingredients.remove(ingredient)

    return actual_ingredients, ls_predicted_ingredients


def is_sauce(ingredients_filepath: str, ingredient: str) -> bool:
    """
    This function checks whether an ingredient was defined as a sauce.
    :param ingredients_filepath: A path to the ingredients metadata of Nutrition5k.
    :param ingredient: Input ingredient.
    :return: True if the ingredient is a sauce and False otherwise.
    """
    ingredients_data = pd.read_csv(ingredients_filepath)
    sauces = ingredients_data[ingredients_data['IsSauce'] == 'Yes']['ingr'].tolist()
    if ingredient in sauces:
        return True
    else:
        return False


def export_results(filename: str, actual_path: str, predicted_path: str, ingredients_path: str) -> None:
    """
    This function exports classification results.
    :param filename: Name out the output file.
    :param actual_path: Path of the csv file of Nutrition5k.
    :param predicted_path: Path of the classification results of the models.
    :param ingredients_path: Path of the csv of Nutrition5k that contains ingredients.
    """
    results_df = pd.read_csv(predicted_path)
    output_df = results_df
    for dish in results_df['Dish ID'].values.tolist():
        actual_ingr, predicted_ingr = calc_actual_and_predicted_ingredients(actual_path, predicted_path,
                                                                            ingredients_path, dish)
        iis = calculate_iis(actual_ingr, predicted_ingr, mode='automatic')
        output_df.loc[output_df['Dish ID'] == dish, 'IIS'] = iis
    output_df.to_csv(filename + ".csv", index=False)

def create_new_dataset(relevant_dishes_filepath: str, curr_dataset_filepath: str) -> None:
    relevant_dishes_df = pd.read_csv(relevant_dishes_filepath)
    curr_dataset_df = pd.read_csv(curr_dataset_filepath, header=None)
    relevant_dishes_ls = relevant_dishes_df['Dish ID'].values.tolist()
    output = curr_dataset_df[curr_dataset_df.iloc[:, 0].isin(relevant_dishes_ls)]
    output.to_csv('Nutrition5kModified700.csv', index=False)

def extract_ingr(input_filepath: str):
    """
    This function extracts the ingredients from the Nutrition5k dataset.
    :param input_filepath: Input file in form of Nutrition5k dataset.
    """
    data = pd.read_csv(input_filepath, header=None)
    ingredients = []
    for index, row in data.iterrows():
        for i in range(len(row) - 1):
            current_value = row[i]
            if type(current_value) is str:
                if current_value.startswith('ingr'):
                    next_value = row[i + 1]
                    if next_value not in ingredients:
                        ingredients.append(next_value)
    res = pd.DataFrame({'Used Ingredients': ingredients})
    res.to_csv('UsedIngredients.csv', index=False)

def remove_sauces(input_filepath: str, sauces_filepath: str):
    ingr_data = pd.read_csv(input_filepath)
    sauces_data = pd.read_csv(sauces_filepath)
    sauces: list = sauces_data[sauces_data['IsSauce'] == 'Yes']['ingr'].values.tolist()
    indices_to_drop = []

    for index, row in ingr_data.iterrows():
       if row[0] in sauces:
           indices_to_drop.append(index)

    res = ingr_data.drop(indices_to_drop)
    res.to_csv('UsedIngredientsDuringClassificationWithoutSauces.csv', index=False)


if __name__ == '__main__':
    actual = r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final Project\Nutrition5k dataset\Nutrition5kModified.csv"
    predicted = r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final Project\Classification\Classification Results\ChatGPT_Results.csv"
    sauces = r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final Project\Nutrition5k dataset\nutrition5k_dataset_metadata_ingredients_metadata.csv"
    ingr = r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final Project\Classification\UsedIngredientsDuringClassification.csv"
    relevant_dishes = r"C:\Users\rotem.geva\OneDrive - Afeka College Of Engineering\Final Project\Classification\Classification Results\ClaudeResults.csv"
    curr_dataset = r"C:\Users\rotem.geva\OneDrive - Afeka College Of Engineering\Final Project\Nutrition5k dataset\nutrition5k_dataset_metadata_dish_metadata_cafe1.csv"
    create_new_dataset(relevant_dishes, curr_dataset)
