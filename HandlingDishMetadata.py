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

def extract_num_of_ingredients_without_sauce_dishes(path_ingr:str, path_dish_metadata:str):
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
    match_results = []
    not_in_ls2 = [] # The ingr the model missed.

    for item1 in ls1:
        # Check if item1 is partially contained in any item of list2
        match_found = any(item1.lower() in item2.lower() for item2 in ls2)
        match_results.append((item1, match_found))
        # If no match found, add to not_in_list2
        if not match_found:
            not_in_ls2.append(item1)

    matched_items = [item1 for item1, match in match_results if match] # The ingr the model predicted correctly.
    return matched_items, not_in_ls2

def calculate_iis(ingr_actual: list, ingr_predicted: list) -> float:
    """
    This function calculates Ingredients Identification Score (IIS), similar to F1 score.
    :param ingr_actual: All the ingredients the model should identify.
    :param ingr_predicted: All the ingredients the model identified.
    :return: IIS
    """
    shared_ingr, missed_ingr = is_contained(ingr_actual, ingr_predicted)
    tp = len(shared_ingr) # How many ingr overlap
    fn = len(missed_ingr) # How many ingr in actual and not in predicted
    fp = abs(len(ingr_predicted) - len(ingr_actual)) # How many ingredients appear in predicted but not in actual?
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    iis = (2 * precision * recall)/(precision + recall) # Ingredients Identification Score
    return iis

def calc_actual_and_predicted_ingredients(actual_path: str, predicted_path: str, ingredients_path: str, dish_id: str):
    """
    This function calculates the actual ingredients as they are listed in Nutrition5k dataset.
    Also, the function calculates the predicted ingredients from the responses of the model.
    :param actual_path: Path to Nutrition5k dish data csv.
    :param predicted_path: Path to model results csv.
    :param ingredients_path: Path to ingredients metadata of Nutrition5k.
    :param dish_id: Dish id as it described in Nutrition5k dataset.
    :return: A tuple contains two lists, one contains the actual ingr and the other the ingr the model predicted.
    """
    actual_df = pd.read_csv(actual_path, header=None)
    predicted_df = pd.read_csv(predicted_path)
    ingredients_df = pd.read_csv(ingredients_path)
    sauces = ingredients_df[ingredients_df['IsSauce'] == 'Yes']['ingr'].to_numpy()

    # Calculating the predicted ingredients of the dish
    predicted_ingredients = str(predicted_df[predicted_df.iloc[:,0] == dish_id]['Predicted Ingredients'].to_numpy())
    ls_predicted_ingredients = re.findall(r"'([^']*)'", predicted_ingredients)
    print(f"The predicted ingredients are: {ls_predicted_ingredients}, total: {len(ls_predicted_ingredients)}")

    # Calculating the actual ingredients of the dish
    row = actual_df[actual_df.iloc[:,0] == dish_id]
    actual_ingredients = []
    for i in range(row.size - 1):
        if row[i].values.dtype == object and pd.isna(row[i].values) != True: # If the cell contains a string
            curr_val = ' '.join(row[i].values)
            if curr_val.startswith('ingr') and row[i+1].values[0] not in sauces:
                actual_ingredients.append(row[i+1].values[0])
    print(f"The actual ingredients are: {actual_ingredients}, total: {len(actual_ingredients)}")

    return actual_ingredients, ls_predicted_ingredients


if __name__ == '__main__':
    actual = r"C:\Users\rotem.geva\OneDrive - Afeka College Of Engineering\Final Project\Nutrition5k dataset\nutrition5k_dataset_metadata_dish_metadata_cafe1.csv"
    predicted = r"C:\Users\rotem.geva\OneDrive - Afeka College Of Engineering\Final Project\Nutrition5k dataset\Scripts\ClaudeResults.csv"
    ingredients = r"C:\Users\rotem.geva\OneDrive - Afeka College Of Engineering\Final Project\Nutrition5k dataset\nutrition5k_dataset_metadata_ingredients_metadata.csv"
    compare_predicted_to_actual(actual, predicted, ingredients, "dish_1558031019")

# extract_num_of_ingredients_without_sauce_dishes(path_ingr=r"C:\Users\rotem.geva\OneDrive - Afeka College Of Engineering\פרויקט גמר\Nutrition5k dataset\nutrition5k_dataset_metadata_ingredients_metadata.csv",
#                                                 path_dish_metadata=r"C:\Users\rotem.geva\OneDrive - Afeka College Of Engineering\פרויקט גמר\Nutrition5k dataset\nutrition5k_dataset_metadata_dish_metadata_cafe1.csv")

