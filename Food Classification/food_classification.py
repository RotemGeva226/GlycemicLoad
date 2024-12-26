import google
from sympy.codegen.ast import Raise
import gemini_utils
import chatgpt_utils
import claude_utils
import pandas as pd
from google.cloud import storage
from pathlib import Path
import numpy as np

def main(model_name: str):
    match model_name.lower():
        case "gemini":
            model = gemini_utils.Gemini()
        case "chatgpt":
            model = chatgpt_utils.ChatGPT()
        case "claude":
            model = claude_utils.Claude()
        case _:
            print("Invalid model name")

def connect_nutrition5k(folder_name: str) -> google.cloud.storage.blob:
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket("nutrition5k_dataset")
    blobs = bucket.list_blobs(prefix=folder_name)
    # folder name e.g. "nutrition5k_dataset/imagery/realsense_overhead/"
    return blobs

def ingredients_analysis(limit: int, model_name: str, result_file_name: str):
    """
    This function iterates cloud stored database Nutrition5k and locates the relevant dishes images.
    Then, it outputs the dish id and the predicted ingredients to a csv.
    :param result_file_name: The output file name.
    :param model_name: The selected model (ChatGPT, Gemini, Claude).
    :param limit: How many dishes the output should contain.
    """
    match model_name.lower():
        case "gemini":
            model = gemini_utils.Gemini()
        case "chatgpt":
            model = chatgpt_utils.ChatGPT()
        case "claude":
            model = claude_utils.Claude()
        case _:
            Raise("Invalid model name")
    try:
        res = pd.DataFrame(columns=['Dish ID', 'Predicted Ingredients'])
        relevant_dishes_df = pd.read_csv(r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final "
                                         r"Project\Nutrition5k dataset\Nutrition5kModified.csv")
        relevant_dishes = relevant_dishes_df.iloc[:, 0].values.tolist()
        blobs = connect_nutrition5k(folder_name="nutrition5k_dataset/imagery/realsense_overhead/")

        for blob in blobs:
            if len(res) != limit:
                if str(blob.name).__contains__('rgb'):
                    current_dish_id = Path(blob.name).parent.name
                    if current_dish_id in relevant_dishes:
                        ingredients = model.identify_ingredients(blob)
                        new_data = {'Dish ID': current_dish_id, 'Predicted Ingredients': ingredients}
                        res.loc[len(res)] = new_data
                        print(f"Added new data, current data points is: {len(res)}")
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        res.to_csv(f'{result_file_name}.csv', index=False)

def portions_analysis(input_filepath: str, actual_portions_filepath: str, result_file_name: str) -> None:
    """
    This function calculates portions estimation for claude.
    :param result_file_name: The output file name.
    :param input_filepath: File that contains models classification results with dish id.
    :param actual_portions_filepath: File that contains Nutrition5k dataset content.
    """
    actual_portions_df = pd.read_csv(actual_portions_filepath, header=None)
    input_df = pd.read_csv(input_filepath)
    res = pd.DataFrame(columns=['Dish ID', 'Actual Portion[g]', 'Estimated Portion[g]'])
    dishes = input_df['Dish ID'].values.tolist()
    try:
        for dish in dishes:
            actual_total_portion = actual_portions_df[actual_portions_df.iloc[:, 0] == dish].iloc[0,2]
            predicted_ingr = input_df[input_df['Dish ID'] == dish].iloc[0, 1]
            predicted_total_portion = identify_portions(dish, predicted_ingr)
            new_data = {'Dish ID': dish, 'Actual Portion[g]': actual_total_portion, 'Estimated Portion[g]': predicted_total_portion}
            res.loc[len(res)] = new_data # Adding the new data point
            print(f'There are currently: {len(res)} data points.')

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        res.to_csv(f'{result_file_name}.csv', index=False)
        print("CSV has been saved.")

def calculate_metric(input_filepath: str, mode: str):
    df = pd.read_csv(input_filepath)
    match mode.lower():
        case 'mae':
            df['error'] = abs(df['Actual Portion[g]'] - df['Estimated Portion[g]'])
            mae = df['error'].mean()
            print(f'The MAE is: {round(mae, 2)}')
        case 'rmse':
            df['squared_error'] = (df['Estimated Portion[g]'] - df['Actual Portion[g]']) ** 2
            mse = df['squared_error'].mean()
            rmse = np.sqrt(mse)
            print(f'The RMSE is: {round(rmse, 2)}')
        case 'rsquared':
            tss = ((df['Actual Portion[g]'] - df['Actual Portion[g]'].mean()) ** 2).sum()
            rss = ((df['Actual Portion[g]'] - df['Estimated Portion[g]']) ** 2).sum()
            r_squared = 1 - (rss / tss)
            print(f'The R-squared is: {round(r_squared, 2)}')

if __name__ == "__main__":
    pass

