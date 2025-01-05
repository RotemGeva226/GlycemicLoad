import google
import Claude, Gemini, ChatGPT
import pandas as pd
from google.cloud import storage
from pathlib import Path
import numpy as np
from typing import Union


def connect_to_nutrition5k(folder_name: str) -> google.cloud.storage.blob:
    BUCKET_NAME = "nutrition5k_dataset"
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=folder_name)
    # folder name e.g. "nutrition5k_dataset/imagery/realsense_overhead/"
    return blobs

def analyze_ingredients(limit: int, result_file_name: str, model: Union[Claude, Gemini, ChatGPT]):
    """
    This function iterates cloud stored database Nutrition5k and locates the relevant dishes images.
    Then, it outputs the dish id and the predicted ingredients to a csv.
    :param model: The model used to identify ingredients.
    :param result_file_name: The output file name.
    :param limit: How many dishes the output should contain.
    """
    try:
        res = pd.DataFrame(columns=['Dish ID', 'Predicted Ingredients'])
        relevant_dishes_df = pd.read_csv(r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final "
                                         r"Project\Nutrition5k dataset\Nutrition5kModified.csv")
        relevant_dishes = relevant_dishes_df.iloc[:, 0].values.tolist()
        blobs = connect_to_nutrition5k(folder_name="nutrition5k_dataset/imagery/realsense_overhead/")

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

def analyze_portions(input_filepath: str, actual_portions_filepath: str, result_file_name: str, model: Union[Claude, Gemini, ChatGPT]) -> None:
    """
    This function calculates portions estimation for claude.
    :param model: the model used to estimate portions.
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
            predicted_total_portion = model.identify_portions(dish, predicted_ingr)
            new_data = {'Dish ID': dish, 'Actual Portion[g]': actual_total_portion,
                        'Estimated Portion[g]': predicted_total_portion}
            res.loc[len(res)] = new_data  # Adding the new data point
            print(f'There are currently: {len(res)} data points.')

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        res.to_csv(f'{result_file_name}.csv', index=False)
        print("CSV has been saved.")

def evaluate(input_filepath: str, mode: str):
    """
    This function calculates metrics used for evaluation.
    :param input_filepath: File that contains models results.
    :param mode: The metric to calculate.
    """
    df = pd.read_csv(input_filepath)
    actual_column_name = next((col for col in df.columns if col.startswith('Actual')), None)
    estimated_column_name = next((col for col in df.columns if col.startswith('Estimated')), None)
    match mode.lower():
        case 'mae':
            df['error'] = abs(df[actual_column_name] - df[estimated_column_name])
            mae = df['error'].mean()
            print(f'The MAE is: {round(mae, 2)}')
        case 'rmse':
            df['squared_error'] = (df[estimated_column_name] - df[actual_column_name]) ** 2
            mse = df['squared_error'].mean()
            rmse = np.sqrt(mse)
            print(f'The RMSE is: {round(rmse, 2)}')
        case 'rsquared':
            tss = ((df[actual_column_name] - df[actual_column_name].mean()) ** 2).sum()
            rss = ((df[actual_column_name] - df[estimated_column_name]) ** 2).sum()
            r_squared = 1 - (rss / tss)
            print(f'The R-squared is: {round(r_squared, 2)}')

def estimate_glycemic_load(dishes_metadata: str, output_filename: str, model: Union[Claude, Gemini, ChatGPT]) -> None:
    """
    This function calculates glycemic load estimation.
    :param dishes_metadata: A csv file with dishes metadata, in the following format:
    Dish ID, Ingredient Name, Mass (g),	Carbs (g), Glycemic Index, Glycemic Load.
    :param output_filename: Name for the output csv file.
    :param model: The model used to estimate glycemic load.
    """
    dishes_metadata_df = pd.read_csv(dishes_metadata)
    res = pd.DataFrame(columns=['Dish ID', 'Actual Glycemic Load', 'Estimated Glycemic Load'])
    dishes = list(set(dishes_metadata_df['Dish ID'].values.tolist()))
    grouped_dishes = dishes_metadata_df.groupby('Dish ID')
    for dish in dishes:
        actual_gl = grouped_dishes.get_group(dish)['Glycemic Load'].sum()
        print(f'Actual glycemic load for dish {dish} is: {actual_gl}')
        predicted_gl = model.estimate_glycemic_load(dish)
        new_data = {'Dish ID': dish, 'Actual Glycemic Load': actual_gl, 'Estimated Glycemic Load': predicted_gl}
        res.loc[len(res)] = new_data
        print(f'There are currently: {len(res)} data points.')

    res.to_csv(f'{output_filename}.csv', index=False)


if __name__ == "__main__":
    actual = r"C:\Users\rotem.geva\OneDrive - Afeka College Of Engineering\Final Project\Portions Estimation\ingredients.csv"
    claude = Claude.Claude()
    estimate_glycemic_load(actual, 'ClaudeGlycemicLoadEstimationResults', claude)



