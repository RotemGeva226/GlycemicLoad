import os
import base64
import google.cloud.storage
import pandas as pd
from google.cloud import storage
from pathlib import Path
import google.generativeai as genai


os.environ["GOOGLE_API_KEY"] = ""

api_key = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-pro")

def identify_ingredients(image_path: google.cloud.storage.Blob):
    """
    This function outputs the ingredients that appear in the plate as a text message.
    :param image_path: The path of the image.
    :return: A list of ingredients im the image.
    """
    image_tmp = image_path.download_as_bytes()
    image_data = base64.b64encode(image_tmp).decode("utf-8")
    prompt = ("List the ingredients of the dish for me,only the ingredients is important, not its location or shape. "
              "don't list sauces, oil, sugar, salt, and spices.")
    response = model.generate_content(
        [{'mime_type': 'image/png', 'data': image_data}, prompt])
    input_list = response.text.splitlines()
    ingredients = [item.lstrip('* ').strip() for item in input_list if item.startswith('*')]
    return ingredients

def ingredients_analysis(limit: int):
    """
    This function iterates cloud stored database Nutrition5k and locates the relevant dishes images.
    Then, it outputs the dish id and the predicted ingredients to a csv.
    :param limit: How many dishes the output should contain.
    """
    res = pd.DataFrame(columns=['Dish ID', 'Predicted Ingredients'])
    relevant_dishes_df = pd.read_csv(r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final "
                                  r"Project\Nutrition5k dataset\dish_ingr_count_without_sauce.csv")
    relevant_dishes = relevant_dishes_df[relevant_dishes_df['ingr_count'] > 3]['dish_id'].tolist()
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket("nutrition5k_dataset")
    blobs = bucket.list_blobs(prefix="nutrition5k_dataset/imagery/realsense_overhead/")

    for blob in blobs:
        if len(res) != limit:
            if str(blob.name).__contains__('rgb'):
                current_dish_id = Path(blob.name).parent.name
                if current_dish_id in relevant_dishes:
                    ingredients = identify_ingredients(blob)
                    new_data = {'Dish ID': current_dish_id, 'Predicted Ingredients': ingredients}
                    res.loc[len(res)] = new_data
                    print(f"Added new data, current data points is: {len(res)}")
    res.to_csv('GeminiResults.csv', index=False)

ingredients_analysis(limit=1)