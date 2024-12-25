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

model = genai.GenerativeModel("gemini-1.5-flash")

def identify_ingredients(image_path: google.cloud.storage.Blob):
    """
    This function outputs the ingredients that appear in the plate as a text message.
    :param image_path: The path of the image.
    :return: A list of ingredients im the image.
    """
    image_tmp = image_path.download_as_bytes()
    image_data = base64.b64encode(image_tmp).decode("utf-8")
    prompt = ("List the ingredients that appear in the image. For example: ['Apple', 'Beef', 'Melon'], without extra details. "
              "Do not add phrases like: Here's a list of the ingredients visible in the image:, Just list the food items you see.")
    response = model.generate_content(
        [{'mime_type': 'image/png', 'data': image_data}, prompt])
    output = response.text.splitlines()
    print(f'The answer from Gemini is: {output}')
    return output

def ingredients_analysis(limit: int):
    """
    This function iterates cloud stored database Nutrition5k and locates the relevant dishes images.
    Then, it outputs the dish id and the predicted ingredients to a csv.
    :param limit: How many dishes the output should contain.
    """
    try:
        res = pd.DataFrame(columns=['Dish ID', 'Predicted Ingredients'])
        relevant_dishes_df = pd.read_csv(r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final "
                                         r"Project\Nutrition5k dataset\Nutrition5kModified.csv")
        relevant_dishes = relevant_dishes_df.iloc[:, 0].values.tolist()
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
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        res.to_csv('GeminiResults091224.csv', index=False)

ingredients_analysis(limit=433)