import base64
import google.cloud.storage
import pandas as pd
from google.cloud import storage
from pathlib import Path
from openai import OpenAI


client = OpenAI(api_key='')

def identify_ingredients(image_path: google.cloud.storage.Blob) -> str:
    """
    This function outputs the ingredients that appear in the plate as a text message.
    :param image_path: The path of the image.
    :return: A list of ingredients im the image.
    """
    image_tmp = image_path.download_as_bytes()
    image_data = base64.b64encode(image_tmp).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "List the food items you see in the image. For example: ['Apple', 'Melon']."
                        "Do not describe the surroundings and do not describe the food items you see.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        },
                    },
                ],
            }
        ],
    )

    pred_ingr = response.choices[0].message.content
    print(f'The predicted ingredients are:{pred_ingr}')
    return pred_ingr

def ingredients_analysis(limit: int):
    """
    This function iterates cloud stored database Nutrition5k and locates the relevant dishes images.
    Then, it outputs the dish id and the predicted ingredients to a csv.
    :param limit: How many dishes the output should contain.
    """
    try:
        res = pd.DataFrame(columns=['Dish ID', 'Predicted Ingredients'])
        relevant_dishes_df = pd.read_csv(r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final Project\Nutrition5k dataset\Nutrition5kModified.csv")
        relevant_dishes = relevant_dishes_df.iloc[:,0].values.tolist()
        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket("nutrition5k_dataset")
        blobs = bucket.list_blobs(prefix="nutrition5k_dataset/imagery/realsense_overhead/")

        for blob in blobs:
            blob_name = Path(blob.name).parent.name
            if len(res) != limit and blob_name in relevant_dishes:
                if str(blob.name).__contains__('rgb'):
                    current_dish_id = blob_name
                    if current_dish_id in relevant_dishes:
                        ingredients = identify_ingredients(blob)
                        new_data = {'Dish ID': current_dish_id, 'Predicted Ingredients': ingredients}
                        res.loc[len(res)] = new_data
                        print(f"Added new data, there are: {len(res)} data points now")
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        res.to_csv('ChatGPT_Results.csv', index=False)
        print("CSV has been saved.")


