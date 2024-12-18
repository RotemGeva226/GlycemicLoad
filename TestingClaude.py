import anthropic
import os
import base64
import google.cloud.storage
import pandas as pd
from google.cloud import storage
from pathlib import Path



os.environ["ANTHROPIC_API_KEY"] = "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"

client = anthropic.Anthropic()

def connect_nutrition5k(folder_name: str) -> google.cloud.storage.blob:
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket("nutrition5k_dataset")
    blobs = bucket.list_blobs(prefix=folder_name)
    # folder name e.g. "nutrition5k_dataset/imagery/realsense_overhead/"
    return blobs

def identify_ingredients(image_path: google.cloud.storage.Blob) -> str:
    """
    This function outputs the ingredients that appear in the plate as a text message.
    :param image_path: The path of the image.
    :return: A list of ingredients im the image.
    """
    image_media_type = "image/png"
    image_tmp = image_path.download_as_bytes()
    image_data = base64.b64encode(image_tmp).decode("utf-8")
    prompt = ("List the food items you see in the image. For example: ['Apple', 'Melon']."
              "Do not describe the surroundings and do not describe the food items you see.")
    message = anthropic.Anthropic().messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data
                        }
                    },
                    {"type": "text", "text": prompt}
                ],
            }
        ],
    )
    ingredients = message.content[0].text
    print(f'Returned ingredients are: {ingredients}')
    return ingredients

def ingredients_analysis(limit: int):
    """
    This function iterates cloud stored database Nutrition5k and locates the relevant dishes images.
    Then, it outputs the dish id and the predicted ingredients to a csv.
    :param limit: How many dishes the output should contain.
    """
    try:
        res = pd.DataFrame(columns=['Dish ID', 'Predicted Ingredients'])
        relevant_dishes_df = pd.read_csv(r"C:\Users\rotem.geva\OneDrive - Afeka College Of Engineering\Final Project"
                                         r"\Nutrition5k dataset\Dishes for classification test- full.csv")
        relevant_dishes = relevant_dishes_df['Dish ID'].values.tolist()
        already_viewed_dishes = pd.read_csv(r"C:\Users\rotem.geva\PycharmProjects\GlycemicLoad\ClaudeResultsSummary.csv")
        already_viewed_dishes = already_viewed_dishes['Dish ID'].values.tolist()
        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket("nutrition5k_dataset")
        blobs = bucket.list_blobs(prefix="nutrition5k_dataset/imagery/realsense_overhead/")

        for blob in blobs:
            blob_name = Path(blob.name).parent.name
            if len(res) != limit and blob_name in relevant_dishes and blob_name not in already_viewed_dishes:
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
        res.to_csv('ClaudeResults.csv', index=False)
        print("CSV has been saved.")

def identify_portions(dish_id: str, ingr: list) -> str:
    """
    This function outputs the portion of each ingredient that appear in the plate as a text message.
    :param dish_id: Dish id according to Nutrition5k indexing.
    :param ingr: A list of the ingredients the model identified in the image.
    :return: A list that contains the portions of the ingredients.
    """
    blobs = connect_nutrition5k(folder_name="nutrition5k_dataset/imagery/realsense_overhead/")
    for blob in blobs:
        if blob.name.__contains__(dish_id):
            if blob.name.__contains__('depth_color.png'):
                img_depth: blob = blob
            elif blob.name.__contains__('rgb.png'):
                img_rgb: blob = blob
                break
    image_media_type = "image/png"
    image_tmp1 = img_rgb.download_as_bytes()
    image_tmp2 = img_depth.download_as_bytes()
    image_data1 = base64.b64encode(image_tmp1).decode("utf-8")
    image_data2 = base64.b64encode(image_tmp2).decode("utf-8")
    prompt = (f"Ingredients in the image: {ingr}. IMPORTANT: Estimate the total serving portion in grams. "
              f"Your ENTIRE response must be EXACTLY in this format: [NUMBER]g - NO additional words, explanation, "
              f"or commentary. Just the number followed by 'g'.")
    message = anthropic.Anthropic().messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data1
                        }
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data2
                        }
                    },
                    {"type": "text", "text": prompt}
                ],
            }
        ],
    )
    portions = message.content[0].text
    print(f'The portion size of {dish_id} is: {portions}')
    return portions


identify_portions('dish_1558026756', ['Bacon', 'Cantaloupe', 'Honeydew melon'])