import anthropic
import os
import base64
import google.cloud.storage
import pandas as pd
from google.cloud import storage
from pathlib import Path



os.environ["ANTHROPIC_API_KEY"] = ""

client = anthropic.Anthropic()

def identify_ingredients(image_path: google.cloud.storage.Blob) -> list:
    """
    This function outputs the ingredients that appear in the plate as a text message.
    :param image_path: The path of the image.
    :return: A list of ingredients im the image.
    """
    image_media_type = "image/png"
    image_tmp = image_path.download_as_bytes()
    image_data = base64.b64encode(image_tmp).decode("utf-8")
    prompt = ("You are a chef, be as precise as possible."
              "List only the identifiable ingredients on this plate, being as specific as possible with varieties. "
              "Write each ingredient without descriptions or additional commentary. "
              "Focus only on what's visible on the plate. Do not add additional notes to the list,"
              " that do not concern the ingredients.")
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
    text = message.content[0].text
    ingredients = [line.replace('- ', '').replace('(appears to be breakfast-style)', '').strip()
                   for line in text.split('\n')
                   if line.strip()]
    return ingredients

def ingredients_analysis(limit: int):
    """
    This function iterates cloud stored database Nutrition5k and locates the relevant dishes images.
    Then, it outputs the dish id and the predicted ingredients to a csv.
    :param limit: How many dishes the output should contain.
    """
    res = pd.DataFrame(columns=['Dish ID', 'Predicted Ingredients'])
    relevant_dishes_df = pd.read_csv(r"C:\Users\rotem.geva\OneDrive - Afeka College Of Engineering\Final "
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
    res.to_csv('ClaudeResults.csv', index=False)

ingredients_analysis(limit=700)