import base64
import anthropic
import os
import google
from google.cloud import storage
from food_classification import connect_nutrition5k

BASE_DIR = r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final Project\Classification"
CLAUDE_API_KEY_FILEPATH = os.path.join(BASE_DIR, "Claude API Key.txt")

class Claude:
    def __init__(self):
        calude_api_key_filepath = CLAUDE_API_KEY_FILEPATH
        try:
            with open(calude_api_key_filepath, "r") as f:
                api_key = f.read()
        except Exception as err:
                print(f'Failed to read Claude API key: {err}')
        self._client = anthropic.Anthropic(api_key=api_key)

def identify_ingredients(self, image_path: google.cloud.storage.Blob) -> str:
    """
    This function outputs the ingredients that appear in the plate as a text message.
    :param self:
    :param image_path: The path of the image.
    :return: A list of ingredients im the image.
    """
    image_media_type = "image/png"
    image_tmp = image_path.download_as_bytes()
    image_data = base64.b64encode(image_tmp).decode("utf-8")
    prompt = ("List the food items you see in the image. For example: ['Apple', 'Melon']."
              "Do not describe the surroundings and do not describe the food items you see.")
    message = self._client.messages.create(
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
    print(f'Claude identified the following ingredients: {ingredients}')
    return ingredients

def identify_portions(self, dish_id: str, ingr: list) -> str:
    """
    This function outputs the portion of each ingredient that appear in the plate as a text message.
    :param self:
    :param dish_id: Dish id according to Nutrition5k indexing.
    :param ingr: A list of the ingredients the models previously identified in the image.
    :return: The portion size in grams.
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
    message = self._client.messages.create(
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
    print(f'According to Claude, the portion size of {dish_id} is: {portions}')
    return portions