import base64
import google
from openai import OpenAI
import os

BASE_DIR = r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final Project\Classification"
CHATGPT_API_KEY_FILEPATH = os.path.join(BASE_DIR, "ChatGPT API Key.txt")

class ChatGPT:
    def __init__(self):
        chatgpt_api_key_filepath = CHATGPT_API_KEY_FILEPATH
        try:
            with open(chatgpt_api_key_filepath, "r") as f:
                api_key = f.read()
        except Exception as err:
                print(f'Failed to read ChatGPT API key: {err}')
        self._client = OpenAI(api_key=api_key)

    def identify_ingredients(self, image_path: google.cloud.storage.Blob) -> str:
        """
        This function outputs the ingredients that appear in the plate as a text message.
        :param client: ChatGPT model.
        :param image_path: The path of the image.
        :return: A list of ingredients im the image.
        """
        image_tmp = image_path.download_as_bytes()
        image_data = base64.b64encode(image_tmp).decode("utf-8")
        response = self._client.chat.completions.create(
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
        print(f'ChatGPT identified the following ingredients:{pred_ingr}')
        return pred_ingr