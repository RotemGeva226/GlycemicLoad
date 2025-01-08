import base64
import os
import google.generativeai as genai
import google

BASE_DIR = r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final Project\Classification"
GEMINI_API_KEY_FILEPATH = os.path.join(BASE_DIR, "Gemini API Key.txt")


class Gemini:
    def __init__(self):
        gemini_api_key_filepath = GEMINI_API_KEY_FILEPATH
        try:
            with open(gemini_api_key_filepath, "r") as f:
                api_key = f.read()
        except Exception as err:
                print(f'Failed to read Gemini API key: {err}')
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-1.5-flash")

    def indentify_ingredients(self, image_path: google.cloud.storage.Blob):
        """
        This function outputs the ingredients that appear in the plate as a text message.
        It is built for Gemini API.
        :param model: Gemini model.
        :param image_path: The path of the image.
        :return: A list of ingredients that appear in the image.
        """
        image_tmp = image_path.download_as_bytes()
        image_data = base64.b64encode(image_tmp).decode("utf-8")
        prompt = (
            "List the ingredients that appear in the image. For example: ['Apple', 'Beef', 'Melon'], without extra details. "
            "Do not add phrases like: Here's a list of the ingredients visible in the image:, Just list the food items you see.")
        response = self._model.generate_content(
            [{'mime_type': 'image/png', 'data': image_data}, prompt])
        output = response.text.splitlines()
        print(f'The answer from Gemini is: {output}')
        return output