import base64
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain.schema.runnable import RunnableLambda
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

llm = ChatAnthropic(
    model = "claude-3-5-sonnet-20241022",
)


def invoke(image_data: str, prompt: str) -> str:
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data,
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    )

    response = llm.invoke([message])
    return response


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

def extract_ingredients(image_path: str) -> str:
    base64_image = encode_image_to_base64(image_path)
    prompt = f"""List the food items you see in the image. For example: ['Apple', 'Melon'].
            Do not describe the surroundings and do not describe the food items you see."""
    return invoke(base64_image, prompt)

claude_chain = RunnableLambda(lambda img_path: {"ingredients": extract_ingredients(img_path)})

if __name__ == "__main__":
    image_path = r"C:\Users\rotem.geva\Desktop\rgb.png"
    res = extract_ingredients(image_path)
    print(res)