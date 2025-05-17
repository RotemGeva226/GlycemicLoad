from langchain.schema.runnable import RunnableParallel, RunnableLambda
from model.model import predict_portions
from llm.claude import extract_ingredients
from utils.nutrition_db import get_nutrition

parallel = RunnableParallel({
    "ingredients": RunnableLambda(extract_ingredients),
    "portions": RunnableLambda(predict_portions)
})

def gl_wrapper(inputs):
    ingredient = inputs["ingredients"]
    grams = inputs["portions"]
    meal_info = {ingredient: grams}
    gl = get_nutrition(meal_info)
    return gl

final_chain = parallel | RunnableLambda(gl_wrapper)