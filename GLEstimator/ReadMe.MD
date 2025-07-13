# GLEstimator
The **GLEstimator** module is a core component in the GlycemicLoad pipeline. It integrates ingredient recognition results with portion estimations to calculate the Glycemic Load (GL) of a meal.

## Purpose
This directory wraps up the final step:
1. Combines detected ingredients (from LLM).
2. Accepts estimated portion sizes (in grams) for each ingredient from vision model.
3. Looks up nutritional values - especially glycemic index (GI) and carbohydrate content.
4. Computes Glycemic Load using the formula: GL = [GI × Available Carbs (g)]/100

## Directory Status
- `chains`: Contains Langchain full flow.
- `llm`: Includes all API for interacting with Claude.
- `model`: Provides vision model setup to predict portions in grams.
- `utils`: Handles local databse API.

 

