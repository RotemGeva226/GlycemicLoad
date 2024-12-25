# Data folder
This folder contains the raw and processed data for the project. 

## Directory Structure
- `raw/`: Contains the original, unprocessed images and any associated files directly downloaded from the source.
- `processed/`: Contains preprocessed images and additional data, such as normalized images, resized versions, and corresponding labels or embeddings.

## Raw Data
The raw data includes images and metadata from the Nutrition5k dataset, downloaded from [https://console.cloud.google.com/storage/browser/nutrition5k_dataset]

### File Structure
- `Nutrition5kModified.csv`: For each dish ID dish_[10 digit timestamp], there is a CSV entry containing the following fields:

dish_id, total_calories, total_mass, total_fat, total_carb, total_protein, num_ingrs, (ingr_1_id, ingr_1_name, ingr_1_grams, ingr_1_calories, ingr_1_fat, ingr_1_carb, ingr_1_protein, ...)

with the last 8 fields are repeated for every ingredient present in the dish.

**Note:** This is a subset of the dataset, containing only the dishes for which Claude achieved 100% accuracy in identifying the ingredients.

- `ClaudeResults.csv`: Each dish ID has a corresponding CSV entry that lists the ingredients predicted by Claude.

## Processed Data

The processed data includes images and labels that have been preprocessed to be compatible with the model.

### Processing Steps
1. Resize images to a standard size.
2. Normalize pixel values to [0, 1] or [-1, 1].
3. Generate embeddings for ingredient lists using the chosen embedding model.
4. Save preprocessed labels in `.csv` format, with columns for:
   - Image ID
   - Ingredients
   - Portion size (in grams)
   - Ingredient embeddings