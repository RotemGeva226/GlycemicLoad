# Data folder
This folder contains the raw and processed data for the project. 

## Directory Structure
- `raw/`: Contains the original, unprocessed images and any associated files directly downloaded from the source.
- `processed/`: Contains preprocessed images and additional data, such as normalized images, resized versions, and corresponding labels or embeddings.

## Raw Data
The raw data includes images and metadata from the Nutrition5k dataset, downloaded from [https://console.cloud.google.com/storage/browser/nutrition5k_dataset]

### File Structure
- `Nutrition5kModified700.csv`: For each dish ID dish_[10 digit timestamp], there is a CSV entry containing the following fields:

dish_id, total_calories, total_mass, total_fat, total_carb, total_protein, num_ingrs, (ingr_1_id, ingr_1_name, ingr_1_grams, ingr_1_calories, ingr_1_fat, ingr_1_carb, ingr_1_protein, ...)

with the last 8 fields are repeated for every ingredient present in the dish.

**Note:** This is a subset of the dataset, containing only the dishes for which Claude achieved 100% accuracy in identifying the ingredients.

- `nutrition5k_dataset_metadata_ingredients_metadata.csv`: For each ingredient from the full Nutrition5k dataset, there is a CSV entry containing the following fields:
ingredient name, id, cal/g, fat(g), carb(g), protein(g), the "IsSauce" column in the dataset indicates whether an ingredient is 
classified as a sauce (Yes) or not. This includes ingredients that were not visible on camera during data collection and and manually sourced glycemic index data.

## Processed Data

The processed data includes images and labels that have been preprocessed to be compatible with the model.

### Ingredients Data
`ingredients.csv` - This dataset contains detailed information about dishes, their ingredients, and their respective nutritional values, designed for analyzing and estimating the glycemic load (GL) of meals. Each dish is broken 
down into individual ingredients, providing their mass, carbohydrate content, glycemic index (GI), and glycemic load (GL).

### File Structure
The dataset is stored in a CSV file with the following columns:
- Dish ID - Dish ID as it is written in original Nutrition5k dataset.
- Ingredient Name - Name of each ingredient in the dish (e.g., "Pasta", "Banana").
- Mass (g) - Mass of the ingredient in grams.
- Carbs (g) - Carbohydrate content of the ingredient for the given mass, in grams.
- Glycemic Index (GI) - Glycemic index of the ingredient (a measure of how much it raises blood sugar levels).
- Glycemic Load (GL) - Glycemic load of the ingredient, calculated as: GL = (GI * Carbs) / 100.

### Images Data
### Processing Steps
- The preprocessing of the RGB image involves several steps to prepare it for input into a neural network. First, the image is resized to a target size using `transforms.Resize`. Then, it is converted to a tensor representation using `transforms.ToTensor`, which scales pixel values to the range [0, 1]. Finally, the image is normalized using `transforms.Normalize` with specified mean and standard deviation values ([0.485, 0.456, 0.406] for mean and [0.229, 0.224, 0.225] for standard deviation), 
aligning it with the normalization used during the pretraining of common CNN models like ResNet. The image is read in RGB format using the `Image.open` method and converted to ensure consistent color channels before applying the transformations.

- The preprocessing for an RGBD image involves separating it into its RGB and depth channels. The RGB channels are merged into a single image using `Image.merge` and processed with a transformation pipeline that resizes the image, converts it to a tensor, 
and normalizes it using predefined mean and standard deviation values. The depth channel is processed separately with a pipeline that resizes and converts it to a tensor. 
Finally, the RGB and depth tensors are concatenated along the channel dimension using `torch.cat` to form an RGBD tensor, 
enabling the model to process both color and depth information simultaneously.
