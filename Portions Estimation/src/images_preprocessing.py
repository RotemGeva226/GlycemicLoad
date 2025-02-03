# Functions for preprocessing images and ingredient labels
import torch
from google.cloud import storage
import os
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

BUCKET_NAME = "nutrition5k_dataset"
REALSENSE_OVERHEAD_PATH = "nutrition5k_dataset/imagery/realsense_overhead/"
CURRENT_DIR = os.path.dirname(os.getcwd())
PROCESSED_DATA_DIR = os.path.join(CURRENT_DIR, r"data\rgb_processed_imagenet")
RAW_DATA_DIR = os.path.join(CURRENT_DIR, r"data\raw")
INGREDIENTS_METADATA_FILEPATH = os.path.join(CURRENT_DIR, r"data/ingredients.csv")


def download_image_from_gcs(bucket_name, source_blob_name, local_temp_path) -> storage.Blob:
    """
    Download an image from Google Cloud Storage.
    :param bucket_name: Database name.
    :param source_blob_name: Image name.
    :param local_temp_path: Local path to save the image.
    :return: Local path to the downloaded image.
    """
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(local_temp_path)
    return local_temp_path

def preprocess_rgb(image_path, target_size=(224, 224),  use_imagenet=True) -> torch.Tensor:
    mean_imagenet = [0.485, 0.456, 0.406]
    std_imagenet = [0.229, 0.224, 0.225]
    mean_strand = [0.0,0.0,0.0]
    std_strand = [1.0,1.0,1.0]

    if use_imagenet:
        mean, std = mean_imagenet, std_imagenet
    else:
        mean, std = mean_strand, std_strand
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def preprocess_rgbd(image_path, target_size=(224, 224)) -> torch.Tensor:
    depth_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[1.0])
    ])
    depth_image = Image.open(image_path).convert("L")
    depth_tensor = depth_transform(depth_image)
    return depth_tensor

def get_glycemic_load(dish_id):
    ingredients_metadata_df = pd.read_csv(INGREDIENTS_METADATA_FILEPATH)
    total_gl = ingredients_metadata_df.groupby("Dish ID").get_group(dish_id)['Glycemic Load'].sum()
    return total_gl

def preprocess_dataset_nutrition5k_format(annotations_file, target_size=(224, 224), mode='split'):
    """
    Preprocess the dataset by temporarily downloading images, processing them, and saving results.
    :param mode: If combined, creates a combined tensor of 7 layers.
    If split, creates two seperate tensors: rgb (3 layers) and rgbd (4 layers).
    :param annotations_file: Input file like in Nutrition5k dataset.
    :param target_size: Size of squeezed image.
    """
    annotations = pd.read_csv(annotations_file, header=None)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    preprocessed_data = []

    for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
        dish_id = row.iloc[0]
        gcs_image_path_rgb = f"{REALSENSE_OVERHEAD_PATH}{dish_id}/rgb.png"
        gcs_image_path_depth = f"{REALSENSE_OVERHEAD_PATH}{dish_id}/depth_raw.png"
        local_temp_path_rgb = "temp_image_rgb.jpg"
        local_temp_path_depth = "temp_image_depth.jpg"

        # Downloading the image and appending new info
        try:
            download_image_from_gcs(BUCKET_NAME, gcs_image_path_rgb, local_temp_path_rgb)
            download_image_from_gcs(BUCKET_NAME, gcs_image_path_depth, local_temp_path_depth)
            preprocessed_rgb_tensor = preprocess_rgb(local_temp_path_rgb, target_size)
            preprocessed_depth_tensor = preprocess_rgbd(local_temp_path_depth, target_size)

            gl = get_glycemic_load(dish_id)

            match mode:
                case 'combined':
                    combined_preprocessed_tensor = torch.cat((preprocessed_rgb_tensor, preprocessed_depth_tensor), dim=0)  # Shape: [4, H, W]
                    output_combined_path = os.path.join(PROCESSED_DATA_DIR, f"combined_{dish_id}.pt")
                    torch.save(combined_preprocessed_tensor, output_combined_path)
                    preprocessed_data.append({
                        "image_id": dish_id,
                        "processed_combined_image_path": output_combined_path,
                        "glycemic load": gl
                    })
                case 'split':
                    output_path_rgb = os.path.join(PROCESSED_DATA_DIR, f"rgb_{dish_id}.pt")
                    output_path_depth = os.path.join(PROCESSED_DATA_DIR, f"rgbd_{dish_id}.pt")
                    torch.save(preprocessed_rgb_tensor, output_path_rgb)
                    torch.save(preprocessed_depth_tensor, output_path_depth)
                    preprocessed_data.append({
                        "image_id": dish_id,
                        "processed_path_rgb_tensor": output_path_rgb,
                        "processed_path_depth_tensor": output_path_depth,
                        "glycemic load": gl
                    })
        finally:
            rgb_full_path = os.path.join(os.getcwd(), local_temp_path_rgb)
            depth_full_path = os.path.join(os.getcwd(), local_temp_path_depth)
            if os.path.exists(rgb_full_path):
                os.remove(rgb_full_path)
            if os.path.exists(depth_full_path):
                os.remove(depth_full_path)

    pd.DataFrame(preprocessed_data).to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_annotations.csv"), index=False)

def preprocess_dataset_portions_classification(annotations_file, target_size=(224, 224)):
    """
    Preprocess the dataset by temporarily downloading images, processing them, and saving results.
    :param annotations_file: Input file: dish_id, dominate_ingredient, total_mass, mass_class
    :param target_size: Size of squeezed image.
    """
    annotations = pd.read_csv(annotations_file)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    preprocessed_data = []

    for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
        dish_id = row.iloc[0]
        mass_class = row.iloc[3]
        gcs_image_path_rgb = f"{REALSENSE_OVERHEAD_PATH}{dish_id}/rgb.png"
        local_temp_path_rgb = "temp_image_rgb.jpg"

        # Downloading the image and appending new info
        try:
            download_image_from_gcs(BUCKET_NAME, gcs_image_path_rgb, local_temp_path_rgb)
            preprocessed_rgb_tensor = preprocess_rgb(local_temp_path_rgb, target_size)

            output_combined_path = os.path.join(PROCESSED_DATA_DIR, f"{dish_id}.pt")
            torch.save(preprocessed_rgb_tensor, output_combined_path)
            preprocessed_data.append({
                "image_id": dish_id,
                "processed_image_path": output_combined_path,
                "class": mass_class
            })
        except Exception as e:
            pass

        finally:
            rgb_full_path = os.path.join(os.getcwd(), local_temp_path_rgb)
            if os.path.exists(rgb_full_path):
                os.remove(rgb_full_path)

    pd.DataFrame(preprocessed_data).to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_portions_classification.csv"), index=False)

def preprocess_dataset(annotations_file, target_size=(224, 224)) -> None:
    """
    Preprocess the dataset by temporarily downloading images, processing them, and saving results.
    :param annotations_file: Input file in format of: Dish ID,Ingredient Name,Mass (g),Carbs (g),Glycemic Index,
    Glycemic Load and Class.
    :param target_size: Size of squeezed image.
    This is only for processing rgb images!
    """
    annotations = pd.read_csv(annotations_file)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    preprocessed_data = []

    grouped_dishes = annotations.groupby("Dish ID")

    for group in tqdm(grouped_dishes.groups, total=len(grouped_dishes.groups)):
        dish_id = group
        gcs_image_path_rgb = f"{REALSENSE_OVERHEAD_PATH}{dish_id}/rgb.png"
        local_temp_path_rgb = "temp_image_rgb.jpg"

        # Downloading the image and appending new info
        try:
            download_image_from_gcs(BUCKET_NAME, gcs_image_path_rgb, local_temp_path_rgb)
            preprocessed_rgb_tensor = preprocess_rgb(local_temp_path_rgb, target_size)
            output_path_rgb = os.path.join(PROCESSED_DATA_DIR, f"rgb_{dish_id}.pt")
            torch.save(preprocessed_rgb_tensor, output_path_rgb)
            classification = grouped_dishes.get_group(dish_id)['Class'].tolist()[0]
            preprocessed_data.append({
                "image_id": dish_id,
                "processed_path_rgb_tensor": output_path_rgb,
                "class": classification
            })
        except Exception as e:
            print(f"Error processing dish {dish_id}: {e}")

        finally:
            rgb_full_path = os.path.join(os.getcwd(), local_temp_path_rgb)
            if os.path.exists(rgb_full_path):
                os.remove(rgb_full_path)

    pd.DataFrame(preprocessed_data).to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_annotations.csv"), index=False)

def visualize_preprocessed_image(image_tensor):
    # Convert the Tensor back to NumPy for visualization
    unnormalized_image = image_tensor.permute(1, 2, 0).numpy()
    plt.imshow(unnormalized_image)
    plt.axis("off")  # Turn off axes for better viewing
    plt.title("Preprocessed Image")
    plt.show()

if __name__ == "__main__":
    path = r"/Portions Estimation/data/ready_for_train/processed_portions_classification.csv"
    preprocess_dataset_portions_classification(path, (224, 224))
