# Functions for preprocessing images and ingredient labels
import torch
from google.cloud import storage
import os
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image


BUCKET_NAME = "nutrition5k_dataset"
REALSENSE_OVERHEAD_PATH = "nutrition5k_dataset/imagery/realsense_overhead/"
CURRENT_DIR = os.path.dirname(os.getcwd())
PROCESSED_DATA_DIR = os.path.join(CURRENT_DIR, r"data\processed_combined")
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

def preprocess_rgb(image_raw, target_size=(224, 224)) -> torch.Tensor:
    # Normalize using ImageNet means and standard deviations
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_raw).convert("RGB")
    return transform(image)

def preprocess_rgbd(image_path, target_size=(224, 224)) -> torch.Tensor:
    image = Image.open(image_path).convert("RGBA")
    rgb_image = image.split()[:3]
    depth_channel = image.split()[3]
    transform_rgb = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    rgb_tensor = transform_rgb(Image.merge("RGB", rgb_image))

    # Depth
    transform_depth = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    depth_tensor = transform_depth(depth_channel)
    rgbd_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0) # Combine both images
    return rgbd_tensor


def get_glycemic_load(dish_id):
    ingredients_metadata_df = pd.read_csv(INGREDIENTS_METADATA_FILEPATH)
    total_gl = ingredients_metadata_df.groupby("Dish ID").get_group(dish_id)['Glycemic Load'].sum()
    return total_gl


def preprocess_dataset(annotations_file, target_size=(224, 224), mode='split'):
    """
    Preprocess the dataset by temporarily downloading images, processing them, and saving results.
    :param mode: If combined, creates a combined tensor of 7 layers.
    If split, creates two seperate tensors: rgb (3 layers) and rgbd (4 layers).
    :param annotations_file: Input file.
    :param target_size: Size of squeezed image.
    """
    annotations = pd.read_csv(annotations_file, header=None)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    preprocessed_data = []

    for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
        dish_id = row.iloc[0]
        gcs_image_path_rgb = f"{REALSENSE_OVERHEAD_PATH}{dish_id}/rgb.png"
        gcs_image_path_rgbd = f"{REALSENSE_OVERHEAD_PATH}{dish_id}/depth_color.png"
        local_temp_path_rgb = "temp_image_rgb.jpg"
        local_temp_path_rgbd = "temp_image_rgbd.jpg"

        # Downloading the image and appending new info
        try:
            download_image_from_gcs(BUCKET_NAME, gcs_image_path_rgb, local_temp_path_rgb)
            download_image_from_gcs(BUCKET_NAME, gcs_image_path_rgbd, local_temp_path_rgbd)
            preprocessed_image_rgb = preprocess_rgb(local_temp_path_rgb, target_size)
            preprocessed_image_rgbd = preprocess_rgbd(local_temp_path_rgbd, target_size)

            gl = get_glycemic_load(dish_id)

            match mode:
                case 'combined':
                    combined_preprocessed_image = torch.cat((preprocessed_image_rgb, preprocessed_image_rgbd), dim=0)  # Shape: [7, H, W]
                    output_combined_path = os.path.join(PROCESSED_DATA_DIR, f"combined_{dish_id}.pt")
                    torch.save(combined_preprocessed_image, output_combined_path)
                    preprocessed_data.append({
                        "image_id": dish_id,
                        "processed_combined_image_path": output_combined_path,
                        "glycemic load": gl
                    })
                case 'split':
                    output_path_rgb = os.path.join(PROCESSED_DATA_DIR, f"rgb_{dish_id}.pt")
                    output_path_rgbd = os.path.join(PROCESSED_DATA_DIR, f"rgbd_{dish_id}.pt")
                    torch.save(preprocessed_image_rgb, output_path_rgb)
                    torch.save(preprocessed_image_rgbd, output_path_rgbd)
                    preprocessed_data.append({
                        "image_id": dish_id,
                        "processed_image_path_rgb": output_path_rgb,
                        "processed_imgae_path_rgbd": output_path_rgbd,
                        "glycemic load": gl
                    })

        finally:
            rgb_full_path = os.path.join(os.getcwd(), local_temp_path_rgb)
            rgbd_full_path = os.path.join(os.getcwd(), local_temp_path_rgbd)
            if os.path.exists(rgb_full_path):
                os.remove(rgb_full_path)
            if os.path.exists(rgbd_full_path):
                os.remove(rgbd_full_path)

    pd.DataFrame(preprocessed_data).to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_combined_annotations.csv"), index=False)



if __name__ == "__main__":
    preprocess_dataset(os.path.join(RAW_DATA_DIR, "Nutrition5kModified700.csv"), (224, 224), mode='combined')