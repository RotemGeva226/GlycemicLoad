# Functions for preprocessing images and ingredient labels
from google.cloud import storage
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

BUCKET_NAME = "nutrition5k_dataset"
REALSENSE_OVERHEAD_PATH = "nutrition5k_dataset/imagery/realsense_overhead/"
CURRENT_DIR = os.path.dirname(os.getcwd())
PROCESSED_DATA_DIR = os.path.join(CURRENT_DIR, r"data\processed")
RAW_DATA_DIR = os.path.join(CURRENT_DIR, r"data\raw")


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


def preprocess_image(image_path, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocess a single image: resize and normalize pixel values.
    :param image_path: Local path to the image.
    :param target_size: Target size for resizing the image.
    :return: Preprocessed image as a NumPy array.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image


def preprocess_dataset(annotations_file, target_size=(224, 224)):
    """
    Preprocess the dataset by temporarily downloading images, processing them, and saving results.
    :param annotations_file:
    :param target_size:
    """
    annotations = pd.read_csv(annotations_file, header=None)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    preprocessed_data = []

    for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
        dish_id = row.iloc[0]
        gcs_image_path = f"{REALSENSE_OVERHEAD_PATH}dish_{dish_id}/rgb.png"
        local_temp_path = "temp_image.jpg"

        # Downloading the image and appending new info
        try:
            download_image_from_gcs(BUCKET_NAME, gcs_image_path, local_temp_path)
            preprocessed_image = preprocess_image(local_temp_path, target_size)
            output_path = os.path.join(PROCESSED_DATA_DIR, f"{dish_id}.npy")
            np.save(output_path, preprocessed_image)

            preprocessed_data.append({
                "image_id": dish_id,
                "processed_image_path": output_path,
                "portion_size": row.iloc[2]
            })

        finally:
            if os.path.exists(local_temp_path):
                os.remove(local_temp_path)

    pd.DataFrame(preprocessed_data).to_csv(os.path.join(PROCESSED_DATA_DIR, "processed_annotations.csv"), index=False)

if __name__ == "__main__":
    preprocess_dataset(os.path.join(RAW_DATA_DIR, "Nutrition5kModified.csv"))