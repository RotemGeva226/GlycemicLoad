from google.cloud import storage
import os


def list_and_download_rgb_files(bucket_name, base_prefix, download=False, download_dir=None, limit: int =1000):
    """Lists and optionally downloads rgb.png files from the specified bucket and prefix, including dish ID in the filename."""
    downloaded_files_path = r'C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final Project\Nutrition5k dataset\Images\Rotem'
    downloaded_files = os.listdir(downloaded_files_path)
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=base_prefix)
    counter = 0

    rgb_files = []

    for blob in blobs:
        if counter != limit:
            if blob.name.endswith('/rgb.png'):
                rgb_files.append(blob.name)
                print(f"Found: {blob.name}")

                if download and download_dir:
                    # Extract dish ID from the path
                    path_parts = blob.name.split('/')
                    dish_id = path_parts[-2]  # Assumes dish ID is the parent directory of rgb.png

                    # Create the directory structure
                    #local_path = os.path.join(download_dir, os.path.dirname(blob.name.replace(base_prefix, '')))
                    #os.makedirs(local_path, exist_ok=True)

                    # Download the file with dish ID in the filename
                    local_file_path = os.path.join(download_dir, f'{dish_id}_rgb.png')
                    if f'{dish_id}_rgb.png' not in downloaded_files:
                        blob.download_to_filename(local_file_path)
                        print(f"Downloaded: {local_file_path}")
                        counter = counter + 1

    return rgb_files


# Set your bucket name
bucket_name = "nutrition5k_dataset"

# Set the base directory
base_prefix = "nutrition5k_dataset/imagery/realsense_overhead/"

# Set to True if you want to download the files
download_files = True

# Set the local directory for downloads (only used if download_files is True)
local_download_dir = r'C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final Project\Nutrition5k dataset\Images\Or'

current_dish_csv = r"C:\Users\rotem\OneDrive - Afeka College Of Engineering\Final Project\Nutrition5k dataset\Dishes for classification test.csv"

# List (and optionally download) the rgb.png files
rgb_files = list_and_download_rgb_files(bucket_name, base_prefix, download=download_files,
                                        download_dir=local_download_dir, limit=4000)

print(f"\nTotal rgb.png files found: {len(rgb_files)}")