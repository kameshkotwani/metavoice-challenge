from pathlib import Path
import boto3
import os
from config import RAW_DATA_DIR
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


def download_from_s3(bucket_name:str,  local_dir:Path, s3_key:str=""):
    
    """
    Download a file from S3 and save it to the local directory.

    :param bucket_name: Name of the S3 bucket
    :param s3_key: Key of the file in the S3 bucket
    :param local_dir: Local directory to save the file
    """


    # Extract the filename from the S3 key
    # local_file_path = os.path.join(local_dir, os.path.basename(s3_key))

    # Initialize S3 client
    s3 = boto3.client(
        's3',
        endpoint_url = os.getenv('S3_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('ACCESS_ID_KEY'),
        aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
    )

    response:dict = s3.list_objects_v2(Bucket=bucket_name, Prefix="data/" )
    print(response.get('Contents', []))
    # # Download the file
    # print(f"Downloading {s3_key} from bucket {bucket_name} to {local_file_path}...")
    # s3.download_file(bucket_name, s3_key, local_file_path)
    # print("Download complete.")

if __name__ == "__main__":
    # Replace these with your S3 bucket details and local directory
    BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
    S3_KEY = "path/to/your/file/in/s3"
    LOCAL_DIR = RAW_DATA_DIR

    download_from_s3(BUCKET_NAME, LOCAL_DIR)