import os
from io import BytesIO
from typing import List

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from prompt2yolo.utils.logger import setup_logger

load_dotenv()


class S3Handler:
    def __init__(self, s3_folder: str, local_dir: str, logger=None):
        self.s3_folder = s3_folder.rstrip("/")
        self.local_dir = local_dir
        self.logger = logger or setup_logger(__name__)

        # Load endpoint URL and bucket name from environment variables
        self.endpoint_url = os.getenv("AWS_S3_ENDPOINT")
        self.region_name = os.getenv("AWS_DEFAULT_REGION")
        self.bucket_name = os.getenv("AWS_S3_BUCKET_NAME")

        if not self.bucket_name or not self.endpoint_url:
            raise ValueError("Missing AWS_S3_BUCKET_NAME or AWS_S3_ENDPOINT in .env.")

        # Initialize boto3 resource and client with endpoint URL
        self.s3_resource = boto3.resource(
            "s3", endpoint_url=self.endpoint_url, region_name=self.region_name
        )
        self.s3_client = boto3.client(
            "s3", endpoint_url=self.endpoint_url, region_name=self.region_name
        )
        self.bucket = self.s3_resource.Bucket(self.bucket_name)

        # Check if the bucket exists
        if not self.check_bucket_exists():
            raise ValueError(f"Bucket '{self.bucket_name}' does not exist.")

    def list_files_in_folder(self) -> List[str]:
        """List files in the specified S3 folder."""
        try:
            # List objects in the specified folder
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=self.s3_folder
            )

            # Extract filenames
            files = [obj["Key"] for obj in response.get("Contents", [])]
            return files
        except ClientError as e:
            self.logger.error(
                f"Error listing files in S3 folder '{self.s3_folder}': {e}"
            )
            return []

    def check_bucket_exists(self) -> bool:
        """Check if the bucket exists."""
        try:
            self.s3_resource.meta.client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as e:
            self.logger.error(f"Bucket check failed: {e}")
            return False

    def download_file_to_stream(self, s3_uri: str) -> BytesIO:
        """Download a file from S3 and return it as a BytesIO object."""
        try:
            bucket, key = self._parse_s3_uri(s3_uri)
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return BytesIO(response["Body"].read())
        except ClientError as e:
            self.logger.error(f"Failed to download file to stream: {e}")
            raise

    def download_files_by_extension(self, file_extensions: List[str]) -> None:
        """Download files from the S3 folder based on specific file extensions."""
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            files_to_download = []

            for page in paginator.paginate(
                Bucket=self.bucket_name, Prefix=self.s3_folder
            ):
                files_to_download.extend(
                    obj["Key"]
                    for obj in page.get("Contents", [])
                    if any(obj["Key"].endswith(ext) for ext in file_extensions)
                )

            if not files_to_download:
                self.logger.warning(
                    f"No files with extensions {file_extensions} found in {self.s3_folder}."
                )
                return  # Exit gracefully if no files match

            for key in files_to_download:
                local_path = os.path.join(self.local_dir, os.path.basename(key))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                try:
                    self.s3_client.download_file(self.bucket_name, key, local_path)
                    self.logger.info(f"Downloaded {key} to {local_path}")
                except Exception as e:
                    self.logger.error(f"Failed to download {key}: {e}")
        except Exception as e:
            self.logger.error(f"Error during file download: {e}")
            raise

    def upload_file(self, local_path: str, s3_path: str) -> bool:
        """Upload a file from local storage to the specified S3 path."""
        try:
            self.logger.info(
                f"Preparing to upload file: {local_path} to S3 path: {s3_path}"
            )

            if not os.path.exists(local_path):
                self.logger.error(f"Local file '{local_path}' does not exist.")
                raise FileNotFoundError(f"Local file '{local_path}' not found.")

            if os.path.getsize(local_path) == 0:
                self.logger.error(f"Local file '{local_path}' is empty.")
                raise ValueError(
                    f"Local file '{local_path}' is empty and cannot be uploaded."
                )

            self.s3_client.upload_file(local_path, self.bucket_name, s3_path)
            self.logger.info(f"Successfully uploaded {local_path} to {s3_path}")

            # Verify upload success by checking if the file exists on S3
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_path)
            if response:
                self.logger.info(f"File verified on S3 at {s3_path}")
                return True
            else:
                self.logger.error(f"Upload verification failed for {s3_path}")
                return False

        except ClientError as e:
            self.logger.error(f"Failed to upload {local_path} to S3: {e}")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during upload: {e}")
            raise

    def download_all_files(self) -> None:
        """Download all files recursively from the S3 folder to the local directory."""
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            operation_parameters = {
                "Bucket": self.bucket_name,
                "Prefix": self.s3_folder,
            }
            page_iterator = paginator.paginate(**operation_parameters)

            for page in page_iterator:
                for obj in page.get("Contents", []):
                    s3_key = obj["Key"]
                    local_path = os.path.join(
                        self.local_dir, os.path.relpath(s3_key, self.s3_folder)
                    )

                    # Create local directories as needed
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    # Download the file
                    self.s3_client.download_file(self.bucket_name, s3_key, local_path)
                    self.logger.info(f"Downloaded {s3_key} to {local_path}")
        except Exception as e:
            self.logger.error(f"Failed to download files from S3: {e}")

    def _parse_s3_uri(self, s3_uri: str) -> (str, str):
        """Parse the S3 URI into bucket name and key."""
        s3_parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket_name = s3_parts[0]
        key = s3_parts[1] if len(s3_parts) > 1 else ""
        return bucket_name, key

    def _parse_s3_folder(self, s3_folder: str) -> (str, str):
        """Parse the S3 folder path into bucket name and prefix."""
        return self._parse_s3_uri(f"s3://{self.bucket_name}/{s3_folder}")
