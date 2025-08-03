from pathlib import Path

from prompt2yolo.configs import S3Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUCKET_NAME = S3Config.BUCKET_NAME
