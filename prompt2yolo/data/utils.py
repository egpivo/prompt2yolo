import os
import shutil
from typing import List, Tuple

from prompt2yolo.utils.logger import setup_logger

LOGGER = setup_logger(__name__)


def write_label_file(
    labels_path: str,
    image_file: str,
    boxes: List[Tuple[int, int, int, int, int]],
    width: int,
    height: int,
):
    os.makedirs(labels_path, exist_ok=True)
    label_file_path = os.path.join(labels_path, image_file.replace(".jpg", ".txt"))

    if not boxes:
        LOGGER.warning(
            f"No boxes to write for {image_file}. Skipping label file creation."
        )
        return

    try:
        with open(label_file_path, "w") as label_file:
            for class_id, x1, y1, x2, y2 in boxes:
                # Convert to YOLO format (normalized x, y, w, h)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                label_file.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
                )
        LOGGER.info(f"Label file saved: {label_file_path}")
    except Exception as e:
        LOGGER.error(f"Failed to write label file for {image_file}: {e}")


def save_visualized_image(image_file: str, images_path: str) -> None:
    os.makedirs(images_path, exist_ok=True)
    output_image_path = os.path.join(images_path, os.path.basename(image_file))
    try:
        shutil.copy(image_file, output_image_path)
        LOGGER.info(f"Image file saved: {output_image_path}")
    except FileNotFoundError as e:
        LOGGER.error(
            f"Failed to copy image file: {image_file} to {images_path}. Reason: {e}"
        )


def clean_directory(dir_path: str, logger):
    """Clean the specified directory by removing all its contents."""
    if os.path.exists(dir_path):
        logger.info(f"Cleaning directory: {dir_path}")
        shutil.rmtree(dir_path)  # Remove all contents
        os.makedirs(dir_path)  # Recreate the directory
    else:
        logger.info(f"Directory does not exist, creating: {dir_path}")
        os.makedirs(dir_path)
