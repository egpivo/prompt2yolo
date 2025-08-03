import os
from typing import List, Tuple

from prompt2yolo.data.utils import save_visualized_image, write_label_file


class FileHandler:
    """Handles saving label data and corresponding images."""

    @staticmethod
    def save_labels_and_images(
        category: str,
        image_path: str,
        image_file: str,
        boxes: List[Tuple[int, int, int, int, int]],
        result_path: str,
        width: int,
        height: int,
    ):
        """Saves labeled files and corresponding images."""
        if boxes:
            new_labels_path = os.path.join(result_path, f"{category}/labels")
            new_images_path = os.path.join(result_path, f"{category}/images")
            write_label_file(new_labels_path, image_file, boxes, width, height)
            save_visualized_image(image_path, new_images_path)
