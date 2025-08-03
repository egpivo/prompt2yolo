import os
import unittest
from unittest.mock import mock_open, patch

from prompt2yolo.data.utils import save_visualized_image, write_label_file


class TestWriteLabelFile(unittest.TestCase):
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_write_label_file_with_boxes(self, mocked_open, mocked_makedirs):
        labels_path = "test_labels"
        image_file = "test_image.jpg"
        boxes = [(0, 10, 20, 30, 40)]
        width = 100
        height = 200

        expected_output = "0 0.200000 0.150000 0.200000 0.100000\n"

        write_label_file(labels_path, image_file, boxes, width, height)

        mocked_makedirs.assert_called_once_with(labels_path, exist_ok=True)
        mocked_open.assert_called_once_with(
            os.path.join(labels_path, "test_image.txt"), "w"
        )
        mocked_open().write.assert_called_once_with(expected_output)

    @patch("os.makedirs")
    def test_write_label_file_no_boxes(self, mocked_makedirs):
        labels_path = "test_labels"
        image_file = "test_image.jpg"
        boxes = []
        width = 100
        height = 200

        with patch("prompt2yolo.data.utils.LOGGER.warning") as mocked_logger:
            write_label_file(labels_path, image_file, boxes, width, height)

        mocked_makedirs.assert_called_once_with(labels_path, exist_ok=True)
        mocked_logger.assert_called_once_with(
            f"No boxes to write for {image_file}. Skipping label file creation."
        )


class TestSaveVisualizedImage(unittest.TestCase):
    @patch("os.makedirs")
    @patch("shutil.copy", side_effect=FileNotFoundError("File not found"))
    @patch("prompt2yolo.data.utils.LOGGER.error")
    def test_save_visualized_image_missing_file(
        self, mocked_logger, mocked_copy, mocked_makedirs
    ):
        image_file = "/path/to/missing_image.jpg"
        images_path = "test_images"

        save_visualized_image(image_file, images_path)

        mocked_makedirs.assert_called_once_with(images_path, exist_ok=True)
        mocked_copy.assert_called_once_with(
            image_file, os.path.join(images_path, "missing_image.jpg")
        )
        mocked_logger.assert_called_once_with(
            f"Failed to copy image file: {image_file} to {images_path}. Reason: File not found"
        )


if __name__ == "__main__":
    unittest.main()
