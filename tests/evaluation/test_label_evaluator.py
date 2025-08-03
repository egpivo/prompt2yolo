import unittest
from unittest.mock import patch

from prompt2yolo.evaluation.label_evaluator import LabelEvaluator
from prompt2yolo.evaluation.prompt_weight_calculator import PromptWeightCalculator


class TestLabelEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up the LabelEvaluator instance and mock dependencies."""
        self.images_folder = "test_images"
        self.ground_truth_labels_folder = "test_ground_truth"
        self.model_detect_labels_folder = "test_model_detect"
        self.result_path = "test_results"
        self.iou_threshold = 0.4

        self.prompt_weight_calculator = PromptWeightCalculator()
        self.evaluator = LabelEvaluator(
            self.images_folder,
            self.ground_truth_labels_folder,
            self.model_detect_labels_folder,
            self.result_path,
            self.prompt_weight_calculator,
            self.iou_threshold,
        )

    @patch("os.listdir", return_value=[])
    def test_process_all_images_no_images(self, mock_listdir):
        """Test process_all_images with no images in the folder."""
        self.evaluator.process_all_images()
        mock_listdir.assert_called_once_with(self.images_folder)

    @patch("prompt2yolo.evaluation.label_evaluator.cv2.imread", return_value=None)
    def test_process_single_image_missing_image(self, mock_imread):
        """Test process_single_image when the image file is missing or unreadable."""
        image_file = "missing_image.jpg"

        with self.assertRaises(ValueError):
            self.evaluator.process_single_image(image_file)

        mock_imread.assert_called_once_with("test_images/missing_image.jpg")

    def test_calculate_prompt_weights(self):
        """Test calculate_prompt_weights function."""
        with patch.object(
            self.prompt_weight_calculator,
            "calculate_weights",
            return_value={"prompt1": 1.0, "prompt2": 0.5},
        ) as mock_calculate_weights:
            weights = self.evaluator.calculate_prompt_weights()

            self.assertEqual(weights, {"prompt1": 1.0, "prompt2": 0.5})
            mock_calculate_weights.assert_called_once()


if __name__ == "__main__":
    unittest.main()
