import unittest
from unittest.mock import patch

from prompt2yolo.evaluation.label_categorizer import LabelCategorizer, calculate_iou


# Unit test class
class TestCalculateIoU(unittest.TestCase):
    def test_no_overlap(self):
        self.assertAlmostEqual(calculate_iou((0, 0, 2, 2), (3, 3, 5, 5)), 0.0)

    def test_partial_overlap(self):
        self.assertAlmostEqual(calculate_iou((0, 0, 4, 4), (2, 2, 6, 6)), 1 / 7)

    def test_full_overlap(self):
        self.assertAlmostEqual(calculate_iou((1, 1, 3, 3), (1, 1, 3, 3)), 1.0)

    def test_edge_touching(self):
        self.assertAlmostEqual(calculate_iou((0, 0, 2, 2), (2, 2, 4, 4)), 0.0)

    def test_one_inside_another(self):
        self.assertAlmostEqual(calculate_iou((0, 0, 4, 4), (1, 1, 3, 3)), 4 / 16)


class TestLabelCategorizer(unittest.TestCase):
    def setUp(self):
        """Set up the LabelCategorizer instance and sample data for testing."""
        self.categorizer = LabelCategorizer(iou_threshold=0.4)
        self.ground_truth_boxes = [
            (0, 10, 20, 50, 60),  # Class 0
            (1, 30, 40, 70, 80),  # Class 1
        ]
        self.model_detect_boxes = [
            (0, 12, 22, 48, 58),  # Class 0, close match to ground_truth_boxes[0]
            (1, 100, 100, 120, 140),  # Class 1, no match with any ground truth
        ]

    @patch("prompt2yolo.evaluation.label_categorizer.calculate_iou")
    def test_categorize(self, mock_calculate_iou):
        # Mock IoU values to control the results of `calculate_iou`
        mock_calculate_iou.side_effect = lambda box1, box2: (
            0.5
            if box1 == (12, 22, 48, 58) and box2 == (10, 20, 50, 60)
            else 0.1  # For all other comparisons
        )

        true_positives, false_positives, false_negatives = self.categorizer.categorize(
            self.ground_truth_boxes, self.model_detect_boxes
        )

        # Expected results
        expected_true_positives = [
            (0, 12, 22, 48, 58)
        ]  # One match for ground_truth_boxes[0]
        expected_false_positives = [
            (1, 100, 100, 120, 140)
        ]  # No matches for model_detect_boxes[1]
        expected_false_negatives = [
            (1, 30, 40, 70, 80)
        ]  # No matches for ground_truth_boxes[1]

        # Assertions
        self.assertEqual(
            true_positives, expected_true_positives, "True positives are incorrect."
        )
        self.assertEqual(
            false_positives, expected_false_positives, "False positives are incorrect."
        )
        self.assertEqual(
            false_negatives, expected_false_negatives, "False negatives are incorrect."
        )
        mock_calculate_iou.assert_called()  # Ensure IoU calculation was called

    def test_empty_inputs(self):
        """Test categorization with empty inputs."""
        true_positives, false_positives, false_negatives = self.categorizer.categorize(
            [], []
        )
        self.assertEqual(
            true_positives, [], "True positives should be empty for empty input."
        )
        self.assertEqual(
            false_positives, [], "False positives should be empty for empty input."
        )
        self.assertEqual(
            false_negatives, [], "False negatives should be empty for empty input."
        )

    def test_no_matches(self):
        """Test categorization with no matches between ground truth and detection."""
        ground_truth_boxes = [(0, 0, 0, 10, 10)]  # Small box
        model_detect_boxes = [(0, 100, 100, 110, 110)]  # Far away box
        true_positives, false_positives, false_negatives = self.categorizer.categorize(
            ground_truth_boxes, model_detect_boxes
        )

        self.assertEqual(
            true_positives, [], "True positives should be empty when no matches exist."
        )
        self.assertEqual(
            false_positives,
            model_detect_boxes,
            "False positives should include unmatched detections.",
        )
        self.assertEqual(
            false_negatives,
            ground_truth_boxes,
            "False negatives should include unmatched ground truths.",
        )


if __name__ == "__main__":
    unittest.main()
