import unittest

from prompt2yolo.evaluation.prompt_weight_calculator import PromptWeightCalculator


class TestPromptWeightCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = PromptWeightCalculator()

    def test_extract_prompt(self):
        filename = "Office_space__afternoon__person__sitting_at_a_desk__working_on_a_computer___5908722711209.jpg"
        expected_prompt = "Office_space__afternoon__person__sitting_at_a_desk__working_on_a_computer__"
        self.assertEqual(self.calculator.extract_prompt(filename), expected_prompt)

    def test_update_counts(self):
        filename = "Office_space__afternoon__person__sitting_at_a_desk__working_on_a_computer___5908722711209.jpg"
        false_positive_boxes = ["fp1", "fp2"]
        total_boxes = ["fp1", "fp2", "tp1"]
        self.calculator.update_counts(false_positive_boxes, total_boxes, filename)

        self.assertEqual(
            self.calculator.fp_counts[
                "Office_space__afternoon__person__sitting_at_a_desk__working_on_a_computer__"
            ],
            2,
        )
        self.assertEqual(
            self.calculator.total_counts[
                "Office_space__afternoon__person__sitting_at_a_desk__working_on_a_computer__"
            ],
            3,
        )

    def test_calculate_weights(self):
        filenames = [
            "Office_space__afternoon__person__sitting_at_a_desk__working_on_a_computer___5908722711209.jpg",
            "Library__quiet_atmosphere__person__sitting__reading_a_book__3039131310460.jpg",
        ]
        # Update counts for two prompts
        self.calculator.update_counts(
            ["fp1"], ["fp1", "tp1", "tp2"], filenames[0]
        )  # FP: 1, Total: 3
        self.calculator.update_counts(
            ["fp1", "fp2"], ["fp1", "fp2", "tp1"], filenames[1]
        )  # FP: 2, Total: 3

        # Extract prompts
        prompts = [
            self.calculator.extract_prompt(filenames[0]),
            self.calculator.extract_prompt(filenames[1]),
        ]

        # Calculate weights
        weights = self.calculator.calculate_weights()

        # Check if weights sum to 1
        self.assertAlmostEqual(sum(weights.values()), 1.0)

        # Expected normalized weights
        raw_weights = {
            prompts[0]: 1 / (1 / 3),  # FP rate: 1/3
            prompts[1]: 1 / (2 / 3),  # FP rate: 2/3
        }
        total_raw_weight = sum(raw_weights.values())
        expected_weights = {
            prompt: raw_weight / total_raw_weight
            for prompt, raw_weight in raw_weights.items()
        }

        # Verify normalized weights
        self.assertAlmostEqual(weights[prompts[0]], expected_weights[prompts[0]])
        self.assertAlmostEqual(weights[prompts[1]], expected_weights[prompts[1]])


if __name__ == "__main__":
    unittest.main()
