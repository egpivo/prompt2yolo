import unittest
from unittest.mock import patch

from prompt2yolo.evaluation.utils import plot_fp_distribution, plot_prompt_weights


class TestPlotFPDistribution(unittest.TestCase):
    def setUp(self):
        self.fp_counts = {
            "Prompt_1": 10,
            "Prompt_2": 20,
            "Prompt_3": 5,
        }
        self.save_path = "test_fp_distribution.png"

    @patch("matplotlib.pyplot.show", return_value=None)
    def test_plot_fp_distribution(self, mock_show):
        """Test the plot generation without saving."""
        plot_fp_distribution(self.fp_counts)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show", return_value=None)
    @patch("matplotlib.pyplot.savefig", return_value=None)
    def test_plot_fp_distribution_save(self, mock_savefig, mock_show):
        """Test the plot generation and saving."""
        plot_fp_distribution(self.fp_counts, save_path=self.save_path)
        mock_savefig.assert_called_once_with(
            self.save_path, dpi=300, bbox_inches="tight"
        )
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show", return_value=None)
    def test_truncate_prompt(self, mock_show):
        """Test prompt truncation logic."""
        truncated_fp_counts = {
            prompt[:7] + "..." if len(prompt) > 7 else prompt: count
            for prompt, count in self.fp_counts.items()
        }
        plot_fp_distribution(truncated_fp_counts, truncate_length=7)
        mock_show.assert_called_once()


class TestPlotPromptWeights(unittest.TestCase):
    def setUp(self):
        self.weights = {
            "Prompt1_with_a_long_description": 1.5,
            "Prompt2_even_longer_description_than_before": 2.0,
            "Prompt3_short": 3.5,
        }
        self.save_path = "test_plot.png"

    @patch("matplotlib.pyplot.show", return_value=None)
    def test_plot_without_save(self, mock_show):
        """Test that the plot is generated and displayed without saving."""
        plot_prompt_weights(self.weights, truncate_length=20)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show", return_value=None)
    @patch("matplotlib.pyplot.savefig", return_value=None)
    def test_plot_with_save(self, mock_savefig, mock_show):
        """Test that the plot is generated and saved correctly."""
        plot_prompt_weights(self.weights, truncate_length=20, save_path=self.save_path)
        mock_savefig.assert_called_once_with(
            self.save_path, dpi=300, bbox_inches="tight"
        )
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
