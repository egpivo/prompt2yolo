from collections import defaultdict
from typing import Dict, List


class PromptWeightCalculator:
    def __init__(self):
        self.fp_counts = defaultdict(int)  # Tracks FP count per prompt
        self.total_counts = defaultdict(int)  # Tracks total predictions per prompt

    def extract_prompt(self, filename: str) -> str:
        """
        Extracts the prompt from a filename. The filename should follow the pattern:
        `<prefix>_<prompt>_<unique_id>.jpg`. Handles cases where the prefix is optional.
        """
        if not filename.endswith(".jpg"):
            raise ValueError(f"Invalid filename format: {filename}")

        # Remove file extension
        base_name = filename.rsplit(".", 1)[0]

        # Split by '_'
        components = base_name.split("_")
        if len(components) < 2:
            raise ValueError(f"Invalid filename format: {filename}")

        # Exclude the last part (unique ID)
        return "_".join(components[:-1])

    def update_counts(
        self, false_positive_boxes: List, total_boxes: List, filename: str
    ) -> None:
        """Updates the FP and total counts for the given prompt."""
        prompt = self.extract_prompt(filename)
        self.fp_counts[prompt] += len(false_positive_boxes)
        self.total_counts[prompt] += len(total_boxes)

    def calculate_fp_rate(self) -> Dict[str, float]:
        """Calculates the false positive rate for each prompt."""
        fp_rates = {}
        for prompt, fp_count in self.fp_counts.items():
            total_count = self.total_counts[prompt]
            fp_rates[prompt] = fp_count / total_count if total_count > 0 else 0.0
        return fp_rates

    def calculate_weights(self) -> Dict[str, float]:
        """Calculates weights as the inverse of the FP rate and normalizes them to sum to 1."""
        fp_rates = self.calculate_fp_rate()
        raw_weights = {
            prompt: 1 / (fp_rate or 1e-6) for prompt, fp_rate in fp_rates.items()
        }
        total_weight = sum(raw_weights.values())
        return {prompt: weight / total_weight for prompt, weight in raw_weights.items()}
