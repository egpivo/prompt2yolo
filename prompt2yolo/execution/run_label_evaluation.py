import argparse
import csv
import os
from dataclasses import asdict

from dotenv import load_dotenv

from prompt2yolo.configs import EvaluationConfig, Paths
from prompt2yolo.evaluation.label_evaluator import LabelEvaluator
from prompt2yolo.evaluation.prompt_weight_calculator import PromptWeightCalculator
from prompt2yolo.evaluation.utils import (
    overwrite_input_yaml,
    plot_fp_distribution,
    plot_prompt_weights,
)
from prompt2yolo.utils.logger import setup_logger

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate labels and categorization")
    parser.add_argument(
        "--input_yaml", type=str, required=True, help="Path to input YAML"
    )
    parser.add_argument(
        "--iteration", type=int, default=1, help="Pipeline iteration number"
    )
    return parser.parse_args()


def create_evaluation_config(paths: Paths) -> EvaluationConfig:
    """Creates an EvaluationConfig object using Paths."""
    return EvaluationConfig(
        images_folder=os.path.join(paths.yolo_data_folder, "test/images"),
        ground_truth_labels_folder=paths.ground_truth_labels_folder,
        model_detect_labels_folder=paths.model_detect_labels_folder,
        result_path=paths.separation_result_folder,
        iou_threshold=0.4,  # Default IoU threshold
    )


def main():
    args = parse_args()
    logger = setup_logger(__name__)
    local_dir = os.getenv("LOCAL_DATA_PATH", "default_local_temp")
    paths = Paths(local_data_path=local_dir, mode="test", iteration=args.iteration)

    eval_config = create_evaluation_config(paths)
    logger.info(f"Evaluation config: {asdict(eval_config)}")

    evaluator = LabelEvaluator(
        images_folder=eval_config.images_folder,
        ground_truth_labels_folder=paths.ground_truth_labels_folder,
        model_detect_labels_folder=paths.model_detect_labels_folder,
        result_path=paths.separation_result_folder,
        prompt_weight_calculator=PromptWeightCalculator(),
        iou_threshold=eval_config.iou_threshold,
        logger=logger,
    )

    evaluator.process_all_images()
    prompt_weights = evaluator.calculate_prompt_weights()
    fp_rates = evaluator.prompt_weight_calculator.calculate_fp_rate()

    # Save max FP rate to file
    output_file = os.path.join(paths.separation_result_folder, "max_fp_rate.txt")
    os.makedirs(paths.separation_result_folder, exist_ok=True)
    max_fp_rate = max(fp_rates.values()) if fp_rates else 0.0
    with open(output_file, "w") as file:
        file.write(f"{max_fp_rate}\n")
    logger.info(f"Max FP rate saved to {output_file}")

    # Save FP rates per prompt to CSV
    csv_file = os.path.join(paths.separation_result_folder, "fp_rates.csv")
    # If iteration == 1, we create a new file with header, else we append
    mode = "w" if args.iteration == 1 else "a"
    write_header = args.iteration == 1

    with open(csv_file, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["iteration", "prompt", "fp_rate"])
        for prompt, fp_rate in fp_rates.items():
            writer.writerow([args.iteration, prompt, fp_rate])
    logger.info(f"FP rates appended to {csv_file}")

    plot_prompt_weights(
        prompt_weights,
        save_path=os.path.join(
            paths.separation_result_folder, "label_prompt_weight.jpg"
        ),
    )
    plot_fp_distribution(
        fp_rates,
        save_path=os.path.join(
            paths.separation_result_folder, "label_prompt_fp_rate_distribution.jpg"
        ),
    )
    logger.info("Label evaluation completed.")

    overwrite_input_yaml(args.input_yaml, prompt_weights)
    logger.info(f"Updated {args.input_yaml} with new weights.")


if __name__ == "__main__":
    main()
