import argparse
import os

from dotenv import load_dotenv

from prompt2yolo.configs import ImageGeneratorConfig, Paths, YoloLabelerConfig
from prompt2yolo.data.data_generation.data_splitter import DataSplitter
from prompt2yolo.data.data_generation.image_generators import GeneratorFactory
from prompt2yolo.data.data_generation.image_labeler import YoloWorldLabeler
from prompt2yolo.data.data_generation.utils import normalize_prompt_weights
from prompt2yolo.data.utils import clean_directory
from prompt2yolo.utils.logger import setup_logger
from prompt2yolo.utils.s3_handler import S3Handler
from prompt2yolo.utils.utils import load_yaml_config

load_dotenv()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Data generation and labeling script")
    parser.add_argument(
        "--input_yaml",
        type=str,
        required=True,
        help="Path to the input YAML file containing prompts and classes",
    )
    parser.add_argument(
        "--image_generator_yaml",
        type=str,
        required=True,
        help="Path to the Image Generator config YAML configuration file",
    )
    parser.add_argument(
        "--image_labeler_yaml",
        type=str,
        required=True,
        help="Path to the Image Labeler YAML configuration file",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./local_temp",
        help="Local directory for storing downloaded files",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Iteration number for the pipeline. Defaults to 1.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(__name__)

    # Load configurations
    input_config = load_yaml_config(args.input_yaml)
    image_generator_config = ImageGeneratorConfig(
        **load_yaml_config(args.image_generator_yaml)
    )
    image_labeler_config = YoloLabelerConfig(
        **load_yaml_config(args.image_labeler_yaml)
    )

    # Adjust test_ratio based on iteration
    if args.iteration > 1:
        logger.info(f"Iteration {args.iteration} detected. Setting test_ratio to 0.")
        image_generator_config.test_ratio = 0

    # Initialize S3Handler
    lora_s3_path = f"model_checkpoints/{image_generator_config.generator}"
    s3_handler = S3Handler(
        s3_folder=lora_s3_path, local_dir=args.local_dir, logger=logger
    )
    os.makedirs(s3_handler.local_dir, exist_ok=True)
    logger.info(f"Using local directory: {s3_handler.local_dir}")
    logger.info(
        f"Files in S3 folder '{s3_handler.s3_folder}': {s3_handler.list_files_in_folder()}"
    )

    # Initialize generator and generate images
    prompts_data = input_config.get("prompts", [])

    paths = Paths(
        local_data_path=os.getenv("LOCAL_DATA_PATH"), iteration=args.iteration
    )

    clean_directory(paths.image_folder, logger)
    clean_directory(paths.label_folder, logger)
    generator = GeneratorFactory.get_generator(
        s3_handler=s3_handler, config=image_generator_config
    )

    for prompt in normalize_prompt_weights(prompts_data):
        generator.generate(
            prompt=prompt.text, weight=prompt.weight, output_dir=paths.image_folder
        )
        logger.info(
            f"Generated images for prompt: '{prompt.text}' with weight: {prompt.weight}"
        )

    # Label images
    image_labeler = YoloWorldLabeler(
        s3_handler=s3_handler,
        custom_classes=input_config.get("classes"),
        config=image_labeler_config,
        logger=logger,
    )
    image_labeler.label(
        local_image_path=paths.image_folder, output_dir=paths.label_folder
    )
    logger.info("Image labeling completed.")

    # Split data
    DataSplitter(
        paths=paths,
        val_ratio=image_generator_config.val_ratio,
        test_ratio=image_generator_config.test_ratio,
        logger=logger,
    ).split()


if __name__ == "__main__":
    main()
