import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from prompt2yolo.data.file_handler import FileHandler
from prompt2yolo.data.label_loader import LabelLoader
from prompt2yolo.enums import Category
from prompt2yolo.evaluation.label_categorizer import LabelCategorizer
from prompt2yolo.evaluation.prompt_weight_calculator import PromptWeightCalculator
from prompt2yolo.utils.logger import setup_logger

BoundingBox = Tuple[int, int, int, int, int]


class LabelEvaluator:
    def __init__(
        self,
        images_folder: str,
        ground_truth_labels_folder: str,
        model_detect_labels_folder: str,
        result_path: str,
        prompt_weight_calculator: PromptWeightCalculator,
        iou_threshold: float = 0.4,
        logger: Optional[logging.Logger] = None,
        file_handler: FileHandler = FileHandler(),
        label_loader: LabelLoader = LabelLoader(),
    ):
        self.images_folder = images_folder
        self.ground_truth_labels_folder = ground_truth_labels_folder
        self.model_detect_labels_folder = model_detect_labels_folder
        self.result_path = result_path
        self.label_categorizer = LabelCategorizer(iou_threshold)
        self.prompt_weight_calculator = prompt_weight_calculator
        self.logger = logger or setup_logger(__name__)
        self.file_handler = file_handler
        self.label_loader = label_loader

    def process_single_image(self, image_file: str) -> None:
        image = self._load_image(image_file)
        gt_boxes, det_boxes, width, height = self._load_boxes(image_file, image.shape)
        tps, fps, fns = self.label_categorizer.categorize(gt_boxes, det_boxes)
        self.prompt_weight_calculator.update_counts(fps, det_boxes, image_file)
        category, boxes = self._decide_category(tps, fps, fns)
        self._save_image_and_boxes(category, image_file, boxes, width, height)
        self.logger.info(f"Processed {image_file}")

    def _load_image(self, image_file: str) -> np.ndarray:
        image_path = os.path.join(self.images_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        return image

    def _load_boxes(
        self, image_file: str, shape: Tuple[int, int, int]
    ) -> Tuple[List[BoundingBox], List[BoundingBox], int, int]:
        height, width, _ = shape
        gt_label_path = os.path.join(
            self.ground_truth_labels_folder, image_file.replace(".jpg", ".txt")
        )
        det_label_path = os.path.join(
            self.model_detect_labels_folder, image_file.replace(".jpg", ".txt")
        )

        gt_boxes = (
            self.label_loader.load_labels(gt_label_path, width, height)
            if os.path.exists(gt_label_path)
            else []
        )
        if not gt_boxes:
            self.logger.info(f"No ground truth labels for {image_file}")

        det_boxes = (
            self.label_loader.load_labels(det_label_path, width, height)
            if os.path.exists(det_label_path)
            else []
        )
        if not det_boxes:
            self.logger.warning(f"No detection labels for {image_file}")

        return gt_boxes, det_boxes, width, height

    def _decide_category(
        self, tps: List[BoundingBox], fps: List[BoundingBox], fns: List[BoundingBox]
    ) -> Tuple[Category, List[BoundingBox]]:
        if fps:
            return Category.FALSE_POSITIVE, fps
        if fns:
            return Category.FALSE_NEGATIVE, fns
        if tps:
            return Category.TRUE_POSITIVE, tps
        return Category.NO_DETECTIONS, []

    def _save_image_and_boxes(
        self,
        category: Category,
        image_file: str,
        boxes: List[BoundingBox],
        width: int,
        height: int,
    ) -> None:
        image_path = os.path.join(self.images_folder, image_file)
        self.file_handler.save_labels_and_images(
            category.value,
            image_path,
            image_file,
            boxes,
            self.result_path,
            width,
            height,
        )

    def process_all_images(self) -> None:
        for image_file in os.listdir(self.images_folder):
            try:
                self.process_single_image(image_file)
            except FileNotFoundError:
                self.logger.warning(f"File not found: {image_file}")
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {e}")

    def calculate_prompt_weights(self) -> Dict[str, float]:
        return self.prompt_weight_calculator.calculate_weights()
