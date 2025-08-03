import os
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Paths:
    mode: str = "train"
    local_data_path: str = os.environ["LOCAL_DATA_PATH"]
    project: str = os.environ["PROJECT"]
    iteration: int = 1

    def __post_init__(self):
        # Convert iteration to a string for consistent path naming
        iteration_folder = f"iteration_{self.iteration}"

        # Local paths
        self.image_folder = os.path.join(self.local_data_path, "images")
        self.label_folder = os.path.join(self.local_data_path, "labels")
        self.yolo_config_folder = os.path.join(self.local_data_path, "yolo", "configs")
        self.yolo_data_init_folder = os.path.join(
            self.local_data_path, "yolo", "data", "iteration_1"
        )
        self.yolo_data_folder = os.path.join(
            self.local_data_path, "yolo", "data", iteration_folder
        )
        self.yolo_model_folder = os.path.join(
            self.local_data_path, "yolo", "models", iteration_folder
        )

        # S3 paths
        self.s3_image_folder = (
            f"projects/{self.project}/data/{iteration_folder}/{self.mode}/images"
        )
        self.s3_label_folder = (
            f"projects/{self.project}/data/{iteration_folder}/{self.mode}/labels"
        )
        self.s3_model_folder = f"projects/{self.project}/models/{iteration_folder}"
        self.s3_model_folder = f"projects/{self.project}/models/{iteration_folder}"
        self.s3_model_weights_folder = f"{self.s3_model_folder}/weights"
        self.s3_model_training_performance_folder = (
            f"{self.s3_model_folder}/evaluation/model/training/performance"
        )
        self.s3_model_inference_performance_folder = (
            f"{self.s3_model_folder}/evaluation/model/inference/performance"
        )
        self.s3_model_training_prediction_folder = (
            f"{self.s3_model_folder}/evaluation/model/training/prediction"
        )
        self.s3_model_inference_prediction_folder = (
            f"{self.s3_model_folder}/evaluation/model/inference/prediction"
        )
        self.s3_label_detection_folder = (
            f"{self.s3_model_folder}/evaluation/label/detection"
        )
        self.s3_label_detection_category_folder = (
            f"{self.s3_model_folder}/evaluation/label/category"
        )
        self.s3_label_detection_statistics_folder = (
            f"{self.s3_model_folder}/evaluation/label/statistics"
        )
        self.ground_truth_labels_folder = os.path.join(
            self.yolo_data_folder, "test/labels"
        )
        self.model_detect_labels_folder = os.path.join(
            self.yolo_data_folder, "test/predicted_labels"
        )
        self.separation_result_folder = os.path.join(
            self.yolo_data_folder, "test/separation_results"
        )


@dataclass
class Prompt:
    text: str
    weight: float = 1.0


@dataclass
class ImageGeneratorConfig:
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_path: str = "madebyollin/sdxl-vae-fp16-fix"
    image_size: Tuple[int, int] = (1024, 1024)
    guidance_scale: int = 7
    steps: int = 40
    num_images: int = 5
    negative_prompt: str = (
        "anime, cartoon, graphic, text, painting, crayon, graphite, abstract"
    )
    generator: Optional[str] = None
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    prompts: List[Prompt] = field(default_factory=list)  # List of Prompt objects


@dataclass
class EvaluationConfig:
    images_folder: str
    ground_truth_labels_folder: str
    model_detect_labels_folder: str
    result_path: str
    iou_threshold: float = 0.4  # Default value


@dataclass
class YoloLabelerConfig:
    yolo_model: str = "yolov8l-world.pt"
    conf: float = 0.3
    iou: float = 0.5
    max_det: int = 15
    image_size: int = 640
    augment: bool = True
    agnostic_nms: bool = True


@dataclass
class YoloV5Hyperparameters:
    lr0: float = 0.01
    lrf: float = 0.2
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box: float = 0.05
    cls: float = 0.3
    cls_pw: float = 1.0
    obj: float = 0.7
    obj_pw: float = 1.0
    iou_t: float = 0.30
    anchor_t: float = 4.0
    fl_gamma: float = 0.0
    hsv_h: float = 0.0
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: int = 1
    mixup: float = 0.0
    copy_paste: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class YoloV5DataConfig:
    train_dir: str
    val_dir: str
    class_names: list

    def to_dict(self):
        return {
            "train": self.train_dir,
            "val": self.val_dir,
            "nc": len(self.class_names),
            "names": self.class_names,
        }


@dataclass
class YoloV3TinyHyperparameters:
    batch_size: int = 64
    max_batches: int = 5000
    learning_rate: float = 0.001
    steps: list = (4000, 4500)
    momentum: float = 0.9
    weight_decay: float = 0.0005
    filters_base: int = 5  # Don't change it

    def to_dict(self):
        return asdict(self)


@dataclass
class YoloV3TinyDataConfig:
    train_dir: str
    class_name_file: str
    num_of_classes: int
    save_model_path: dir

    def save_to_obj_data(self, file_path: str):
        """Save the data configuration to an 'obj.data' file in the required format."""
        with open(file_path, "w") as file:
            file.write(f"classes = {self.num_of_classes}\n")
            file.write(f"train = {self.train_dir}\n")
            file.write(f"names = {self.class_name_file}\n")
            file.write(f"backup = {self.save_model_path}\n")


class YoloModelArchitecture(Enum):
    YOLOV3_TINY = auto()
    # Add more architectures as needed

    @property
    def layer_indexes(self):
        mapping = {
            YoloModelArchitecture.YOLOV3_TINY: [126, 170],
        }
        return mapping.get(self, [])


class S3Config:
    BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
    if not BUCKET_NAME:
        raise ValueError("AWS_S3_BUCKET_NAME is not set in the environment.")
