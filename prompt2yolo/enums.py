from enum import Enum


class Category(Enum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    NO_DETECTIONS = "no_detections"
