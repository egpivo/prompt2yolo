from typing import List, Tuple


def calculate_iou(
    box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]
) -> float:
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)

    if intersection_x1 >= intersection_x2 or intersection_y1 >= intersection_y2:
        return 0.0

    intersection_area = (intersection_x2 - intersection_x1) * (
        intersection_y2 - intersection_y1
    )
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)
    return intersection_area / (area_box1 + area_box2 - intersection_area)


class LabelCategorizer:
    """Categorizes labels into true positives, false positives, and false negatives."""

    def __init__(self, iou_threshold: float = 0.4):
        self.iou_threshold = iou_threshold

    def categorize(
        self,
        ground_truth_boxes: List[Tuple[int, int, int, int, int]],
        model_detect_boxes: List[Tuple[int, int, int, int, int]],
    ) -> Tuple[List, List, List]:
        true_positive_boxes = []
        false_positive_boxes = []
        false_negative_boxes = []

        # Track matches to ensure no duplication
        matched_gt_indices = set()
        matched_det_indices = set()

        # Step 1: Match detection boxes to ground truth boxes (True Positives)
        for det_idx, det in enumerate(model_detect_boxes):
            cls, x1, y1, x2, y2 = det
            for gt_idx, gt in enumerate(ground_truth_boxes):
                if gt_idx in matched_gt_indices or det_idx in matched_det_indices:
                    continue

                iou = calculate_iou((x1, y1, x2, y2), gt[1:])
                if iou >= self.iou_threshold:
                    true_positive_boxes.append(det)
                    matched_gt_indices.add(gt_idx)
                    matched_det_indices.add(det_idx)
                    break

        # Step 2: Any unmatched detection boxes are False Positives
        for det_idx, det in enumerate(model_detect_boxes):
            if det_idx not in matched_det_indices:
                false_positive_boxes.append(det)

        # Step 3: Any unmatched ground truth boxes are False Negatives
        for gt_idx, gt in enumerate(ground_truth_boxes):
            if gt_idx not in matched_gt_indices:
                false_negative_boxes.append(gt)

        return true_positive_boxes, false_positive_boxes, false_negative_boxes
