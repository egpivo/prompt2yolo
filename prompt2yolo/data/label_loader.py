from typing import List, Tuple


class LabelLoader:
    """Handles the loading and transformation of label data."""

    @staticmethod
    def load_labels(
        label_path: str, width: int, height: int
    ) -> List[Tuple[int, int, int, int, int]]:
        """Loads bounding boxes from a label file and converts normalized coordinates to pixel coordinates."""
        boxes = []
        with open(label_path, "r") as label_file:
            for line in label_file:
                class_id, x, y, w, h = map(float, line.strip().split())
                class_id = int(class_id)
                x1 = int((x - w / 2) * width)
                y1 = int((y - h / 2) * height)
                x2 = int((x + w / 2) * width)
                y2 = int((y + h / 2) * height)
                boxes.append((class_id, x1, y1, x2, y2))
        return boxes
