from __future__ import annotations

from pathlib import Path
from typing import Any

from aerial_vision.detection import BoundingBox, ObjectDetection, filter_detections


RFDETR_MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


class RFDetrObjectDetector:
    """RF-DETR adapter.

    RF-DETR is a transformer-based detector. The default public checkpoints are
    COCO-trained, but this adapter can also load custom fine-tuned checkpoints.
    """

    def __init__(self, model_size: str = "medium", checkpoint_path: str | Path | None = None) -> None:
        try:
            import rfdetr
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR is not installed. Install it with: python3 -m pip install '.[rfdetr]'"
            ) from exc

        model_class_name = RFDETR_MODEL_CLASSES.get(model_size)
        if model_class_name is None:
            valid_sizes = ", ".join(sorted(RFDETR_MODEL_CLASSES))
            raise ValueError(f"Unsupported RF-DETR model size '{model_size}'. Use one of: {valid_sizes}.")

        model_class = getattr(rfdetr, model_class_name)
        self.model_size = model_size
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.class_names = COCO_CLASSES

        if self.checkpoint_path:
            self.model = model_class(pretrain_weights=self.checkpoint_path)
            self.source_model = f"rfdetr-{model_size}:{self.checkpoint_path}"
        else:
            self.model = model_class()
            self.source_model = f"rfdetr-{model_size}"

    def detect(
        self,
        image_path: str | Path,
        *,
        min_confidence: float = 0.25,
        objects_of_interest: set[str] | None = None,
    ) -> list[ObjectDetection]:
        raw_detections = self.model.predict(str(image_path), threshold=min_confidence)
        detections = self._convert_detections(raw_detections)

        if objects_of_interest is None:
            return filter_detections(detections, min_confidence=min_confidence)

        return filter_detections(
            detections,
            objects_of_interest=objects_of_interest,
            min_confidence=min_confidence,
        )

    def _convert_detections(self, raw_detections: Any) -> list[ObjectDetection]:
        detections: list[ObjectDetection] = []

        for xyxy, confidence, class_id in zip(
            raw_detections.xyxy,
            raw_detections.confidence,
            raw_detections.class_id,
        ):
            x_min, y_min, x_max, y_max = [float(value) for value in xyxy]
            class_index = int(class_id)
            label = self.class_names[class_index] if class_index < len(self.class_names) else str(class_index)
            detections.append(
                ObjectDetection(
                    label=label,
                    confidence=float(confidence),
                    box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    source_model=self.source_model,
                )
            )

        return detections
