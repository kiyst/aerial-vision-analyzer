from __future__ import annotations

from pathlib import Path
from typing import Any

from aerial_vision.detection import BoundingBox, ObjectDetection, filter_detections


class YoloObjectDetector:
    """Ultralytics YOLO adapter.

    This class is intentionally isolated so the rest of the project can be
    tested without installing a large ML stack.
    """

    def __init__(self, model_path: str | Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install the optional ML dependencies "
                "with: python3 -m pip install '.[ml]'"
            ) from exc

        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)

    def detect(
        self,
        image_path: str | Path,
        *,
        min_confidence: float = 0.25,
        objects_of_interest: set[str] | None = None,
    ) -> list[ObjectDetection]:
        results = self.model.predict(str(image_path), conf=min_confidence, verbose=False)
        detections: list[ObjectDetection] = []

        for result in results:
            detections.extend(self._detections_from_result(result))

        if objects_of_interest is None:
            return filter_detections(detections, min_confidence=min_confidence)

        return filter_detections(
            detections,
            objects_of_interest=objects_of_interest,
            min_confidence=min_confidence,
        )

    def _detections_from_result(self, result: Any) -> list[ObjectDetection]:
        if getattr(result, "obb", None) is not None:
            obb_detections = self._detections_from_obb(result)
            if obb_detections:
                return obb_detections

        return self._detections_from_boxes(result)

    def _detections_from_boxes(self, result: Any) -> list[ObjectDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: list[ObjectDetection] = []
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x_min, y_min, x_max, y_max = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                ObjectDetection(
                    label=result.names[class_id],
                    confidence=confidence,
                    box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    source_model=self.model_path,
                )
            )

        return detections

    def _detections_from_obb(self, result: Any) -> list[ObjectDetection]:
        obb = result.obb
        detections: list[ObjectDetection] = []

        for index in range(len(obb.cls)):
            class_id = int(obb.cls[index].item())
            confidence = float(obb.conf[index].item())
            x_min, y_min, x_max, y_max = [float(value) for value in obb.xyxy[index].tolist()]
            polygon = tuple(
                (float(point[0]), float(point[1]))
                for point in obb.xyxyxyxy[index].tolist()
            )
            detections.append(
                ObjectDetection(
                    label=result.names[class_id],
                    confidence=confidence,
                    box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    source_model=self.model_path,
                    polygon=polygon,
                )
            )

        return detections
