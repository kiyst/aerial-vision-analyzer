from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from aerial_vision.detection import ObjectDetection, normalize_label


@dataclass(frozen=True)
class ImageAnalysis:
    image_path: str
    image_width: int
    image_height: int
    detections: list[ObjectDetection]

    @property
    def total_detections(self) -> int:
        return len(self.detections)

    @property
    def counts_by_label(self) -> dict[str, int]:
        counts = Counter(normalize_label(detection.label) for detection in self.detections)
        return dict(sorted(counts.items()))

    @property
    def average_confidence(self) -> float | None:
        if not self.detections:
            return None
        return sum(detection.confidence for detection in self.detections) / len(self.detections)

    @property
    def largest_detection(self) -> ObjectDetection | None:
        if not self.detections:
            return None
        return max(self.detections, key=lambda detection: detection.box.area)

    def to_dict(self) -> dict[str, object]:
        largest = self.largest_detection
        return {
            "image": {
                "path": self.image_path,
                "width_px": self.image_width,
                "height_px": self.image_height,
            },
            "summary": {
                "total_detections": self.total_detections,
                "counts_by_label": self.counts_by_label,
                "average_confidence": self.average_confidence,
                "largest_detection": largest.to_dict() if largest else None,
            },
            "detections": [detection.to_dict() for detection in self.detections],
        }

