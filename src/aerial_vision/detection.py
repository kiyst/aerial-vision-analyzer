from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


DEFAULT_OBJECTS_OF_INTEREST = frozenset(
    {
        "person",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "dog",
        "cat",
        "horse",
        "cow",
        "sheep",
        "bird",
        "building",
        "airport",
        "baseball diamond",
        "basketball court",
        "bridge",
        "container crane",
        "ground track field",
        "harbor",
        "helicopter",
        "helipad",
        "large vehicle",
        "plane",
        "roundabout",
        "ship",
        "small vehicle",
        "soccer ball field",
        "storage tank",
        "swimming pool",
        "tennis court",
        "structure",
        "road",
        "path",
    }
)


ANIMAL_LABELS = frozenset({"bird", "cat", "cow", "dog", "horse", "sheep"})


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space bounding box with origin at the image's top-left corner."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max < self.x_min:
            raise ValueError("x_max must be greater than or equal to x_min.")
        if self.y_max < self.y_min:
            raise ValueError("y_max must be greater than or equal to y_min.")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.width / 2, self.y_min + self.height / 2)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    confidence: float
    box: BoundingBox
    source_model: str | None = None
    polygon: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Detection label cannot be empty.")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if self.polygon is not None and len(self.polygon) < 3:
            raise ValueError("Polygon detections must include at least three points.")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["box"] = asdict(self.box)
        payload["center"] = self.box.center
        payload["area_px"] = self.box.area
        return payload


@dataclass(frozen=True)
class GeometryRule:
    max_area_ratio: float | None = None
    max_width_ratio: float | None = None
    max_height_ratio: float | None = None
    min_area_px: float | None = None


OVERHEAD_VEHICLE_GEOMETRY_RULES = {
    "car": GeometryRule(max_area_ratio=0.01, max_width_ratio=0.10, max_height_ratio=0.10, min_area_px=40),
    "motor": GeometryRule(max_area_ratio=0.006, max_width_ratio=0.08, max_height_ratio=0.08, min_area_px=25),
    "motorcycle": GeometryRule(max_area_ratio=0.006, max_width_ratio=0.08, max_height_ratio=0.08, min_area_px=25),
    "bicycle": GeometryRule(max_area_ratio=0.006, max_width_ratio=0.08, max_height_ratio=0.08, min_area_px=25),
    "pedestrian": GeometryRule(max_area_ratio=0.004, max_width_ratio=0.06, max_height_ratio=0.06, min_area_px=15),
    "people": GeometryRule(max_area_ratio=0.004, max_width_ratio=0.06, max_height_ratio=0.06, min_area_px=15),
    "truck": GeometryRule(max_area_ratio=0.04, max_width_ratio=0.28, max_height_ratio=0.28, min_area_px=80),
    "van": GeometryRule(max_area_ratio=0.025, max_width_ratio=0.20, max_height_ratio=0.20, min_area_px=60),
    "bus": GeometryRule(max_area_ratio=0.04, max_width_ratio=0.28, max_height_ratio=0.28, min_area_px=80),
}


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def filter_detections(
    detections: Iterable[ObjectDetection],
    objects_of_interest: Iterable[str] = DEFAULT_OBJECTS_OF_INTEREST,
    *,
    min_confidence: float = 0.25,
) -> list[ObjectDetection]:
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1.")

    wanted = {normalize_label(label) for label in objects_of_interest}
    return [
        detection
        for detection in detections
        if normalize_label(detection.label) in wanted and detection.confidence >= min_confidence
    ]


def filter_by_geometry(
    detections: Iterable[ObjectDetection],
    *,
    image_width: int,
    image_height: int,
    max_area_ratio: float | None = None,
    max_width_ratio: float | None = None,
    max_height_ratio: float | None = None,
    min_area_px: float | None = None,
    per_label_rules: dict[str, GeometryRule] | None = None,
) -> list[ObjectDetection]:
    image_area = image_width * image_height
    filtered: list[ObjectDetection] = []

    for detection in detections:
        box = detection.box
        label_rule = None
        if per_label_rules is not None:
            label_rule = per_label_rules.get(normalize_label(detection.label))

        rule = GeometryRule(
            max_area_ratio=label_rule.max_area_ratio if label_rule and label_rule.max_area_ratio is not None else max_area_ratio,
            max_width_ratio=label_rule.max_width_ratio if label_rule and label_rule.max_width_ratio is not None else max_width_ratio,
            max_height_ratio=label_rule.max_height_ratio if label_rule and label_rule.max_height_ratio is not None else max_height_ratio,
            min_area_px=label_rule.min_area_px if label_rule and label_rule.min_area_px is not None else min_area_px,
        )

        if rule.max_area_ratio is not None and box.area / image_area > rule.max_area_ratio:
            continue
        if rule.max_width_ratio is not None and box.width / image_width > rule.max_width_ratio:
            continue
        if rule.max_height_ratio is not None and box.height / image_height > rule.max_height_ratio:
            continue
        if rule.min_area_px is not None and box.area < rule.min_area_px:
            continue
        filtered.append(detection)

    return filtered


def collapse_labels(
    detections: Iterable[ObjectDetection],
    label_groups: dict[str, Iterable[str]],
) -> list[ObjectDetection]:
    normalized_groups = {
        target_label: {normalize_label(source_label) for source_label in source_labels}
        for target_label, source_labels in label_groups.items()
    }
    collapsed: list[ObjectDetection] = []

    for detection in detections:
        label = detection.label
        normalized_label = normalize_label(label)
        for target_label, source_labels in normalized_groups.items():
            if normalized_label in source_labels:
                label = target_label
                break

        collapsed.append(
            ObjectDetection(
                label=label,
                confidence=detection.confidence,
                box=detection.box,
                source_model=detection.source_model,
                polygon=detection.polygon,
            )
        )

    return collapsed
