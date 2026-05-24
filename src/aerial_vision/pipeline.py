from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aerial_vision.analysis import ImageAnalysis
from aerial_vision.detection import ANIMAL_LABELS, OVERHEAD_VEHICLE_GEOMETRY_RULES, collapse_labels, filter_by_geometry
from aerial_vision.image_io import read_image_size
from aerial_vision.rfdetr_detector import RFDetrObjectDetector
from aerial_vision.tiling import Detector, detect_tiled, non_max_suppression
from aerial_vision.yolo_detector import YoloObjectDetector


PROFILE_CONFIGS = {
    "general": {
        "model": "models/best.pt",
        "objects": ["car", "van", "truck", "bus", "motor", "bicycle", "pedestrian", "people"],
        "min_confidence": 0.15,
        "tile_size": 800,
        "tile_overlap": 240,
        "geometry_preset": "overhead-vehicles",
        "extra_model": "yolo11x.pt",
        "extra_objects": ["horse", "cow", "sheep", "dog", "cat", "bird"],
        "extra_min_confidence": 0.05,
        "collapse_animals": True,
    },
    "vehicles": {
        "model": "models/best.pt",
        "objects": ["car", "van", "truck", "bus", "motor", "bicycle"],
        "min_confidence": 0.15,
        "tile_size": 800,
        "tile_overlap": 240,
        "geometry_preset": "overhead-vehicles",
    },
    "humans": {
        "model": "models/best.pt",
        "objects": ["pedestrian", "people"],
        "min_confidence": 0.20,
        "tile_size": 800,
        "tile_overlap": 240,
        "geometry_preset": "overhead-vehicles",
    },
    "animals": {
        "model": "yolo11x.pt",
        "objects": ["horse", "cow", "sheep", "dog", "cat", "bird"],
        "min_confidence": 0.05,
        "tile_size": 800,
        "tile_overlap": 240,
        "collapse_animals": True,
    },
}


@dataclass(frozen=True)
class DetectionSettings:
    backend: str = "yolo"
    model: str = "yolo11n.pt"
    rfdetr_size: str = "medium"
    min_confidence: float = 0.25
    objects: list[str] | None = None
    extra_model: str | None = None
    extra_objects: list[str] | None = None
    extra_min_confidence: float | None = None
    tile_size: int | None = None
    tile_overlap: int = 128
    nms_iou: float = 0.45
    max_area_ratio: float | None = None
    max_width_ratio: float | None = None
    max_height_ratio: float | None = None
    min_area_px: float | None = None
    geometry_preset: str | None = None
    collapse_animals: bool = False


@dataclass(frozen=True)
class DetectorBundle:
    primary: Detector
    extra: Detector | None = None


def settings_from_profile(profile: str) -> DetectionSettings:
    profile_config = PROFILE_CONFIGS[profile]
    return DetectionSettings(**profile_config)


def build_detector(backend: str, model: str, rfdetr_size: str) -> Detector:
    if backend == "rfdetr":
        checkpoint_path = None if model == "yolo11n.pt" else model
        return RFDetrObjectDetector(model_size=rfdetr_size, checkpoint_path=checkpoint_path)

    return YoloObjectDetector(model)


def build_detector_bundle(settings: DetectionSettings) -> DetectorBundle:
    primary = build_detector(settings.backend, settings.model, settings.rfdetr_size)
    extra = YoloObjectDetector(settings.extra_model) if settings.extra_model else None
    return DetectorBundle(primary=primary, extra=extra)


def run_detector(
    detector: Detector,
    image_path: Path,
    *,
    min_confidence: float,
    objects: set[str] | None,
    tile_size: int | None,
    tile_overlap: int,
    nms_iou: float,
) -> list:
    if tile_size:
        return detect_tiled(
            detector,
            image_path,
            tile_size=tile_size,
            overlap=tile_overlap,
            min_confidence=min_confidence,
            objects_of_interest=objects,
            nms_iou_threshold=nms_iou,
        )

    return detector.detect(
        image_path,
        min_confidence=min_confidence,
        objects_of_interest=objects,
    )


def analyze_image_with_detectors(
    image_path: Path,
    settings: DetectionSettings,
    detectors: DetectorBundle,
) -> ImageAnalysis:
    objects = set(settings.objects) if settings.objects else None
    width_px, height_px = read_image_size(image_path)
    detections = run_detector(
        detectors.primary,
        image_path,
        min_confidence=settings.min_confidence,
        objects=objects,
        tile_size=settings.tile_size,
        tile_overlap=settings.tile_overlap,
        nms_iou=settings.nms_iou,
    )

    if detectors.extra is not None:
        extra_objects = set(settings.extra_objects) if settings.extra_objects else None
        extra_min_confidence = (
            settings.extra_min_confidence
            if settings.extra_min_confidence is not None
            else settings.min_confidence
        )
        detections.extend(
            run_detector(
                detectors.extra,
                image_path,
                min_confidence=extra_min_confidence,
                objects=extra_objects,
                tile_size=settings.tile_size,
                tile_overlap=settings.tile_overlap,
                nms_iou=settings.nms_iou,
            )
        )
        detections = non_max_suppression(detections, iou_threshold=settings.nms_iou)

    detections = filter_by_geometry(
        detections,
        image_width=width_px,
        image_height=height_px,
        max_area_ratio=settings.max_area_ratio,
        max_width_ratio=settings.max_width_ratio,
        max_height_ratio=settings.max_height_ratio,
        min_area_px=settings.min_area_px,
        per_label_rules=OVERHEAD_VEHICLE_GEOMETRY_RULES
        if settings.geometry_preset == "overhead-vehicles"
        else None,
    )
    if settings.collapse_animals:
        detections = collapse_labels(detections, {"animal": ANIMAL_LABELS})

    return ImageAnalysis(
        image_path=str(image_path),
        image_width=width_px,
        image_height=height_px,
        detections=detections,
    )


def analyze_image(image_path: Path, settings: DetectionSettings) -> ImageAnalysis:
    return analyze_image_with_detectors(image_path, settings, build_detector_bundle(settings))
