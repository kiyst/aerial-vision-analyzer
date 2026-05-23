from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from aerial_vision.detection import BoundingBox, ObjectDetection, normalize_label


class Detector(Protocol):
    def detect(
        self,
        image_path: str | Path,
        *,
        min_confidence: float = 0.25,
        objects_of_interest: set[str] | None = None,
    ) -> list[ObjectDetection]:
        ...


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    width: int
    height: int


def generate_tiles(image_width: int, image_height: int, tile_size: int, overlap: int) -> list[Tile]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size.")

    stride = tile_size - overlap
    x_positions = _tile_starts(image_width, tile_size, stride)
    y_positions = _tile_starts(image_height, tile_size, stride)

    return [
        Tile(
            x=x,
            y=y,
            width=min(tile_size, image_width - x),
            height=min(tile_size, image_height - y),
        )
        for y in y_positions
        for x in x_positions
    ]


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def detect_tiled(
    detector: Detector,
    image_path: Path,
    *,
    tile_size: int,
    overlap: int,
    min_confidence: float,
    objects_of_interest: set[str] | None,
    nms_iou_threshold: float = 0.45,
) -> list[ObjectDetection]:
    if not 0 <= nms_iou_threshold <= 1:
        raise ValueError("nms_iou_threshold must be between 0 and 1.")

    detections: list[ObjectDetection] = []

    with Image.open(image_path).convert("RGB") as image:
        tiles = generate_tiles(image.width, image.height, tile_size, overlap)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for index, tile in enumerate(tiles):
                tile_image = image.crop((tile.x, tile.y, tile.x + tile.width, tile.y + tile.height))
                tile_path = temp_path / f"tile_{index}.png"
                tile_image.save(tile_path)
                tile_detections = detector.detect(
                    tile_path,
                    min_confidence=min_confidence,
                    objects_of_interest=objects_of_interest,
                )
                detections.extend(shift_detections(tile_detections, x_offset=tile.x, y_offset=tile.y))

    return non_max_suppression(detections, iou_threshold=nms_iou_threshold)


def shift_detections(
    detections: list[ObjectDetection],
    *,
    x_offset: float,
    y_offset: float,
) -> list[ObjectDetection]:
    shifted: list[ObjectDetection] = []
    for detection in detections:
        box = detection.box
        polygon = None
        if detection.polygon is not None:
            polygon = tuple((x + x_offset, y + y_offset) for x, y in detection.polygon)

        shifted.append(
            ObjectDetection(
                label=detection.label,
                confidence=detection.confidence,
                box=BoundingBox(
                    x_min=box.x_min + x_offset,
                    y_min=box.y_min + y_offset,
                    x_max=box.x_max + x_offset,
                    y_max=box.y_max + y_offset,
                ),
                source_model=detection.source_model,
                polygon=polygon,
            )
        )
    return shifted


def non_max_suppression(detections: list[ObjectDetection], *, iou_threshold: float) -> list[ObjectDetection]:
    kept: list[ObjectDetection] = []

    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        same_label_conflict = any(
            normalize_label(detection.label) == normalize_label(existing.label)
            and bbox_iou(detection.box, existing.box) > iou_threshold
            for existing in kept
        )
        if not same_label_conflict:
            kept.append(detection)

    return kept


def bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    x_min = max(first.x_min, second.x_min)
    y_min = max(first.y_min, second.y_min)
    x_max = min(first.x_max, second.x_max)
    y_max = min(first.y_max, second.y_max)

    intersection_width = max(0.0, x_max - x_min)
    intersection_height = max(0.0, y_max - y_min)
    intersection_area = intersection_width * intersection_height

    union_area = first.area + second.area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area

