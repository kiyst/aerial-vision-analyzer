from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerial_vision.detection import BoundingBox, normalize_label
from aerial_vision.pipeline import PROFILE_CONFIGS


@dataclass(frozen=True)
class TrackObservation:
    frame_index: int
    timestamp_sec: float
    track_id: int
    label: str
    confidence: float
    box: BoundingBox
    center: tuple[float, float]
    color_histogram: list[float]
    aspect_ratio: float
    area_px: float
    color_similarity: float | None = None
    shape_similarity: float | None = None
    identity_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "track_id": self.track_id,
            "label": self.label,
            "confidence": self.confidence,
            "box": {
                "x_min": self.box.x_min,
                "y_min": self.box.y_min,
                "x_max": self.box.x_max,
                "y_max": self.box.y_max,
            },
            "center": self.center,
            "color_histogram": self.color_histogram,
            "aspect_ratio": self.aspect_ratio,
            "area_px": self.area_px,
            "color_similarity": self.color_similarity,
            "shape_similarity": self.shape_similarity,
            "identity_score": self.identity_score,
        }


@dataclass(frozen=True)
class TargetLockObservation:
    frame_index: int
    timestamp_sec: float
    target_id: int
    state: str
    reason: str
    center_offset: tuple[float, float] | None
    normalized_offset: tuple[float, float] | None
    box: BoundingBox | None
    confidence: float | None
    identity_score: float | None
    overlap_risk: float

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "target_id": self.target_id,
            "state": self.state,
            "reason": self.reason,
            "center_offset": self.center_offset,
            "normalized_offset": self.normalized_offset,
            "box": None
            if self.box is None
            else {
                "x_min": self.box.x_min,
                "y_min": self.box.y_min,
                "x_max": self.box.x_max,
                "y_max": self.box.y_max,
            },
            "confidence": self.confidence,
            "identity_score": self.identity_score,
            "overlap_risk": self.overlap_risk,
        }


@dataclass(frozen=True)
class ControlIntent:
    frame_index: int
    timestamp_sec: float
    mode: str
    reason: str
    yaw_rate_deg_s: float
    camera_pitch_rate_deg_s: float
    forward_mps: float
    normalized_offset: tuple[float, float] | None
    center_error: float | None
    tag_enabled: bool
    box_area_ratio: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "mode": self.mode,
            "reason": self.reason,
            "yaw_rate_deg_s": self.yaw_rate_deg_s,
            "camera_pitch_rate_deg_s": self.camera_pitch_rate_deg_s,
            "forward_mps": self.forward_mps,
            "normalized_offset": self.normalized_offset,
            "center_error": self.center_error,
            "tag_enabled": self.tag_enabled,
            "box_area_ratio": self.box_area_ratio,
        }


def prepare_tracking_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def clamp_box(box: BoundingBox, *, width: int, height: int) -> tuple[int, int, int, int]:
    x_min = max(0, min(width, int(round(box.x_min))))
    y_min = max(0, min(height, int(round(box.y_min))))
    x_max = max(0, min(width, int(round(box.x_max))))
    y_max = max(0, min(height, int(round(box.y_max))))
    return x_min, y_min, x_max, y_max


def color_histogram(frame: np.ndarray, box: BoundingBox, *, bins: int = 8) -> list[float]:
    height, width = frame.shape[:2]
    x_min, y_min, x_max, y_max = clamp_box(box, width=width, height=height)
    if x_max <= x_min or y_max <= y_min:
        return [0.0] * (bins * 3)

    crop = frame[y_min:y_max, x_min:x_max]
    channels = []
    for channel_index in range(3):
        histogram, _ = np.histogram(crop[:, :, channel_index], bins=bins, range=(0, 256))
        channels.extend(histogram.astype(float).tolist())

    total = sum(channels)
    if total <= 0:
        return [0.0] * (bins * 3)
    return [value / total for value in channels]


def cosine_similarity(first: list[float], second: list[float]) -> float:
    first_array = np.array(first, dtype=float)
    second_array = np.array(second, dtype=float)
    denominator = float(np.linalg.norm(first_array) * np.linalg.norm(second_array))
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(first_array, second_array) / denominator)))


def shape_similarity(current: TrackObservation, previous: TrackObservation) -> float:
    aspect_delta = abs(current.aspect_ratio - previous.aspect_ratio) / max(previous.aspect_ratio, 1e-6)
    area_delta = abs(current.area_px - previous.area_px) / max(previous.area_px, 1e-6)
    score = 1.0 - min(1.0, (aspect_delta + area_delta) / 2)
    return max(0.0, min(1.0, score))


def identity_score(color_score: float, shape_score: float) -> float:
    return 0.65 * color_score + 0.35 * shape_score


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


def target_center_offsets(observation: TrackObservation, *, frame_width: int, frame_height: int) -> tuple[tuple[float, float], tuple[float, float]]:
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    offset_x = observation.center[0] - frame_center_x
    offset_y = observation.center[1] - frame_center_y
    return (
        (offset_x, offset_y),
        (offset_x / frame_center_x, offset_y / frame_center_y),
    )


def max_same_label_overlap(target: TrackObservation, observations: list[TrackObservation]) -> float:
    overlaps = [
        bbox_iou(target.box, observation.box)
        for observation in observations
        if observation.track_id != target.track_id and normalize_label(observation.label) == normalize_label(target.label)
    ]
    return max(overlaps, default=0.0)


def points_for_box(box: BoundingBox, *, frame_width: int, frame_height: int, grid_size: int = 5) -> np.ndarray:
    x_min, y_min, x_max, y_max = clamp_box(box, width=frame_width, height=frame_height)
    if x_max <= x_min or y_max <= y_min:
        return np.empty((0, 1, 2), dtype=np.float32)

    x_padding = max(1, int((x_max - x_min) * 0.15))
    y_padding = max(1, int((y_max - y_min) * 0.15))
    x_values = np.linspace(x_min + x_padding, x_max - x_padding, grid_size)
    y_values = np.linspace(y_min + y_padding, y_max - y_padding, grid_size)
    points = [[x, y] for y in y_values for x in x_values]
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)


def shifted_box_from_points(
    previous_box: BoundingBox,
    previous_points: np.ndarray,
    current_points: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
) -> BoundingBox:
    deltas = current_points.reshape(-1, 2) - previous_points.reshape(-1, 2)
    dx = float(np.median(deltas[:, 0]))
    dy = float(np.median(deltas[:, 1]))
    x_min = max(0.0, min(frame_width, previous_box.x_min + dx))
    y_min = max(0.0, min(frame_height, previous_box.y_min + dy))
    x_max = max(0.0, min(frame_width, previous_box.x_max + dx))
    y_max = max(0.0, min(frame_height, previous_box.y_max + dy))
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def optical_flow_observation(
    *,
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    current_frame: np.ndarray,
    previous_observation: TrackObservation,
    frame_index: int,
    timestamp_sec: float,
    min_points: int = 6,
) -> TrackObservation | None:
    import cv2

    previous_points = points_for_box(
        previous_observation.box,
        frame_width=previous_gray.shape[1],
        frame_height=previous_gray.shape[0],
    )
    if len(previous_points) < min_points:
        return None

    current_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if current_points is None or status is None:
        return None

    valid = status.reshape(-1) == 1
    if int(valid.sum()) < min_points:
        return None

    box = shifted_box_from_points(
        previous_observation.box,
        previous_points[valid],
        current_points[valid],
        frame_width=current_gray.shape[1],
        frame_height=current_gray.shape[0],
    )
    return observation_from_box(
        frame=current_frame,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        track_id=previous_observation.track_id,
        label=previous_observation.label,
        confidence=max(0.0, previous_observation.confidence * 0.98),
        box=box,
        previous=previous_observation,
    )


def target_lock_state(
    target: TrackObservation | None,
    observations: list[TrackObservation],
    *,
    target_id: int,
    frame_index: int,
    timestamp_sec: float,
    frame_width: int,
    frame_height: int,
    identity_threshold: float = 0.65,
    overlap_threshold: float = 0.25,
) -> TargetLockObservation:
    if target is None:
        return TargetLockObservation(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            target_id=target_id,
            state="lost",
            reason="target track not present in this frame",
            center_offset=None,
            normalized_offset=None,
            box=None,
            confidence=None,
            identity_score=None,
            overlap_risk=0.0,
        )

    center_offset, normalized_offset = target_center_offsets(target, frame_width=frame_width, frame_height=frame_height)
    overlap_risk = max_same_label_overlap(target, observations)
    score = target.identity_score
    state = "locked"
    reason = "target visible"

    if score is not None and score < identity_threshold:
        state = "weak_lock"
        reason = "target appearance changed"
    if overlap_risk >= overlap_threshold:
        state = "id_switch_risk" if state == "locked" else state
        reason = "target overlaps a similar object"

    return TargetLockObservation(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        target_id=target_id,
        state=state,
        reason=reason,
        center_offset=center_offset,
        normalized_offset=normalized_offset,
        box=target.box,
        confidence=target.confidence,
        identity_score=target.identity_score,
        overlap_risk=overlap_risk,
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def control_intent_from_lock(
    observation: TargetLockObservation,
    *,
    previous_offset: tuple[float, float] | None = None,
    previous_box_area: float | None = None,
    tag_enabled: bool = False,
    max_yaw_rate_deg_s: float = 35.0,
    max_pitch_rate_deg_s: float = 25.0,
    max_forward_mps: float = 6.0,
    follow_forward_mps: float = 2.0,
    center_deadband: float = 0.08,
    follow_center_gate: float = 0.35,
    tag_center_gate: float = 0.45,
    away_area_ratio_threshold: float = 0.97,
    search_yaw_rate_deg_s: float = 18.0,
) -> ControlIntent:
    if observation.normalized_offset is None:
        search_direction = 1.0
        if previous_offset is not None and abs(previous_offset[0]) > 0.05:
            search_direction = 1.0 if previous_offset[0] > 0 else -1.0
        return ControlIntent(
            frame_index=observation.frame_index,
            timestamp_sec=observation.timestamp_sec,
            mode="search",
            reason="target not visible; rotate toward last known horizontal side",
            yaw_rate_deg_s=search_direction * search_yaw_rate_deg_s,
            camera_pitch_rate_deg_s=0.0,
            forward_mps=0.0,
            normalized_offset=None,
            center_error=None,
            tag_enabled=tag_enabled,
            box_area_ratio=None,
        )

    offset_x, offset_y = observation.normalized_offset
    center_error = float((offset_x * offset_x + offset_y * offset_y) ** 0.5)
    yaw_rate = clamp(offset_x * max_yaw_rate_deg_s, -max_yaw_rate_deg_s, max_yaw_rate_deg_s)
    pitch_rate = clamp(offset_y * max_pitch_rate_deg_s, -max_pitch_rate_deg_s, max_pitch_rate_deg_s)
    box = getattr(observation, "box", None)
    box_area = box.area if box is not None else None
    box_area_ratio = None
    if box_area is not None and previous_box_area is not None and previous_box_area > 0:
        box_area_ratio = box_area / previous_box_area

    mode = "center"
    reason = "target visible; center camera on selected target"
    forward = 0.0
    if observation.state in {"weak_lock", "id_switch_risk"}:
        mode = "hold"
        reason = observation.reason
    elif tag_enabled:
        mode = "tag"
        reason = "tag mode active; target centered enough for forward intent"
        if center_error <= tag_center_gate:
            forward = max_forward_mps * max(0.0, 1.0 - center_error / max(tag_center_gate, 1e-6))
        else:
            forward = 0.0
            reason = "tag mode active; center target before forward intent"
    elif (
        box_area_ratio is not None
        and box_area_ratio < away_area_ratio_threshold
        and center_error <= follow_center_gate
    ):
        mode = "follow"
        reason = "target appears to be moving away; keep it framed"
        forward = follow_forward_mps
    elif center_error <= center_deadband:
        mode = "centered"
        reason = "target centered; hold position"

    return ControlIntent(
        frame_index=observation.frame_index,
        timestamp_sec=observation.timestamp_sec,
        mode=mode,
        reason=reason,
        yaw_rate_deg_s=yaw_rate,
        camera_pitch_rate_deg_s=pitch_rate,
        forward_mps=forward,
        normalized_offset=observation.normalized_offset,
        center_error=center_error,
        tag_enabled=tag_enabled,
        box_area_ratio=box_area_ratio,
    )


def observation_from_box(
    *,
    frame: np.ndarray,
    frame_index: int,
    timestamp_sec: float,
    track_id: int,
    label: str,
    confidence: float,
    box: BoundingBox,
    previous: TrackObservation | None = None,
) -> TrackObservation:
    histogram = color_histogram(frame, box)
    center = box.center
    aspect_ratio = box.width / max(box.height, 1e-6)
    current = TrackObservation(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        track_id=track_id,
        label=label,
        confidence=confidence,
        box=box,
        center=center,
        color_histogram=histogram,
        aspect_ratio=aspect_ratio,
        area_px=box.area,
    )

    if previous is None:
        return current

    color_score = cosine_similarity(histogram, previous.color_histogram)
    shape_score = shape_similarity(current, previous)
    return TrackObservation(
        frame_index=current.frame_index,
        timestamp_sec=current.timestamp_sec,
        track_id=current.track_id,
        label=current.label,
        confidence=current.confidence,
        box=current.box,
        center=current.center,
        color_histogram=current.color_histogram,
        aspect_ratio=current.aspect_ratio,
        area_px=current.area_px,
        color_similarity=color_score,
        shape_similarity=shape_score,
        identity_score=identity_score(color_score, shape_score),
    )


def profile_model_and_classes(profile: str) -> tuple[str, set[str]]:
    config = PROFILE_CONFIGS[profile]
    return str(config["model"]), {normalize_label(label) for label in config["objects"]}


def profile_min_confidence(profile: str) -> float:
    return float(PROFILE_CONFIGS[profile]["min_confidence"])


def class_ids_for_model(model: object, wanted_classes: set[str]) -> list[int] | None:
    names = getattr(model, "names", None)
    if names is None:
        return None
    items = names.items() if isinstance(names, dict) else enumerate(names)
    class_ids = [
        int(class_id)
        for class_id, label in items
        if normalize_label(str(label)) in wanted_classes
    ]
    return class_ids or None


def yolo_track_kwargs(
    *,
    tracker: str,
    class_ids: list[int] | None,
    min_confidence: float,
    imgsz: int | None,
    max_det: int,
    device: str | None,
    half: bool,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "persist": True,
        "tracker": tracker,
        "verbose": False,
        "conf": min_confidence,
        "max_det": max_det,
    }
    if class_ids is not None:
        kwargs["classes"] = class_ids
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    if device:
        kwargs["device"] = device
    if half:
        kwargs["half"] = True
    return kwargs


def select_target_observation(observations: list[TrackObservation], target_label: str) -> TrackObservation | None:
    normalized_target = normalize_label(target_label)
    candidates = [
        observation
        for observation in observations
        if normalize_label(observation.label) == normalized_target
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda observation: (observation.confidence, observation.area_px))


def draw_track(frame: np.ndarray, observation: TrackObservation) -> None:
    import cv2

    x_min, y_min, x_max, y_max = clamp_box(observation.box, width=frame.shape[1], height=frame.shape[0])
    color = (0, 255, 255)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    score = observation.identity_score
    score_text = "" if score is None else f" sim={score:.2f}"
    text = f"#{observation.track_id} {observation.label} {observation.confidence:.2f}{score_text}"
    cv2.rectangle(frame, (x_min, max(0, y_min - 20)), (min(frame.shape[1], x_min + 260), y_min), color, -1)
    cv2.putText(frame, text, (x_min + 3, max(14, y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_target_lock(frame: np.ndarray, observation: TargetLockObservation) -> None:
    import cv2

    if observation.box is None:
        cv2.putText(frame, f"TARGET #{observation.target_id}: LOST", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return

    x_min, y_min, x_max, y_max = clamp_box(observation.box, width=frame.shape[1], height=frame.shape[0])
    color = (0, 255, 0) if observation.state == "locked" else (0, 165, 255)
    if observation.state == "lost":
        color = (0, 0, 255)

    target_center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 4)
    cv2.circle(frame, target_center, 6, color, -1)
    cv2.circle(frame, frame_center, 7, (255, 255, 255), 2)
    cv2.line(frame, frame_center, target_center, color, 2)
    score = "n/a" if observation.identity_score is None else f"{observation.identity_score:.2f}"
    text = f"TARGET #{observation.target_id} {observation.state} score={score}"
    cv2.rectangle(frame, (16, 12), (min(frame.shape[1], 520), 52), color, -1)
    cv2.putText(frame, text, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)


def draw_control_intent(frame: np.ndarray, intent: ControlIntent) -> None:
    import cv2

    color = (80, 220, 80)
    if intent.mode == "search":
        color = (0, 165, 255)
    if intent.mode == "hold":
        color = (0, 0, 255)
    if intent.mode == "tag":
        color = (255, 120, 30)

    tag = " TAG" if intent.tag_enabled else ""
    line_1 = f"INTENT {intent.mode.upper()}{tag} yaw={intent.yaw_rate_deg_s:+.1f} pitch={intent.camera_pitch_rate_deg_s:+.1f}"
    line_2 = f"forward={intent.forward_mps:.1f} m/s"
    cv2.rectangle(frame, (16, 58), (min(frame.shape[1], 600), 112), color, -1)
    cv2.putText(frame, line_1, (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    cv2.putText(frame, line_2, (24, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)


def track_video(
    video_path: Path,
    *,
    profile: str,
    out_dir: Path,
    sample_every_sec: float = 0.0,
    max_frames: int | None = None,
    min_track_frames: int = 3,
    resize_width: int | None = None,
    target_id: int | None = None,
    target_label: str | None = None,
    detect_every: int = 1,
    imgsz: int | None = None,
    max_det: int = 100,
    device: str | None = None,
    half: bool = False,
    tracker: str = "bytetrack.yaml",
    max_intent_yaw_rate_deg_s: float = 35.0,
    max_intent_pitch_rate_deg_s: float = 25.0,
    max_intent_forward_mps: float = 6.0,
    follow_intent_forward_mps: float = 2.0,
    intent_center_deadband: float = 0.08,
    tag_mode: bool = False,
) -> dict[str, object]:
    if sample_every_sec < 0:
        raise ValueError("sample_every_sec cannot be negative.")
    if min_track_frames <= 0:
        raise ValueError("min_track_frames must be positive.")
    if resize_width is not None and resize_width <= 0:
        raise ValueError("resize_width must be positive.")
    if detect_every <= 0:
        raise ValueError("detect_every must be positive.")
    if detect_every > 1 and target_id is None and target_label is None:
        raise ValueError("detect_every > 1 requires --target-id or --target-label because only the selected target is tracked between detections.")
    if imgsz is not None and imgsz <= 0:
        raise ValueError("imgsz must be positive.")
    if max_det <= 0:
        raise ValueError("max_det must be positive.")

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Tracking requires OpenCV and Ultralytics. Install with: python3 -m pip install -e '.[ml]'") from exc

    model_path, wanted_classes = profile_model_and_classes(profile)
    min_confidence = profile_min_confidence(profile)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Video metadata could not be determined.")

    process_width = width
    process_height = height
    if resize_width is not None and resize_width < width:
        process_width = resize_width
        process_height = int(round(height * (resize_width / width)))

    frame_step = max(1, int(round(fps * sample_every_sec))) if sample_every_sec > 0 else 1
    prepare_tracking_dir(out_dir)
    preview_path = out_dir / "tracked_preview.mp4"
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / frame_step,
        (process_width, process_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not write tracking preview: {preview_path}")

    model = YOLO(model_path)
    class_ids = class_ids_for_model(model, wanted_classes)
    track_kwargs = yolo_track_kwargs(
        tracker=tracker,
        class_ids=class_ids,
        min_confidence=min_confidence,
        imgsz=imgsz,
        max_det=max_det,
        device=device,
        half=half,
    )
    previous_by_track: dict[int, TrackObservation] = {}
    observations: list[TrackObservation] = []
    target_lock_observations: list[TargetLockObservation] = []
    control_intents: list[ControlIntent] = []
    last_target_observation: TrackObservation | None = None
    last_target_offset: tuple[float, float] | None = None
    last_target_box_area: float | None = None
    active_target_id = target_id
    previous_gray: np.ndarray | None = None
    detector_frames = 0
    optical_flow_frames = 0
    frames_processed = 0
    frame_index = 0
    start_time = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step != 0:
            frame_index += 1
            continue
        if max_frames is not None and frames_processed >= max_frames:
            break

        process_frame = frame
        if process_width != width or process_height != height:
            process_frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
        current_gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)

        frame_observations: list[TrackObservation] = []
        use_detector = frames_processed % detect_every == 0
        if use_detector:
            detector_frames += 1
            results = model.track(process_frame, **track_kwargs)
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None or boxes.id is None:
                    continue

                for box in boxes:
                    label = result.names[int(box.cls.item())]
                    if normalize_label(label) not in wanted_classes:
                        continue

                    track_id = int(box.id.item())
                    confidence = float(box.conf.item())
                    x_min, y_min, x_max, y_max = [float(value) for value in box.xyxy[0].tolist()]
                    bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
                    observation = observation_from_box(
                        frame=process_frame,
                        frame_index=frame_index,
                        timestamp_sec=frame_index / fps,
                        track_id=track_id,
                        label=label,
                        confidence=confidence,
                        box=bbox,
                        previous=previous_by_track.get(track_id),
                    )
                    previous_by_track[track_id] = observation
                    frame_observations.append(observation)
                    observations.append(observation)
                    draw_track(process_frame, observation)
                    if active_target_id is not None and track_id == active_target_id:
                        last_target_observation = observation

            if active_target_id is None and target_label is not None:
                selected = select_target_observation(frame_observations, target_label)
                if selected is not None:
                    active_target_id = selected.track_id
                    last_target_observation = selected
        elif active_target_id is not None and previous_gray is not None and last_target_observation is not None:
            optical_flow_frames += 1
            optical_observation = optical_flow_observation(
                previous_gray=previous_gray,
                current_gray=current_gray,
                current_frame=process_frame,
                previous_observation=last_target_observation,
                frame_index=frame_index,
                timestamp_sec=frame_index / fps,
            )
            if optical_observation is not None:
                last_target_observation = optical_observation
                previous_by_track[active_target_id] = optical_observation
                frame_observations.append(optical_observation)
                observations.append(optical_observation)

        if active_target_id is not None:
            target = next((observation for observation in frame_observations if observation.track_id == active_target_id), None)
            lock_observation = target_lock_state(
                target,
                frame_observations,
                target_id=active_target_id,
                frame_index=frame_index,
                timestamp_sec=frame_index / fps,
                frame_width=process_width,
                frame_height=process_height,
            )
            target_lock_observations.append(lock_observation)
            draw_target_lock(process_frame, lock_observation)
            intent = control_intent_from_lock(
                lock_observation,
                previous_offset=last_target_offset,
                previous_box_area=last_target_box_area,
                tag_enabled=tag_mode,
                max_yaw_rate_deg_s=max_intent_yaw_rate_deg_s,
                max_pitch_rate_deg_s=max_intent_pitch_rate_deg_s,
                max_forward_mps=max_intent_forward_mps,
                follow_forward_mps=follow_intent_forward_mps,
                center_deadband=intent_center_deadband,
            )
            control_intents.append(intent)
            draw_control_intent(process_frame, intent)
            if lock_observation.normalized_offset is not None:
                last_target_offset = lock_observation.normalized_offset
            if lock_observation.box is not None:
                last_target_box_area = lock_observation.box.area

        writer.write(process_frame)
        previous_gray = current_gray
        frames_processed += 1
        frame_index += 1

    capture.release()
    writer.release()
    elapsed_sec = time.perf_counter() - start_time

    tracks_by_id: dict[int, list[TrackObservation]] = {}
    for observation in observations:
        tracks_by_id.setdefault(observation.track_id, []).append(observation)
    kept_tracks = {
        track_id: track_observations
        for track_id, track_observations in tracks_by_id.items()
        if len(track_observations) >= min_track_frames
    }

    output = {
        "video": {
            "path": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "width_px": width,
            "height_px": height,
            "processed_width_px": process_width,
            "processed_height_px": process_height,
        },
        "profile": profile,
        "tracker": tracker,
        "sample_every_sec": sample_every_sec,
        "min_track_frames": min_track_frames,
        "resize_width": resize_width,
        "requested_target_id": target_id,
        "target_label": target_label,
        "selected_target_id": active_target_id,
        "detect_every": detect_every,
        "imgsz": imgsz,
        "max_det": max_det,
        "device": device,
        "half": half,
        "frames_processed": frames_processed,
        "detector_frames": detector_frames,
        "optical_flow_frames": optical_flow_frames,
        "elapsed_sec": elapsed_sec,
        "processed_fps": frames_processed / elapsed_sec if elapsed_sec > 0 else None,
        "raw_track_count": len(tracks_by_id),
        "track_count": len(kept_tracks),
        "preview_video": str(preview_path.relative_to(out_dir)),
        "tracks": {
            str(track_id): [observation.to_dict() for observation in track_observations]
            for track_id, track_observations in sorted(kept_tracks.items())
        },
    }
    (out_dir / "tracks.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if active_target_id is not None:
        states: dict[str, int] = {}
        for observation in target_lock_observations:
            states[observation.state] = states.get(observation.state, 0) + 1
        target_output = {
            "target_id": active_target_id,
            "requested_target_id": target_id,
            "target_label": target_label,
            "video": output["video"],
            "profile": profile,
            "tracker": tracker,
            "detect_every": detect_every,
            "imgsz": imgsz,
            "max_det": max_det,
            "device": device,
            "half": half,
            "frames_processed": frames_processed,
            "detector_frames": detector_frames,
            "optical_flow_frames": optical_flow_frames,
            "state_counts": states,
            "observations": [observation.to_dict() for observation in target_lock_observations],
            "control_intents": [intent.to_dict() for intent in control_intents],
        }
        (out_dir / "target_lock.json").write_text(json.dumps(target_output, indent=2) + "\n", encoding="utf-8")
        output["target_lock"] = "target_lock.json"
        intent_output = {
            "target_id": active_target_id,
            "video": output["video"],
            "profile": profile,
            "frames_processed": frames_processed,
            "intent_settings": {
                "max_yaw_rate_deg_s": max_intent_yaw_rate_deg_s,
                "max_pitch_rate_deg_s": max_intent_pitch_rate_deg_s,
                "max_forward_mps": max_intent_forward_mps,
                "follow_forward_mps": follow_intent_forward_mps,
                "center_deadband": intent_center_deadband,
                "tag_mode": tag_mode,
            },
            "intents": [intent.to_dict() for intent in control_intents],
        }
        (out_dir / "control_intent.json").write_text(json.dumps(intent_output, indent=2) + "\n", encoding="utf-8")
        output["control_intent"] = "control_intent.json"
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track objects through a video and write track IDs plus appearance scores.")
    parser.add_argument("video", type=Path, help="Path to a video file.")
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="vehicles", help="Detection profile to track.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for tracking results.")
    parser.add_argument("--sample-every-sec", type=float, default=0.0, help="Optional seconds between tracked frames. Default tracks every frame.")
    parser.add_argument("--max-frames", type=int, help="Optional limit for quick smoke tests.")
    parser.add_argument("--min-track-frames", type=int, default=3, help="Only export tracks observed for at least this many processed frames.")
    parser.add_argument("--resize-width", type=int, help="Resize frames to this width before tracking. Preserves aspect ratio.")
    parser.add_argument("--target-id", type=int, help="Highlight and log target-lock state for this track ID.")
    parser.add_argument("--target-label", help="Automatically lock onto the highest-confidence track with this label, e.g. truck.")
    parser.add_argument("--detect-every", type=int, default=1, help="Run detector every N processed frames and use optical flow for the target between detections. Requires --target-id or --target-label when greater than 1.")
    parser.add_argument("--imgsz", type=int, help="YOLO inference image size. Smaller values can improve speed.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per detector refresh.")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu, mps, 0, cuda:0.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on supported GPU devices.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config, e.g. bytetrack.yaml or botsort.yaml.")
    parser.add_argument("--max-intent-yaw-rate-deg-s", type=float, default=35.0, help="Maximum yaw-rate intent written to control_intent.json.")
    parser.add_argument("--max-intent-pitch-rate-deg-s", type=float, default=25.0, help="Maximum camera pitch-rate intent written to control_intent.json.")
    parser.add_argument("--max-intent-forward-mps", type=float, default=6.0, help="Maximum forward-speed intent written to control_intent.json.")
    parser.add_argument("--follow-intent-forward-mps", type=float, default=2.0, help="Forward-speed intent used only to keep a non-tag target framed when it appears to move away.")
    parser.add_argument("--intent-center-deadband", type=float, default=0.08, help="Normalized center error below which forward approach intent is allowed.")
    parser.add_argument("--tag-mode", action="store_true", help="Allow higher-function tag movement intent for the selected target.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = track_video(
        args.video,
        profile=args.profile,
        out_dir=args.out,
        sample_every_sec=args.sample_every_sec,
        max_frames=args.max_frames,
        min_track_frames=args.min_track_frames,
        resize_width=args.resize_width,
        target_id=args.target_id,
        target_label=args.target_label,
        detect_every=args.detect_every,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        tracker=args.tracker,
        max_intent_yaw_rate_deg_s=args.max_intent_yaw_rate_deg_s,
        max_intent_pitch_rate_deg_s=args.max_intent_pitch_rate_deg_s,
        max_intent_forward_mps=args.max_intent_forward_mps,
        follow_intent_forward_mps=args.follow_intent_forward_mps,
        intent_center_deadband=args.intent_center_deadband,
        tag_mode=args.tag_mode,
    )
    print(f"video: {args.video}")
    print(f"profile: {args.profile}")
    print(f"frames_processed: {output['frames_processed']}")
    print(f"detector_frames: {output['detector_frames']}")
    print(f"optical_flow_frames: {output['optical_flow_frames']}")
    print(f"elapsed_sec: {output['elapsed_sec']:.1f}")
    if output["processed_fps"] is not None:
        print(f"processed_fps: {output['processed_fps']:.2f}")
    print(f"track_count: {output['track_count']}")
    print(f"processed_size: {output['video']['processed_width_px']} x {output['video']['processed_height_px']}")
    print(f"tracks: {args.out / 'tracks.json'}")
    if "target_lock" in output:
        print(f"target_lock: {args.out / output['target_lock']}")
    if "control_intent" in output:
        print(f"control_intent: {args.out / output['control_intent']}")
    print(f"preview: {args.out / output['preview_video']}")


if __name__ == "__main__":
    main()
