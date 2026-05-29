from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerial_vision.detection import BoundingBox, normalize_label
from aerial_vision.pipeline import PROFILE_CONFIGS
from aerial_vision.track_video import (
    class_ids_for_model,
    profile_min_confidence,
    profile_model_and_classes,
    track_video,
    yolo_track_kwargs,
)


@dataclass(frozen=True)
class TargetChoice:
    choice: int
    track_id: int
    label: str
    confidence: float
    box: BoundingBox

    @property
    def center(self) -> tuple[float, float]:
        return self.box.center

    def to_dict(self) -> dict[str, object]:
        return {
            "choice": self.choice,
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
            "area_px": self.box.area,
        }


def prepare_pick_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def choose_frame(
    video_path: Path,
    *,
    frame_index: int | None,
    timestamp_sec: float | None,
    resize_width: int | None,
) -> tuple[np.ndarray, int, float, float, int, int, int, int]:
    if frame_index is not None and frame_index < 0:
        raise ValueError("frame_index cannot be negative.")
    if timestamp_sec is not None and timestamp_sec < 0:
        raise ValueError("timestamp_sec cannot be negative.")
    if resize_width is not None and resize_width <= 0:
        raise ValueError("resize_width must be positive.")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Target picking requires OpenCV. Install with: python3 -m pip install -e '.[ml]'") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or source_width <= 0 or source_height <= 0:
        capture.release()
        raise RuntimeError("Video metadata could not be determined.")

    selected_index = frame_index if frame_index is not None else 0
    if timestamp_sec is not None:
        selected_index = int(round(timestamp_sec * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, selected_index)

    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {selected_index} from {video_path}")

    process_width = source_width
    process_height = source_height
    if resize_width is not None and resize_width < source_width:
        process_width = resize_width
        process_height = int(round(source_height * (resize_width / source_width)))
        frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)

    return frame, selected_index, selected_index / fps, fps, source_width, source_height, process_width, process_height


def choices_from_yolo_results(
    results: object,
    *,
    wanted_classes: set[str],
    target_label: str | None = None,
) -> list[TargetChoice]:
    label_filter = normalize_label(target_label) if target_label else None
    pending: list[tuple[int, str, float, BoundingBox]] = []

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None:
            continue

        for box in boxes:
            label = str(result.names[int(box.cls.item())])
            normalized_label = normalize_label(label)
            if normalized_label not in wanted_classes:
                continue
            if label_filter is not None and normalized_label != label_filter:
                continue

            track_id = int(box.id.item())
            confidence = float(box.conf.item())
            x_min, y_min, x_max, y_max = [float(value) for value in box.xyxy[0].tolist()]
            pending.append(
                (
                    track_id,
                    label,
                    confidence,
                    BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                )
            )

    pending.sort(key=lambda item: (item[2], item[3].area), reverse=True)
    return [
        TargetChoice(choice=index, track_id=track_id, label=label, confidence=confidence, box=box)
        for index, (track_id, label, confidence, box) in enumerate(pending, start=1)
    ]


def draw_choices(frame: np.ndarray, choices: list[TargetChoice], output_path: Path) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Target picking requires OpenCV. Install with: python3 -m pip install -e '.[ml]'") from exc

    annotated = frame.copy()
    for choice in choices:
        x_min = int(round(choice.box.x_min))
        y_min = int(round(choice.box.y_min))
        x_max = int(round(choice.box.x_max))
        y_max = int(round(choice.box.y_max))
        color = (0, 255, 255)
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
        label = f"{choice.choice}: #{choice.track_id} {choice.label} {choice.confidence:.2f}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_width = min(annotated.shape[1] - x_min, text_size[0] + 8)
        cv2.rectangle(annotated, (x_min, max(0, y_min - 22)), (x_min + label_width, y_min), color, -1)
        cv2.putText(annotated, label, (x_min + 4, max(14, y_min - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)


def pick_targets(
    video_path: Path,
    *,
    profile: str,
    out_dir: Path,
    frame_index: int | None = None,
    timestamp_sec: float | None = None,
    target_label: str | None = None,
    resize_width: int | None = 640,
    imgsz: int | None = 512,
    max_det: int = 50,
    device: str | None = None,
    half: bool = False,
    tracker: str = "bytetrack.yaml",
) -> dict[str, object]:
    if imgsz is not None and imgsz <= 0:
        raise ValueError("imgsz must be positive.")
    if max_det <= 0:
        raise ValueError("max_det must be positive.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Target picking requires Ultralytics. Install with: python3 -m pip install -e '.[ml]'") from exc

    prepare_pick_dir(out_dir)
    frame, selected_index, selected_time, fps, source_width, source_height, process_width, process_height = choose_frame(
        video_path,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        resize_width=resize_width,
    )

    model_path, wanted_classes = profile_model_and_classes(profile)
    model = YOLO(model_path)
    track_kwargs = yolo_track_kwargs(
        tracker=tracker,
        class_ids=class_ids_for_model(model, wanted_classes),
        min_confidence=profile_min_confidence(profile),
        imgsz=imgsz,
        max_det=max_det,
        device=device,
        half=half,
    )
    results = model.track(frame, **track_kwargs)
    choices = choices_from_yolo_results(results, wanted_classes=wanted_classes, target_label=target_label)

    selection_image = out_dir / "target_choices.jpg"
    draw_choices(frame, choices, selection_image)
    output = {
        "video": {
            "path": str(video_path),
            "fps": fps,
            "source_width_px": source_width,
            "source_height_px": source_height,
            "processed_width_px": process_width,
            "processed_height_px": process_height,
        },
        "profile": profile,
        "frame_index": selected_index,
        "timestamp_sec": selected_time,
        "target_label": target_label,
        "resize_width": resize_width,
        "imgsz": imgsz,
        "max_det": max_det,
        "device": device,
        "half": half,
        "tracker": tracker,
        "selection_image": str(selection_image.relative_to(out_dir)),
        "choices": [choice.to_dict() for choice in choices],
    }
    (out_dir / "target_choices.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def selected_track_id(choices: list[dict[str, object]], choice_number: int) -> int:
    if choice_number <= 0:
        raise ValueError("choice must be positive.")
    for choice in choices:
        if int(choice["choice"]) == choice_number:
            return int(choice["track_id"])
    raise ValueError(f"Choice {choice_number} was not found.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a numbered target-selection frame and optionally track the selected target.")
    parser.add_argument("video", type=Path, help="Path to a video file.")
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="vehicles", help="Detection profile.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for target picker files.")
    parser.add_argument("--frame-index", type=int, help="Frame index to use for target selection.")
    parser.add_argument("--timestamp-sec", type=float, help="Timestamp to use for target selection.")
    parser.add_argument("--target-label", help="Only show choices with this label, e.g. truck.")
    parser.add_argument("--choice", type=int, help="Choice number to track after creating the picker image.")
    parser.add_argument("--detect-every", type=int, default=15, help="Detector refresh interval if --choice is provided.")
    parser.add_argument("--max-frames", type=int, help="Optional tracking frame limit if --choice is provided.")
    parser.add_argument("--min-track-frames", type=int, default=3, help="Minimum frames for exported tracks if --choice is provided.")
    parser.add_argument("--resize-width", type=int, default=640, help="Resize frames to this width before picking/tracking.")
    parser.add_argument("--imgsz", type=int, default=512, help="YOLO inference image size.")
    parser.add_argument("--max-det", type=int, default=50, help="Maximum detections per detector refresh.")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu, mps, 0, cuda:0.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on supported GPU devices.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = pick_targets(
        args.video,
        profile=args.profile,
        out_dir=args.out,
        frame_index=args.frame_index,
        timestamp_sec=args.timestamp_sec,
        target_label=args.target_label,
        resize_width=args.resize_width,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        tracker=args.tracker,
    )
    print(f"selection_image: {args.out / output['selection_image']}")
    print(f"choices: {args.out / 'target_choices.json'}")
    for choice in output["choices"]:
        print(
            f"{choice['choice']}: track_id={choice['track_id']} "
            f"label={choice['label']} conf={choice['confidence']:.2f} "
            f"center={choice['center']}"
        )

    if args.choice is None:
        return

    target_id = selected_track_id(output["choices"], args.choice)
    track_output = track_video(
        args.video,
        profile=args.profile,
        out_dir=args.out / "tracking",
        max_frames=args.max_frames,
        min_track_frames=args.min_track_frames,
        resize_width=args.resize_width,
        target_id=target_id,
        detect_every=args.detect_every,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        tracker=args.tracker,
    )
    print(f"selected_target_id: {target_id}")
    print(f"preview: {args.out / 'tracking' / track_output['preview_video']}")
    if "target_lock" in track_output:
        print(f"target_lock: {args.out / 'tracking' / track_output['target_lock']}")


if __name__ == "__main__":
    main()
