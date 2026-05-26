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


def track_video(
    video_path: Path,
    *,
    profile: str,
    out_dir: Path,
    sample_every_sec: float = 0.0,
    max_frames: int | None = None,
    min_track_frames: int = 3,
    resize_width: int | None = None,
    tracker: str = "bytetrack.yaml",
) -> dict[str, object]:
    if sample_every_sec < 0:
        raise ValueError("sample_every_sec cannot be negative.")
    if min_track_frames <= 0:
        raise ValueError("min_track_frames must be positive.")
    if resize_width is not None and resize_width <= 0:
        raise ValueError("resize_width must be positive.")

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Tracking requires OpenCV and Ultralytics. Install with: python3 -m pip install -e '.[ml]'") from exc

    model_path, wanted_classes = profile_model_and_classes(profile)
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
    previous_by_track: dict[int, TrackObservation] = {}
    observations: list[TrackObservation] = []
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

        results = model.track(process_frame, persist=True, tracker=tracker, verbose=False)
        frame_observations: list[TrackObservation] = []
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

        writer.write(process_frame)
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
        "frames_processed": frames_processed,
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
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config, e.g. bytetrack.yaml or botsort.yaml.")
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
        tracker=args.tracker,
    )
    print(f"video: {args.video}")
    print(f"profile: {args.profile}")
    print(f"frames_processed: {output['frames_processed']}")
    print(f"elapsed_sec: {output['elapsed_sec']:.1f}")
    if output["processed_fps"] is not None:
        print(f"processed_fps: {output['processed_fps']:.2f}")
    print(f"track_count: {output['track_count']}")
    print(f"processed_size: {output['video']['processed_width_px']} x {output['video']['processed_height_px']}")
    print(f"tracks: {args.out / 'tracks.json'}")
    print(f"preview: {args.out / output['preview_video']}")


if __name__ == "__main__":
    main()
