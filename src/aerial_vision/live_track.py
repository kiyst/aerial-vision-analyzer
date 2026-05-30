from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from aerial_vision.detection import BoundingBox, normalize_label
from aerial_vision.live_source import playback_status, realtime_delay_sec, should_drop_frame
from aerial_vision.pipeline import PROFILE_CONFIGS
from aerial_vision.track_video import (
    ControlIntent,
    TargetLockObservation,
    TrackObservation,
    class_ids_for_model,
    clamp_box,
    control_intent_from_lock,
    draw_control_intent,
    draw_target_lock,
    observation_from_box,
    optical_flow_observation,
    profile_min_confidence,
    profile_model_and_classes,
    select_target_observation,
    target_lock_state,
    yolo_track_kwargs,
)


def prepare_live_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def resolve_label_filter(available_labels: set[str], requested_labels: list[str] | None) -> set[str]:
    if not requested_labels:
        return available_labels
    normalized = {normalize_label(label) for label in requested_labels}
    if "all" in normalized:
        return available_labels
    unknown = sorted(normalized - available_labels)
    if unknown:
        raise ValueError(f"Unknown label(s): {', '.join(unknown)}. Available labels: {', '.join(sorted(available_labels))}.")
    return normalized


def prompt_label_filter(available_labels: set[str]) -> list[str]:
    labels = sorted(available_labels)
    print("Selectable labels:")
    print("  all")
    for label in labels:
        print(f"  {label}")
    response = input("Track which labels? Type all or comma-separated labels: ").strip()
    if not response:
        return ["all"]
    return [part.strip() for part in response.replace(",", " ").split() if part.strip()]


def observation_contains_point(observation: TrackObservation, x: float, y: float) -> bool:
    return observation.box.x_min <= x <= observation.box.x_max and observation.box.y_min <= y <= observation.box.y_max


def select_observation_at_point(observations: list[TrackObservation], x: float, y: float) -> TrackObservation | None:
    candidates = [observation for observation in observations if observation_contains_point(observation, x, y)]
    if not candidates:
        return None
    return min(candidates, key=lambda observation: observation.area_px)


def draw_selectable_observation(frame: object, observation: TrackObservation, *, selected: bool) -> None:
    import cv2

    color = (0, 255, 0) if selected else (0, 255, 255)
    thickness = 3 if selected else 2
    x_min, y_min, x_max, y_max = clamp_box(observation.box, width=frame.shape[1], height=frame.shape[0])
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
    label = f"#{observation.track_id} {observation.label} {observation.confidence:.2f}"
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_width = min(frame.shape[1] - x_min, text_size[0] + 8)
    cv2.rectangle(frame, (x_min, max(0, y_min - 22)), (x_min + label_width, y_min), color, -1)
    cv2.putText(frame, label, (x_min + 4, max(14, y_min - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def live_track(
    source: str,
    *,
    profile: str,
    out_dir: Path,
    target_id: int | None = None,
    target_label: str | None = None,
    detect_every: int = 10,
    resize_width: int | None = 960,
    max_frames: int | None = None,
    realtime: bool = True,
    max_latency_sec: float = 0.5,
    display: bool = False,
    interactive_select: bool = False,
    select_labels: list[str] | None = None,
    prompt_labels: bool = True,
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
) -> dict[str, object]:
    if detect_every <= 0:
        raise ValueError("detect_every must be positive.")
    if resize_width is not None and resize_width <= 0:
        raise ValueError("resize_width must be positive.")
    if max_latency_sec < 0:
        raise ValueError("max_latency_sec cannot be negative.")
    if imgsz is not None and imgsz <= 0:
        raise ValueError("imgsz must be positive.")
    if max_det <= 0:
        raise ValueError("max_det must be positive.")
    if target_id is None and target_label is None and not interactive_select:
        raise ValueError("target_id, target_label, or interactive_select is required.")
    if interactive_select:
        display = True

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Live tracking requires OpenCV and Ultralytics. Install with: python3 -m pip install -e '.[ml]'") from exc

    capture = cv2.VideoCapture(parse_source(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("Source dimensions could not be determined.")

    process_width = source_width
    process_height = source_height
    if resize_width is not None and resize_width < source_width:
        process_width = resize_width
        process_height = int(round(source_height * (resize_width / source_width)))

    prepare_live_dir(out_dir)
    model_path, wanted_classes = profile_model_and_classes(profile)
    if interactive_select and select_labels is None and prompt_labels:
        select_labels = prompt_label_filter(wanted_classes)
    wanted_classes = resolve_label_filter(wanted_classes, select_labels)
    min_confidence = profile_min_confidence(profile)
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
    target_lock_observations: list[TargetLockObservation] = []
    control_intents: list[ControlIntent] = []
    last_target_observation: TrackObservation | None = None
    last_target_offset: tuple[float, float] | None = None
    last_target_box_area: float | None = None
    active_target_id = target_id
    tag_mode = False
    latest_detector_observations: list[TrackObservation] = []
    target_switches: list[dict[str, object]] = []
    pending_click: tuple[int, int] | None = None
    selection_mode = interactive_select and active_target_id is None and target_label is None
    force_detector_refresh = False
    previous_gray = None
    processed_frames = 0
    dropped_frames = 0
    detector_frames = 0
    optical_flow_frames = 0
    frame_index = 0
    start_time = time.perf_counter()
    window_name = "aerial_vision live_track"

    if display:
        cv2.namedWindow(window_name)

        if interactive_select:
            def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
                del flags, param
                nonlocal pending_click, force_detector_refresh
                if event == cv2.EVENT_LBUTTONDOWN:
                    pending_click = (x, y)
                    force_detector_refresh = True

            cv2.setMouseCallback(window_name, on_mouse)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if max_frames is not None and processed_frames >= max_frames:
            break

        wall_elapsed = time.perf_counter() - start_time
        video_time = frame_index / fps
        pace_realtime = realtime and not interactive_select
        if pace_realtime:
            delay = realtime_delay_sec(video_time_sec=video_time, wall_elapsed_sec=wall_elapsed)
            if delay > 0:
                time.sleep(delay)
                wall_elapsed = time.perf_counter() - start_time
            if should_drop_frame(
                video_time_sec=video_time,
                wall_elapsed_sec=wall_elapsed,
                max_latency_sec=max_latency_sec,
            ):
                dropped_frames += 1
                frame_index += 1
                continue

        process_frame = frame
        if process_width != source_width or process_height != source_height:
            process_frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
        current_gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)

        frame_observations: list[TrackObservation] = []
        use_detector = selection_mode or force_detector_refresh or processed_frames % detect_every == 0
        if use_detector:
            force_detector_refresh = False
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
                    observation = observation_from_box(
                        frame=process_frame,
                        frame_index=frame_index,
                        timestamp_sec=video_time,
                        track_id=track_id,
                        label=label,
                        confidence=confidence,
                        box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                        previous=previous_by_track.get(track_id),
                    )
                    previous_by_track[track_id] = observation
                    frame_observations.append(observation)
                    if active_target_id is not None and track_id == active_target_id:
                        last_target_observation = observation

            latest_detector_observations = frame_observations.copy()
            if active_target_id is None and target_label is not None:
                selected = select_target_observation(frame_observations, target_label)
                if selected is not None:
                    active_target_id = selected.track_id
                    last_target_observation = selected
                    target_switches.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_sec": video_time,
                            "target_id": active_target_id,
                            "label": selected.label,
                            "confidence": selected.confidence,
                            "method": "target_label",
                        }
                    )

            if pending_click is not None:
                click_x, click_y = pending_click
                pending_click = None
                selected = select_observation_at_point(frame_observations, click_x, click_y)
                if selected is not None:
                    active_target_id = selected.track_id
                    last_target_observation = selected
                    previous_by_track[active_target_id] = selected
                    selection_mode = False
                    target_switches.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_sec": video_time,
                            "target_id": active_target_id,
                            "label": selected.label,
                            "confidence": selected.confidence,
                            "click": [click_x, click_y],
                            "method": "click",
                        }
                    )
        elif previous_gray is not None and last_target_observation is not None:
            optical_flow_frames += 1
            optical_observation = optical_flow_observation(
                previous_gray=previous_gray,
                current_gray=current_gray,
                current_frame=process_frame,
                previous_observation=last_target_observation,
                frame_index=frame_index,
                timestamp_sec=video_time,
            )
            if optical_observation is not None:
                last_target_observation = optical_observation
                if active_target_id is not None:
                    previous_by_track[active_target_id] = optical_observation
                frame_observations.append(optical_observation)

        if active_target_id is not None:
            target = next((observation for observation in frame_observations if observation.track_id == active_target_id), None)
            lock_observation = target_lock_state(
                target,
                frame_observations,
                target_id=active_target_id,
                frame_index=frame_index,
                timestamp_sec=video_time,
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

        if display:
            if interactive_select and selection_mode:
                for observation in latest_detector_observations:
                    draw_selectable_observation(
                        process_frame,
                        observation,
                        selected=active_target_id is not None and observation.track_id == active_target_id,
                    )
            elapsed_for_display = max(time.perf_counter() - start_time, 1e-6)
            output_fps = (processed_frames + 1) / elapsed_for_display
            status = playback_status(
                frame_index=frame_index,
                fps=fps,
                wall_elapsed_sec=time.perf_counter() - start_time,
                processed_frames=processed_frames + 1,
                dropped_frames=dropped_frames,
            )
            cv2.putText(
                process_frame,
                f"fps={output_fps:.2f} mode={'select' if selection_mode else 'track'} drop={dropped_frames}",
                (20, process_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            if interactive_select:
                instruction = "click target | t tag | space boxes | c clear | q quit"
                if active_target_id is not None:
                    tag_text = "TAG ON" if tag_mode else "tag off"
                    instruction = f"target #{active_target_id} {tag_text} | " + instruction
                cv2.putText(
                    process_frame,
                    instruction,
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
            cv2.imshow(window_name, process_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if interactive_select and key == ord(" "):
                force_detector_refresh = True
                selection_mode = True
            if interactive_select and key == ord("c"):
                active_target_id = None
                last_target_observation = None
                last_target_offset = None
                last_target_box_area = None
                tag_mode = False
                selection_mode = True
            if interactive_select and key == ord("t") and active_target_id is not None:
                tag_mode = not tag_mode
                target_switches.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": video_time,
                        "target_id": active_target_id,
                        "method": "tag_toggle",
                        "tag_mode": tag_mode,
                    }
                )

        previous_gray = current_gray
        processed_frames += 1
        frame_index += 1

    capture.release()
    if display:
        cv2.destroyAllWindows()

    elapsed_sec = time.perf_counter() - start_time
    state_counts: dict[str, int] = {}
    for observation in target_lock_observations:
        state_counts[observation.state] = state_counts.get(observation.state, 0) + 1
    status = playback_status(
        frame_index=max(frame_index - 1, 0),
        fps=fps,
        wall_elapsed_sec=elapsed_sec,
        processed_frames=processed_frames,
        dropped_frames=dropped_frames,
    )
    output = {
        "source": source,
        "profile": profile,
        "target_id": active_target_id,
        "requested_target_id": target_id,
        "target_label": target_label,
        "select_labels": sorted(wanted_classes),
        "interactive_select": interactive_select,
        "target_switches": target_switches,
        "tag_mode": tag_mode,
        "detect_every": detect_every,
        "resize_width": resize_width,
        "imgsz": imgsz,
        "max_det": max_det,
        "device": device,
        "half": half,
        "tracker": tracker,
        "realtime": realtime,
        "max_latency_sec": max_latency_sec,
        "source_width_px": source_width,
        "source_height_px": source_height,
        "processed_width_px": process_width,
        "processed_height_px": process_height,
        "fps": fps,
        "processed_frames": processed_frames,
        "dropped_frames": dropped_frames,
        "detector_frames": detector_frames,
        "optical_flow_frames": optical_flow_frames,
        "elapsed_sec": elapsed_sec,
        "processed_fps": processed_frames / elapsed_sec if elapsed_sec > 0 else None,
        "playback_status": status.to_dict(),
        "state_counts": state_counts,
        "observations": [observation.to_dict() for observation in target_lock_observations],
        "control_intents": [intent.to_dict() for intent in control_intents],
    }
    (out_dir / "live_session.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live-like target tracking from a video file or camera source.")
    parser.add_argument("source", help="Video path, camera index, or stream URL.")
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="vehicles", help="Detection profile.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for live session JSON.")
    parser.add_argument("--target-id", type=int, help="Track ID to lock onto.")
    parser.add_argument("--target-label", help="Automatically lock onto the highest-confidence track with this label, e.g. truck.")
    parser.add_argument("--detect-every", type=int, default=10, help="Detector refresh interval in processed frames.")
    parser.add_argument("--resize-width", type=int, default=960, help="Resize source to this width before tracking.")
    parser.add_argument("--max-frames", type=int, help="Optional frame limit for tests.")
    parser.add_argument("--max-latency-sec", type=float, default=0.5, help="Drop frames when realtime processing falls behind by this much.")
    parser.add_argument("--no-realtime", action="store_true", help="Process as fast as possible instead of pacing to source FPS.")
    parser.add_argument("--display", action="store_true", help="Show a live OpenCV preview window. Press q to quit.")
    parser.add_argument("--interactive-select", action="store_true", help="Show a preview where clicking a detected box selects or switches the active target.")
    parser.add_argument("--select-labels", nargs="+", help="Limit selectable/detected labels, e.g. truck bus, or all. Interactive mode prompts when omitted.")
    parser.add_argument("--no-label-prompt", action="store_true", help="Do not prompt for labels in interactive mode; use the full profile.")
    parser.add_argument("--imgsz", type=int, help="YOLO inference image size. Smaller values can improve speed.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per detector refresh.")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu, mps, 0, cuda:0.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on supported GPU devices.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--max-intent-yaw-rate-deg-s", type=float, default=35.0, help="Maximum yaw-rate intent written to the live session.")
    parser.add_argument("--max-intent-pitch-rate-deg-s", type=float, default=25.0, help="Maximum camera pitch-rate intent written to the live session.")
    parser.add_argument("--max-intent-forward-mps", type=float, default=6.0, help="Maximum tag-mode forward-speed intent.")
    parser.add_argument("--follow-intent-forward-mps", type=float, default=2.0, help="Forward-speed intent used to keep a non-tag target framed when it appears to move away.")
    parser.add_argument("--intent-center-deadband", type=float, default=0.08, help="Normalized center error below which the target is considered centered.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = live_track(
        args.source,
        profile=args.profile,
        out_dir=args.out,
        target_id=args.target_id,
        target_label=args.target_label,
        detect_every=args.detect_every,
        resize_width=args.resize_width,
        max_frames=args.max_frames,
        realtime=not args.no_realtime,
        max_latency_sec=args.max_latency_sec,
        display=args.display,
        interactive_select=args.interactive_select,
        select_labels=args.select_labels,
        prompt_labels=not args.no_label_prompt,
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
    )
    print(f"source: {args.source}")
    print(f"target_id: {output['target_id']}")
    print(f"processed_frames: {output['processed_frames']}")
    print(f"dropped_frames: {output['dropped_frames']}")
    print(f"detector_frames: {output['detector_frames']}")
    print(f"optical_flow_frames: {output['optical_flow_frames']}")
    print(f"processed_fps: {output['processed_fps']:.2f}" if output["processed_fps"] else "processed_fps: n/a")
    print(f"state_counts: {output['state_counts']}")
    if output["control_intents"]:
        print(f"control_intents: {len(output['control_intents'])}")
    print(f"session: {args.out / 'live_session.json'}")


if __name__ == "__main__":
    main()
