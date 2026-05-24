from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerial_vision.analysis import ImageAnalysis
from aerial_vision.annotate import draw_detections
from aerial_vision.pipeline import DetectionSettings, PROFILE_CONFIGS, analyze_image


def clean_previous_outputs(output_paths: list[Path]) -> None:
    output_dirs = {path.parent for path in output_paths if path.parent.name == "outputs"}

    for output_dir in output_dirs:
        if not output_dir.exists():
            continue

        for output_file in output_dir.iterdir():
            if output_file.is_file():
                output_file.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect objects of interest in an aerial image.")
    parser.add_argument("image", type=Path, help="Path to a plain image file.")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_CONFIGS),
        help="Use a preset detection workflow. Explicit flags override profile defaults.",
    )
    parser.add_argument(
        "--backend",
        choices=("yolo", "rfdetr"),
        default="yolo",
        help="Detector backend to use.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO model path/name, or RF-DETR checkpoint path with --backend rfdetr.",
    )
    parser.add_argument(
        "--rfdetr-size",
        choices=("nano", "small", "medium", "large"),
        default="medium",
        help="RF-DETR model size. Used with --backend rfdetr.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.25, help="Minimum confidence from 0 to 1.")
    parser.add_argument("--objects", nargs="*", help="Optional object labels to keep.")
    parser.add_argument(
        "--extra-model",
        help="Optional second YOLO model path/name to run and merge into the same output.",
    )
    parser.add_argument(
        "--extra-objects",
        nargs="*",
        help="Object labels to keep from --extra-model. Useful for animals.",
    )
    parser.add_argument(
        "--extra-min-confidence",
        type=float,
        help="Minimum confidence for --extra-model. Defaults to --min-confidence.",
    )
    parser.add_argument("--tile-size", type=int, help="Run tiled inference with this tile size in pixels.")
    parser.add_argument("--tile-overlap", type=int, default=128, help="Tile overlap in pixels. Used with --tile-size.")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="Duplicate suppression IoU threshold for tiled runs.")
    parser.add_argument("--max-area-ratio", type=float, help="Drop boxes larger than this fraction of the image area.")
    parser.add_argument("--max-width-ratio", type=float, help="Drop boxes wider than this fraction of the image width.")
    parser.add_argument("--max-height-ratio", type=float, help="Drop boxes taller than this fraction of the image height.")
    parser.add_argument("--min-area-px", type=float, help="Drop boxes smaller than this pixel area.")
    parser.add_argument(
        "--geometry-preset",
        choices=("overhead-vehicles",),
        help="Apply tuned per-label geometry filters.",
    )
    parser.add_argument(
        "--collapse-animals",
        action="store_true",
        help="Rename animal proxy labels like bird/horse/cow/sheep/dog/cat to animal.",
    )
    parser.add_argument("--json-out", type=Path, help="Write full analysis JSON to this file.")
    parser.add_argument("--annotated-out", type=Path, help="Write an annotated image to this path.")
    return parser


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.profile:
        return args

    profile = PROFILE_CONFIGS[args.profile]
    parser_defaults = {
        "model": "yolo11n.pt",
        "min_confidence": 0.25,
        "objects": None,
        "tile_size": None,
        "tile_overlap": 128,
        "geometry_preset": None,
        "extra_model": None,
        "extra_objects": None,
        "extra_min_confidence": None,
        "collapse_animals": False,
    }

    for key, value in profile.items():
        if getattr(args, key) == parser_defaults[key]:
            setattr(args, key, value)

    return args


def format_summary(analysis: ImageAnalysis) -> str:
    lines = [
        f"image: {analysis.image_path}",
        f"size: {analysis.image_width} x {analysis.image_height} px",
        f"detections: {analysis.total_detections}",
    ]

    if analysis.average_confidence is not None:
        lines.append(f"average_confidence: {analysis.average_confidence:.3f}")

    if analysis.counts_by_label:
        counts = ", ".join(f"{label}={count}" for label, count in analysis.counts_by_label.items())
        lines.append(f"counts: {counts}")

    return "\n".join(lines)


def main() -> None:
    args = apply_profile_defaults(build_parser().parse_args())
    output_paths = [path for path in (args.json_out, args.annotated_out) if path is not None]
    clean_previous_outputs(output_paths)

    settings = DetectionSettings(
        backend=args.backend,
        model=args.model,
        rfdetr_size=args.rfdetr_size,
        min_confidence=args.min_confidence,
        objects=args.objects,
        extra_model=args.extra_model,
        extra_objects=args.extra_objects,
        extra_min_confidence=args.extra_min_confidence,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        nms_iou=args.nms_iou,
        max_area_ratio=args.max_area_ratio,
        max_width_ratio=args.max_width_ratio,
        max_height_ratio=args.max_height_ratio,
        min_area_px=args.min_area_px,
        geometry_preset=args.geometry_preset,
        collapse_animals=args.collapse_animals,
    )
    analysis = analyze_image(args.image, settings)
    output = json.dumps(analysis.to_dict(), indent=2)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    if args.annotated_out:
        args.annotated_out.parent.mkdir(parents=True, exist_ok=True)
        draw_detections(args.image, analysis.detections, args.annotated_out)

    if args.json_out or args.annotated_out:
        print(format_summary(analysis))


if __name__ == "__main__":
    main()
