from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def safe_run_path(run_dir: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None

    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe run-relative path: {relative_path}")

    resolved_run_dir = run_dir.resolve()
    resolved_candidate = (run_dir / candidate).resolve()
    if resolved_run_dir != resolved_candidate and resolved_run_dir not in resolved_candidate.parents:
        raise ValueError(f"Path escapes run directory: {relative_path}")

    return resolved_candidate


def load_decisions(decisions_path: Path) -> dict[str, object]:
    with decisions_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("review decisions must include an events list.")

    return payload


def confirmed_events(decisions: dict[str, object]) -> list[dict[str, object]]:
    return [
        event
        for event in decisions["events"]
        if isinstance(event, dict) and event.get("status") == "confirmed"
    ]


def prepare_export_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "detections").mkdir(parents=True, exist_ok=True)


def copy_if_present(source: Path | None, destination_dir: Path) -> str | None:
    if source is None:
        return None
    if not source.exists():
        raise FileNotFoundError(f"Missing referenced file: {source}")

    destination = destination_dir / source.name
    shutil.copy2(source, destination)
    return str(destination.relative_to(destination_dir.parent))


def export_confirmed(decisions_path: Path, *, run_dir: Path | None = None, out_dir: Path | None = None) -> dict[str, object]:
    decisions = load_decisions(decisions_path)
    resolved_run_dir = run_dir if run_dir is not None else decisions_path.parent
    resolved_out_dir = out_dir if out_dir is not None else resolved_run_dir / "confirmed_export"
    confirmed = confirmed_events(decisions)

    prepare_export_dir(resolved_out_dir)

    exported_events: list[dict[str, object]] = []
    for event in confirmed:
        annotated_source = safe_run_path(resolved_run_dir, event.get("annotated_image"))
        detections_source = safe_run_path(resolved_run_dir, event.get("detections_json"))
        exported_event = dict(event)
        exported_event["exported_annotated_image"] = copy_if_present(
            annotated_source,
            resolved_out_dir / "images",
        )
        exported_event["exported_detections_json"] = copy_if_present(
            detections_source,
            resolved_out_dir / "detections",
        )
        exported_events.append(exported_event)

    output = {
        "source_review_decisions": str(decisions_path),
        "run_dir": str(resolved_run_dir),
        "confirmed_count": len(exported_events),
        "video": decisions.get("video"),
        "profile": decisions.get("profile"),
        "events": exported_events,
    }
    (resolved_out_dir / "confirmed_events.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export confirmed events from a review_decisions.json file.")
    parser.add_argument("decisions", type=Path, help="Path to review_decisions.json exported from review.html.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Field run directory containing positives/ and detections/. Defaults to the decisions file parent.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory for confirmed-only export. Defaults to <run-dir>/confirmed_export.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = export_confirmed(args.decisions, run_dir=args.run_dir, out_dir=args.out)
    print(f"confirmed_count: {output['confirmed_count']}")
    print(f"export: {args.out or (args.run_dir or args.decisions.parent) / 'confirmed_export'}")


if __name__ == "__main__":
    main()
