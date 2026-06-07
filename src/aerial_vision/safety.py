from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from aerial_vision.control_sim import clamp
from aerial_vision.track_video import ControlIntent


@dataclass(frozen=True)
class SafetyLimits:
    min_visible_ratio: float = 0.95
    max_lost_ratio: float = 0.02
    max_average_center_error: float = 0.35
    min_required_pass_ratio: float = 0.80
    max_yaw_rate_deg_s: float = 45.0
    max_pitch_rate_deg_s: float = 35.0
    max_forward_mps: float = 6.0
    max_tag_center_error: float = 0.45


def safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def safety_filter_intent(intent: ControlIntent, limits: SafetyLimits = SafetyLimits()) -> ControlIntent:
    yaw = clamp(intent.yaw_rate_deg_s, -limits.max_yaw_rate_deg_s, limits.max_yaw_rate_deg_s)
    pitch = clamp(intent.camera_pitch_rate_deg_s, -limits.max_pitch_rate_deg_s, limits.max_pitch_rate_deg_s)
    forward = clamp(intent.forward_mps, 0.0, limits.max_forward_mps)
    mode = intent.mode
    reason = intent.reason

    if mode in {"center", "centered", "hold", "search"}:
        forward = 0.0
    if mode == "tag" and intent.center_error is not None and intent.center_error > limits.max_tag_center_error:
        forward = 0.0
        reason = f"{reason}; safety gate blocked tag forward motion because target is off-center"

    return ControlIntent(
        frame_index=intent.frame_index,
        timestamp_sec=intent.timestamp_sec,
        mode=mode,
        reason=reason,
        yaw_rate_deg_s=yaw,
        camera_pitch_rate_deg_s=pitch,
        forward_mps=forward,
        normalized_offset=intent.normalized_offset,
        center_error=intent.center_error,
        tag_enabled=intent.tag_enabled,
        box_area_ratio=intent.box_area_ratio,
    )


def evaluate_race_summary(summary: dict[str, object], limits: SafetyLimits = SafetyLimits()) -> dict[str, object]:
    frames = max(1.0, safe_float(summary.get("frames"), 1.0))
    visible_ratio = safe_float(summary.get("visible_ratio"))
    lost_ratio = safe_float(summary.get("lost_frames")) / frames
    average_error = summary.get("average_center_error")
    average_error_value = None if average_error is None else float(average_error)

    checks = {
        "visible_ratio": visible_ratio >= limits.min_visible_ratio,
        "lost_ratio": lost_ratio <= limits.max_lost_ratio,
        "average_center_error": average_error_value is not None and average_error_value <= limits.max_average_center_error,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "visible_ratio": visible_ratio,
            "lost_ratio": lost_ratio,
            "average_center_error": average_error_value,
        },
        "limits": asdict(limits),
    }


def evaluate_race_benchmark(report: dict[str, object], limits: SafetyLimits = SafetyLimits()) -> dict[str, object]:
    runs = report.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    evaluated_runs: list[dict[str, object]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        summary = run.get("summary", {})
        if not isinstance(summary, dict):
            continue
        summary_for_eval = dict(summary)
        if "frames" not in summary_for_eval:
            trial_summaries = summary_for_eval.get("trial_summaries", [])
            if isinstance(trial_summaries, list) and trial_summaries:
                first_trial = trial_summaries[0]
                if isinstance(first_trial, dict) and "frames" in first_trial:
                    summary_for_eval["frames"] = first_trial["frames"]
        result = evaluate_race_summary(summary_for_eval, limits)
        evaluated_runs.append(
            {
                "scenario": run.get("scenario"),
                "tag_mode": run.get("tag_mode"),
                "target_speed_mps": run.get("target_speed_mps"),
                "horizontal_fov_deg": run.get("horizontal_fov_deg"),
                "evaluation": result,
            }
        )

    passed_count = sum(1 for run in evaluated_runs if run["evaluation"]["passed"])
    pass_ratio = passed_count / len(evaluated_runs) if evaluated_runs else 0.0
    return {
        "passed": pass_ratio >= limits.min_required_pass_ratio,
        "pass_ratio": pass_ratio,
        "passed_runs": passed_count,
        "total_runs": len(evaluated_runs),
        "limits": asdict(limits),
        "runs": evaluated_runs,
    }


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate simulator reports against pre-autopilot safety gates.")
    parser.add_argument("report", type=Path, help="Path to race_benchmark.json or race_report.json.")
    parser.add_argument("--out", type=Path, help="Optional path for safety_report.json.")
    parser.add_argument("--min-visible-ratio", type=float, default=0.95)
    parser.add_argument("--max-lost-ratio", type=float, default=0.02)
    parser.add_argument("--max-average-center-error", type=float, default=0.35)
    parser.add_argument("--min-required-pass-ratio", type=float, default=0.80)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    limits = SafetyLimits(
        min_visible_ratio=args.min_visible_ratio,
        max_lost_ratio=args.max_lost_ratio,
        max_average_center_error=args.max_average_center_error,
        min_required_pass_ratio=args.min_required_pass_ratio,
    )
    payload = load_json(args.report)
    result = (
        evaluate_race_benchmark(payload, limits)
        if "runs" in payload
        else evaluate_race_summary(payload.get("summary", {}), limits)
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"safety_report: {args.out}")
    print(f"passed: {result['passed']}")
    if "pass_ratio" in result:
        print(f"pass_ratio: {result['pass_ratio']:.2%}")
        print(f"passed_runs: {result['passed_runs']} / {result['total_runs']}")


if __name__ == "__main__":
    main()
