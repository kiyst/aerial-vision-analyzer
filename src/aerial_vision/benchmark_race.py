from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from aerial_vision.race_sim import RaceConfig, run_race_simulation


SCENARIOS = ("swerve", "sharp_turns", "fast_break")


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one number.")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All values must be positive.")
    return values


def parse_str_list(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one value.")
    return values


def parse_bool_modes(value: str) -> list[bool]:
    modes: list[bool] = []
    for part in value.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item in {"center", "follow", "false", "0", "off"}:
            modes.append(False)
        elif item in {"tag", "true", "1", "on"}:
            modes.append(True)
        else:
            raise argparse.ArgumentTypeError(f"Unknown mode: {part}")
    if not modes:
        raise argparse.ArgumentTypeError("Expected at least one mode.")
    return modes


def race_score(summary: dict[str, object], *, tag_mode: bool) -> float:
    visible_ratio = float(summary["visible_ratio"])
    average_error = summary["average_center_error"]
    error_penalty = float(average_error) if average_error is not None else 2.0
    lost_ratio = float(summary["lost_frames"]) / max(1.0, float(summary["frames"]))
    final_distance = summary["final_distance_m"]
    distance_score = 0.0
    if tag_mode and final_distance is not None:
        distance_score = max(0.0, 1.0 - float(final_distance) / 40.0)
    return max(0.0, visible_ratio * 100.0 - error_penalty * 35.0 - lost_ratio * 80.0 + distance_score * 20.0)


def summarize_run(report: dict[str, object], *, tag_mode: bool) -> dict[str, object]:
    summary = report["summary"]
    return {
        "frames": summary["frames"],
        "visible_ratio": summary["visible_ratio"],
        "lost_frames": summary["lost_frames"],
        "average_center_error": summary["average_center_error"],
        "final_distance_m": summary["final_distance_m"],
        "mode_counts": summary["mode_counts"],
        "score": race_score(summary, tag_mode=tag_mode),
    }


def aggregate_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    if not summaries:
        return {
            "score": 0.0,
            "visible_ratio": 0.0,
            "average_center_error": None,
            "lost_frames": 0,
            "trials": 0,
        }
    scores = [float(summary["score"]) for summary in summaries]
    frames = [int(summary["frames"]) for summary in summaries]
    visible = [float(summary["visible_ratio"]) for summary in summaries]
    errors = [
        float(summary["average_center_error"])
        for summary in summaries
        if summary["average_center_error"] is not None
    ]
    lost = [int(summary["lost_frames"]) for summary in summaries]
    distance = [
        float(summary["final_distance_m"])
        for summary in summaries
        if summary["final_distance_m"] is not None
    ]
    return {
        "score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "frames": statistics.mean(frames),
        "visible_ratio": statistics.mean(visible),
        "min_visible_ratio": min(visible),
        "average_center_error": statistics.mean(errors) if errors else None,
        "lost_frames": statistics.mean(lost),
        "final_distance_m": statistics.mean(distance) if distance else None,
        "trials": len(summaries),
        "trial_summaries": summaries,
    }


def benchmark_race(
    *,
    out_dir: Path,
    scenarios: list[str],
    durations: list[float],
    target_speeds: list[float],
    horizontal_fovs: list[float],
    tag_modes: list[bool],
    trials: int,
    dt_sec: float,
    seed: int,
) -> dict[str, object]:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    started = time.perf_counter()
    run_index = 0

    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        for duration in durations:
            for speed in target_speeds:
                for fov in horizontal_fovs:
                    for tag_mode in tag_modes:
                        run_index += 1
                        trial_summaries: list[dict[str, object]] = []
                        for trial in range(trials):
                            config = RaceConfig(
                                duration_sec=duration,
                                dt_sec=dt_sec,
                                seed=seed + trial,
                                scenario=scenario,
                                tag_mode=tag_mode,
                                target_speed_mps=speed,
                                horizontal_fov_deg=fov,
                                vertical_fov_deg=max(35.0, fov * 0.65),
                            )
                            report = run_race_simulation(config)
                            trial_summaries.append(summarize_run(report, tag_mode=tag_mode))
                        summary = aggregate_summaries(trial_summaries)
                        runs.append(
                            {
                                "scenario": scenario,
                                "duration_sec": duration,
                                "target_speed_mps": speed,
                                "horizontal_fov_deg": fov,
                                "tag_mode": tag_mode,
                                "summary": summary,
                            }
                        )

    best = max(runs, key=lambda run: float(run["summary"]["score"])) if runs else None
    report = {
        "elapsed_sec": time.perf_counter() - started,
        "trials": trials,
        "dt_sec": dt_sec,
        "seed": seed,
        "best": best,
        "runs": runs,
    }
    (out_dir / "race_benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark closed-loop race/tag simulator settings.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", type=parse_str_list, default=parse_str_list("swerve,sharp_turns,fast_break"))
    parser.add_argument("--durations", type=parse_float_list, default=parse_float_list("120"))
    parser.add_argument("--target-speeds", type=parse_float_list, default=parse_float_list("3,6,10"))
    parser.add_argument("--horizontal-fovs", type=parse_float_list, default=parse_float_list("45,70,110"))
    parser.add_argument("--modes", type=parse_bool_modes, default=parse_bool_modes("center,tag"))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--dt-sec", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=11)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = benchmark_race(
        out_dir=args.out,
        scenarios=args.scenarios,
        durations=args.durations,
        target_speeds=args.target_speeds,
        horizontal_fovs=args.horizontal_fovs,
        tag_modes=args.modes,
        trials=args.trials,
        dt_sec=args.dt_sec,
        seed=args.seed,
    )
    best = report["best"]
    print(f"benchmark: {args.out / 'race_benchmark.json'}")
    if best is not None:
        summary = best["summary"]
        mode = "tag" if best["tag_mode"] else "center"
        print(
            "best: "
            f"scenario={best['scenario']} "
            f"mode={mode} "
            f"speed={best['target_speed_mps']} "
            f"fov={best['horizontal_fov_deg']} "
            f"visible={summary['visible_ratio']:.2%} "
            f"error={summary['average_center_error']:.3f} "
            f"score={summary['score']:.2f}"
        )


if __name__ == "__main__":
    main()
