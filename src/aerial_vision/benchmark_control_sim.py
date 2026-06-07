from __future__ import annotations

import argparse
import html
import json
import statistics
import time
from pathlib import Path

from aerial_vision.control_sim import SCENARIOS, SimConfig, run_simulation


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
    unknown = [item for item in values if item not in SCENARIOS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown scenario(s): {', '.join(unknown)}")
    return values


def lost_ratio(summary: dict[str, object]) -> float:
    return float(summary["lost_frames"]) / max(1.0, float(summary["steps"]))


def false_redetect_ratio(summary: dict[str, object]) -> float:
    return float(summary["false_redetect_frames"]) / max(1.0, float(summary["steps"]))


def control_sim_score(summary: dict[str, object]) -> float:
    visible_ratio = float(summary["visible_ratio"])
    error = summary["average_screen_error"]
    error_penalty = float(error) if error is not None else 2.0
    lost_penalty = lost_ratio(summary)
    false_penalty = false_redetect_ratio(summary)
    reacquire_penalty = min(1.0, float(summary.get("longest_lost_streak_sec", 0.0)) / 3.0)
    smoothness_penalty = min(1.0, float(summary.get("average_yaw_rate_change_deg_s", 0.0)) / 80.0)
    return max(
        0.0,
        visible_ratio * 100.0
        - error_penalty * 35.0
        - lost_penalty * 80.0
        - false_penalty * 100.0
        - reacquire_penalty * 20.0
        - smoothness_penalty * 5.0,
    )


def control_sim_run_passed(
    summary: dict[str, object],
    *,
    min_visible_ratio: float = 0.95,
    max_lost_ratio: float = 0.02,
    max_average_screen_error: float = 0.35,
    max_longest_lost_sec: float = 1.5,
    max_false_redetect_ratio: float = 0.0,
) -> tuple[bool, dict[str, bool]]:
    average_error = summary["average_screen_error"]
    average_error_ok = average_error is not None and float(average_error) <= max_average_screen_error
    checks = {
        "visible_ratio": float(summary["visible_ratio"]) >= min_visible_ratio,
        "lost_ratio": lost_ratio(summary) <= max_lost_ratio,
        "average_screen_error": average_error_ok,
        "longest_lost_streak_sec": float(summary.get("longest_lost_streak_sec", 0.0)) <= max_longest_lost_sec,
        "false_redetect_ratio": false_redetect_ratio(summary) <= max_false_redetect_ratio,
    }
    return all(checks.values()), checks


def summarize_run(
    report: dict[str, object],
    *,
    scenario: str,
    speed: float,
    fov: float,
    latency_ms: float,
    seed: int,
) -> dict[str, object]:
    summary = report["summary"]
    passed, checks = control_sim_run_passed(summary)
    return {
        "scenario": scenario,
        "target_speed_mps": speed,
        "horizontal_fov_deg": fov,
        "latency_ms": latency_ms,
        "seed": seed,
        "passed": passed,
        "checks": checks,
        "score": control_sim_score(summary),
        "summary": {
            "steps": summary["steps"],
            "duration_sec": summary["duration_sec"],
            "visible_ratio": summary["visible_ratio"],
            "lost_frames": summary["lost_frames"],
            "lost_ratio": lost_ratio(summary),
            "average_screen_error": summary["average_screen_error"],
            "max_screen_error": summary["max_screen_error"],
            "longest_lost_streak_sec": summary["longest_lost_streak_sec"],
            "average_reacquisition_time_sec": summary["average_reacquisition_time_sec"],
            "false_redetect_frames": summary["false_redetect_frames"],
            "false_redetect_ratio": false_redetect_ratio(summary),
            "average_yaw_rate_change_deg_s": summary["average_yaw_rate_change_deg_s"],
            "grade": summary["grade"],
        },
    }


def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    if not runs:
        return {"passed": False, "pass_ratio": 0.0, "passed_runs": 0, "total_runs": 0}
    scores = [float(run["score"]) for run in runs]
    passed_runs = sum(1 for run in runs if run["passed"])
    visible = [float(run["summary"]["visible_ratio"]) for run in runs]
    errors = [
        float(run["summary"]["average_screen_error"])
        for run in runs
        if run["summary"]["average_screen_error"] is not None
    ]
    lost = [float(run["summary"]["lost_ratio"]) for run in runs]
    longest_lost = [float(run["summary"]["longest_lost_streak_sec"]) for run in runs]
    pass_ratio = passed_runs / len(runs)
    return {
        "passed": pass_ratio >= 0.80,
        "pass_ratio": pass_ratio,
        "passed_runs": passed_runs,
        "total_runs": len(runs),
        "mean_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "min_visible_ratio": min(visible),
        "mean_visible_ratio": statistics.mean(visible),
        "mean_average_screen_error": statistics.mean(errors) if errors else None,
        "max_lost_ratio": max(lost),
        "max_longest_lost_streak_sec": max(longest_lost),
    }


def render_control_benchmark_html(report: dict[str, object]) -> str:
    aggregate = report["aggregate"]
    runs = sorted(
        report["runs"],
        key=lambda run: (bool(run["passed"]), float(run["score"])),
    )
    status = "PASSED" if aggregate["passed"] else "NOT READY"
    rows = "\n".join(render_run_row(run) for run in runs)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Control Simulation Benchmark</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f6f8; color: #1d232f; }}
    header {{ padding: 24px 28px; background: #172033; color: white; }}
    main {{ padding: 24px 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .status {{ display: inline-block; padding: 5px 10px; border-radius: 4px; background: #30415f; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: white; border: 1px solid #d9dee8; border-radius: 6px; padding: 12px; }}
    .metric span {{ display: block; color: #657086; font-size: 12px; margin-bottom: 6px; }}
    .metric strong {{ font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9dee8; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e8ef; text-align: left; font-size: 13px; }}
    th {{ background: #eef1f6; color: #3f4a5f; position: sticky; top: 0; }}
    tr.fail {{ background: #fff1f1; }}
    tr.pass {{ background: #f2fbf5; }}
    code {{ background: #eef1f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Control Simulation Benchmark</h1>
    <div class="status">{html.escape(status)}</div>
  </header>
  <main>
    <section class="grid">
      <div class="metric"><span>Pass ratio</span><strong>{aggregate['pass_ratio']:.1%}</strong></div>
      <div class="metric"><span>Passed runs</span><strong>{aggregate['passed_runs']} / {aggregate['total_runs']}</strong></div>
      <div class="metric"><span>Mean score</span><strong>{aggregate['mean_score']:.1f}</strong></div>
      <div class="metric"><span>Mean screen error</span><strong>{format_optional(aggregate['mean_average_screen_error'])}</strong></div>
      <div class="metric"><span>Worst lost streak</span><strong>{aggregate['max_longest_lost_streak_sec']:.2f}s</strong></div>
    </section>
    <table>
      <thead>
        <tr>
          <th>Status</th><th>Scenario</th><th>Speed</th><th>FOV</th><th>Latency</th>
          <th>Visible</th><th>Error</th><th>Lost</th><th>Longest Lost</th><th>False Re-ID</th><th>Score</th><th>Failed Checks</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def render_run_row(run: dict[str, object]) -> str:
    summary = run["summary"]
    failed = [name for name, passed in run["checks"].items() if not passed]
    css_class = "pass" if run["passed"] else "fail"
    status = "pass" if run["passed"] else "fail"
    return (
        f"<tr class=\"{css_class}\">"
        f"<td>{status}</td>"
        f"<td>{html.escape(str(run['scenario']))}</td>"
        f"<td>{float(run['target_speed_mps']):.1f}</td>"
        f"<td>{float(run['horizontal_fov_deg']):.0f}</td>"
        f"<td>{float(run['latency_ms']):.0f} ms</td>"
        f"<td>{float(summary['visible_ratio']):.1%}</td>"
        f"<td>{format_optional(summary['average_screen_error'])}</td>"
        f"<td>{float(summary['lost_ratio']):.1%}</td>"
        f"<td>{float(summary['longest_lost_streak_sec']):.2f}s</td>"
        f"<td>{float(summary['false_redetect_ratio']):.1%}</td>"
        f"<td>{float(run['score']):.1f}</td>"
        f"<td>{html.escape(', '.join(failed) if failed else '-')}</td>"
        "</tr>"
    )


def format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def benchmark_control_sim(
    *,
    out_dir: Path,
    scenarios: list[str],
    durations: list[float],
    target_speeds: list[float],
    horizontal_fovs: list[float],
    latencies_ms: list[float],
    trials: int,
    dt_sec: float,
    seed: int,
) -> dict[str, object]:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runs: list[dict[str, object]] = []
    for scenario in scenarios:
        for duration in durations:
            for speed in target_speeds:
                for fov in horizontal_fovs:
                    for latency_ms in latencies_ms:
                        for trial in range(trials):
                            run_seed = seed + trial
                            config = SimConfig(
                                duration_sec=duration,
                                dt_sec=dt_sec,
                                seed=run_seed,
                                scenario=scenario,
                                target_speed_mps=speed,
                                horizontal_fov_deg=fov,
                                vertical_fov_deg=max(35.0, fov * 0.65),
                                latency_ms=latency_ms,
                            )
                            sim_report = run_simulation(config)
                            runs.append(
                                summarize_run(
                                    sim_report,
                                    scenario=scenario,
                                    speed=speed,
                                    fov=fov,
                                    latency_ms=latency_ms,
                                    seed=run_seed,
                                )
                            )
    report = {
        "elapsed_sec": time.perf_counter() - started,
        "trials": trials,
        "dt_sec": dt_sec,
        "seed": seed,
        "gate": {
            "min_visible_ratio": 0.95,
            "max_lost_ratio": 0.02,
            "max_average_screen_error": 0.35,
            "max_longest_lost_sec": 1.5,
            "max_false_redetect_ratio": 0.0,
            "min_required_pass_ratio": 0.80,
        },
        "aggregate": aggregate_runs(runs),
        "runs": runs,
    }
    (out_dir / "control_benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "control_benchmark.html").write_text(render_control_benchmark_html(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the richer closed-loop drone control simulator.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", type=parse_str_list, default=parse_str_list("wander,swerve,sharp_turns,fast_break"))
    parser.add_argument("--durations", type=parse_float_list, default=parse_float_list("60"))
    parser.add_argument("--target-speeds", type=parse_float_list, default=parse_float_list("3,6,10"))
    parser.add_argument("--horizontal-fovs", type=parse_float_list, default=parse_float_list("70,110"))
    parser.add_argument("--latencies-ms", type=parse_float_list, default=parse_float_list("100,250,400"))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--dt-sec", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=11)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = benchmark_control_sim(
        out_dir=args.out,
        scenarios=args.scenarios,
        durations=args.durations,
        target_speeds=args.target_speeds,
        horizontal_fovs=args.horizontal_fovs,
        latencies_ms=args.latencies_ms,
        trials=args.trials,
        dt_sec=args.dt_sec,
        seed=args.seed,
    )
    aggregate = report["aggregate"]
    print(f"benchmark: {args.out / 'control_benchmark.json'}")
    print(f"html: {args.out / 'control_benchmark.html'}")
    print(f"passed: {aggregate['passed']}")
    print(f"pass_ratio: {aggregate['pass_ratio']:.2%}")
    print(f"passed_runs: {aggregate['passed_runs']} / {aggregate['total_runs']}")


if __name__ == "__main__":
    main()
