from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from aerial_vision.live_track import live_track
from aerial_vision.pipeline import PROFILE_CONFIGS


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All values must be positive integers.")
    return values


def parse_optional_int_list(value: str) -> list[int | None]:
    values: list[int | None] = []
    for part in value.split(","):
        item = part.strip().lower()
        if not item:
            continue
        values.append(None if item in {"none", "native"} else int(item))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one value.")
    if any(item is not None and item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All numeric values must be positive integers.")
    return values


def state_ratio(result: dict[str, object], state: str) -> float:
    observations = result.get("observations", [])
    total = len(observations) if isinstance(observations, list) else 0
    if total <= 0:
        return 0.0
    state_counts = result.get("state_counts", {})
    if not isinstance(state_counts, dict):
        return 0.0
    return float(state_counts.get(state, 0)) / total


def live_score(result: dict[str, object]) -> float:
    processed_fps = float(result.get("processed_fps") or 0.0)
    locked = state_ratio(result, "locked")
    weak = state_ratio(result, "weak_lock")
    risky = state_ratio(result, "id_switch_risk")
    lost = state_ratio(result, "lost")
    dropped = float(result.get("dropped_frames") or 0)
    processed = float(result.get("processed_frames") or 0)
    drop_ratio = dropped / max(1.0, processed + dropped)

    stability = locked + 0.55 * weak + 0.35 * risky - lost - 0.5 * drop_ratio
    return max(0.0, processed_fps * max(0.0, stability))


def summarize_result(result: dict[str, object]) -> dict[str, object]:
    state_counts = result.get("state_counts", {})
    return {
        "processed_fps": result.get("processed_fps"),
        "processed_frames": result.get("processed_frames"),
        "dropped_frames": result.get("dropped_frames"),
        "detector_frames": result.get("detector_frames"),
        "optical_flow_frames": result.get("optical_flow_frames"),
        "state_counts": state_counts,
        "locked_ratio": state_ratio(result, "locked"),
        "score": live_score(result),
    }


def aggregate_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    if not summaries:
        return {
            "processed_fps": 0.0,
            "median_processed_fps": 0.0,
            "locked_ratio": 0.0,
            "score": 0.0,
            "trials": 0,
        }

    fps_values = [float(summary.get("processed_fps") or 0.0) for summary in summaries]
    score_values = [float(summary.get("score") or 0.0) for summary in summaries]
    lock_values = [float(summary.get("locked_ratio") or 0.0) for summary in summaries]
    dropped_values = [int(summary.get("dropped_frames") or 0) for summary in summaries]
    return {
        "processed_fps": statistics.mean(fps_values),
        "median_processed_fps": statistics.median(fps_values),
        "locked_ratio": statistics.mean(lock_values),
        "min_locked_ratio": min(lock_values),
        "dropped_frames": statistics.mean(dropped_values),
        "score": statistics.median(score_values),
        "trials": len(summaries),
        "trial_summaries": summaries,
    }


def benchmark_live(
    source: str,
    *,
    profile: str,
    target_id: int,
    out_dir: Path,
    resize_widths: list[int | None],
    detect_everies: list[int],
    imgsizes: list[int | None],
    max_dets: list[int],
    max_frames: int | None,
    realtime: bool,
    max_latency_sec: float,
    trials: int,
    device: str | None,
    half: bool,
    tracker: str,
) -> dict[str, object]:
    if trials <= 0:
        raise ValueError("trials must be positive.")

    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    run_index = 0
    started = time.perf_counter()

    for resize_width in resize_widths:
        for detect_every in detect_everies:
            for imgsz in imgsizes:
                for max_det in max_dets:
                    run_index += 1
                    trial_summaries: list[dict[str, object]] = []
                    trial_dirs: list[str] = []
                    for trial_index in range(1, trials + 1):
                        run_dir = out_dir / f"run_{run_index:03d}_trial_{trial_index:02d}"
                        result = live_track(
                            source,
                            profile=profile,
                            out_dir=run_dir,
                            target_id=target_id,
                            detect_every=detect_every,
                            resize_width=resize_width,
                            max_frames=max_frames,
                            realtime=realtime,
                            max_latency_sec=max_latency_sec,
                            display=False,
                            imgsz=imgsz,
                            max_det=max_det,
                            device=device,
                            half=half,
                            tracker=tracker,
                        )
                        trial_summaries.append(summarize_result(result))
                        trial_dirs.append(str(run_dir))
                    summary = aggregate_summaries(trial_summaries)
                    runs.append(
                        {
                            "trial_dirs": trial_dirs,
                            "resize_width": resize_width,
                            "detect_every": detect_every,
                            "imgsz": imgsz,
                            "max_det": max_det,
                            "summary": summary,
                        }
                    )

    best = max(runs, key=lambda run: float(run["summary"]["score"])) if runs else None
    report = {
        "source": source,
        "profile": profile,
        "target_id": target_id,
        "realtime": realtime,
        "max_latency_sec": max_latency_sec,
        "max_frames": max_frames,
        "trials": trials,
        "device": device,
        "half": half,
        "tracker": tracker,
        "elapsed_sec": time.perf_counter() - started,
        "best": best,
        "runs": runs,
    }
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark live-like target tracking settings against one video source.")
    parser.add_argument("source", help="Video path, camera index, or stream URL.")
    parser.add_argument("--profile", choices=tuple(PROFILE_CONFIGS), default="vehicles", help="Detection profile.")
    parser.add_argument("--target-id", type=int, required=True, help="Track ID to lock onto.")
    parser.add_argument("--out", type=Path, required=True, help="Directory for benchmark runs and benchmark.json.")
    parser.add_argument("--resize-widths", type=parse_optional_int_list, default=parse_optional_int_list("960,720"), help="Comma list of resize widths, or native.")
    parser.add_argument("--detect-everies", type=parse_int_list, default=parse_int_list("10,15"), help="Comma list of detector refresh intervals.")
    parser.add_argument("--imgsizes", type=parse_optional_int_list, default=parse_optional_int_list("640,512"), help="Comma list of YOLO imgsz values, or native.")
    parser.add_argument("--max-dets", type=parse_int_list, default=parse_int_list("50"), help="Comma list of max detections per refresh.")
    parser.add_argument("--max-frames", type=int, default=120, help="Frame limit for a fast benchmark.")
    parser.add_argument("--max-latency-sec", type=float, default=0.5, help="Realtime drop tolerance.")
    parser.add_argument("--trials", type=int, default=1, help="Repeat each settings combination this many times.")
    parser.add_argument("--realtime", action="store_true", help="Pace to source FPS and allow frame dropping.")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu, mps, 0, cuda:0.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on supported GPU devices.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = benchmark_live(
        args.source,
        profile=args.profile,
        target_id=args.target_id,
        out_dir=args.out,
        resize_widths=args.resize_widths,
        detect_everies=args.detect_everies,
        imgsizes=args.imgsizes,
        max_dets=args.max_dets,
        max_frames=args.max_frames,
        realtime=args.realtime,
        max_latency_sec=args.max_latency_sec,
        trials=args.trials,
        device=args.device,
        half=args.half,
        tracker=args.tracker,
    )
    best = report["best"]
    print(f"benchmark: {args.out / 'benchmark.json'}")
    if best is not None:
        summary = best["summary"]
        print(
            "best: "
            f"resize_width={best['resize_width']} "
            f"detect_every={best['detect_every']} "
            f"imgsz={best['imgsz']} "
            f"max_det={best['max_det']} "
            f"fps={summary['processed_fps']:.2f} "
            f"locked={summary['locked_ratio']:.2%} "
            f"score={summary['score']:.2f}"
        )


if __name__ == "__main__":
    main()
