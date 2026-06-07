from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from aerial_vision.control_sim import (
    CameraObservation,
    DroneState,
    SimConfig,
    TargetState,
    Vec3,
    clamp,
    observe_target,
    scenario_turn_rate_deg_s,
    wrap_angle,
)
from aerial_vision.track_video import ControlIntent


@dataclass(frozen=True)
class RaceConfig:
    duration_sec: float = 30.0
    dt_sec: float = 0.1
    seed: int = 11
    scenario: str = "swerve"
    tag_mode: bool = False
    target_speed_mps: float = 6.0
    drone_altitude_m: float = 30.0
    drone_start_x_m: float = -10.0
    drone_start_y_m: float = -10.0
    camera_pitch_deg: float = -45.0
    horizontal_fov_deg: float = 70.0
    vertical_fov_deg: float = 45.0
    yaw_gain_deg_s: float = 45.0
    pitch_gain_deg_s: float = 35.0
    max_yaw_rate_deg_s: float = 90.0
    max_pitch_rate_deg_s: float = 60.0
    max_forward_mps: float = 12.0
    follow_forward_mps: float = 2.5
    tag_forward_mps: float = 8.0
    center_deadband: float = 0.08
    follow_center_gate: float = 0.35
    tag_center_gate: float = 0.45
    away_box_ratio_threshold: float = 0.985
    tracking_noise: float = 0.015
    confidence_noise: float = 0.04
    lost_confidence: float = 0.2
    lost_search_yaw_deg_s: float = 25.0

    @property
    def steps(self) -> int:
        return max(1, int(round(self.duration_sec / self.dt_sec)))


def sim_config(config: RaceConfig) -> SimConfig:
    return SimConfig(
        duration_sec=config.duration_sec,
        dt_sec=config.dt_sec,
        seed=config.seed,
        scenario=config.scenario,
        target_speed_mps=config.target_speed_mps,
        drone_altitude_m=config.drone_altitude_m,
        camera_pitch_deg=config.camera_pitch_deg,
        horizontal_fov_deg=config.horizontal_fov_deg,
        vertical_fov_deg=config.vertical_fov_deg,
        max_yaw_rate_deg_s=config.max_yaw_rate_deg_s,
        max_pitch_rate_deg_s=config.max_pitch_rate_deg_s,
        max_forward_mps=config.max_forward_mps,
        tracking_noise=config.tracking_noise,
        confidence_noise=config.confidence_noise,
        lost_confidence=config.lost_confidence,
    )


def step_race_target(target: TargetState, config: RaceConfig, rng: random.Random, *, step: int) -> TargetState:
    base = sim_config(config)
    speed = config.target_speed_mps
    if config.scenario == "fast_break":
        speed = config.target_speed_mps * (1.0 + 0.5 * max(0.0, math.sin(step * config.dt_sec * 1.5)))
    heading = wrap_angle(target.heading_rad + math.radians(scenario_turn_rate_deg_s(step, target, base, rng)) * config.dt_sec)
    position = Vec3(
        target.position.x + math.cos(heading) * speed * config.dt_sec,
        target.position.y + math.sin(heading) * speed * config.dt_sec,
        0.0,
    )
    bounds = base.target_bounds_m
    if abs(position.x) > bounds:
        heading = wrap_angle(math.pi - heading)
        position = Vec3(clamp(position.x, -bounds, bounds), position.y, 0.0)
    if abs(position.y) > bounds:
        heading = wrap_angle(-heading)
        position = Vec3(position.x, clamp(position.y, -bounds, bounds), 0.0)
    return TargetState(position=position, heading_rad=heading, speed_mps=speed)


def intent_from_observation(
    observation: CameraObservation,
    *,
    previous_box_size: float | None,
    previous_offset: tuple[float, float] | None,
    tag_mode: bool,
    config: RaceConfig,
) -> ControlIntent:
    if not observation.visible or observation.confidence < config.lost_confidence or observation.norm_x is None or observation.norm_y is None:
        search_direction = 1.0
        if previous_offset is not None and abs(previous_offset[0]) > 0.05:
            search_direction = 1.0 if previous_offset[0] > 0 else -1.0
        return ControlIntent(
            frame_index=0,
            timestamp_sec=0.0,
            mode="search",
            reason="target not visible; rotate toward last known side",
            yaw_rate_deg_s=search_direction * config.lost_search_yaw_deg_s,
            camera_pitch_rate_deg_s=0.0,
            forward_mps=0.0,
            normalized_offset=None,
            center_error=None,
            tag_enabled=tag_mode,
            box_area_ratio=None,
        )

    offset = (observation.norm_x, observation.norm_y)
    center_error = math.hypot(observation.norm_x, observation.norm_y)
    yaw = clamp(observation.norm_x * config.yaw_gain_deg_s, -config.max_yaw_rate_deg_s, config.max_yaw_rate_deg_s)
    pitch = clamp(-observation.norm_y * config.pitch_gain_deg_s, -config.max_pitch_rate_deg_s, config.max_pitch_rate_deg_s)
    box_ratio = None
    if previous_box_size is not None and previous_box_size > 0 and observation.box_size is not None:
        box_ratio = observation.box_size / previous_box_size

    mode = "center"
    reason = "target visible; center camera"
    forward = 0.0
    if tag_mode:
        mode = "tag"
        reason = "tag mode active"
        if center_error <= config.tag_center_gate:
            forward = config.tag_forward_mps * max(0.0, 1.0 - center_error / max(config.tag_center_gate, 1e-6))
        else:
            reason = "tag mode active; center before forward motion"
    elif box_ratio is not None and box_ratio < config.away_box_ratio_threshold and center_error <= config.follow_center_gate:
        mode = "follow"
        reason = "target appears to be moving away"
        forward = config.follow_forward_mps
    elif center_error <= config.center_deadband:
        mode = "centered"
        reason = "target centered; hold position"

    return ControlIntent(
        frame_index=0,
        timestamp_sec=0.0,
        mode=mode,
        reason=reason,
        yaw_rate_deg_s=yaw,
        camera_pitch_rate_deg_s=pitch,
        forward_mps=forward,
        normalized_offset=offset,
        center_error=center_error,
        tag_enabled=tag_mode,
        box_area_ratio=box_ratio,
    )


def step_drone_from_intent(drone: DroneState, intent: ControlIntent, config: RaceConfig) -> DroneState:
    yaw = wrap_angle(drone.yaw_rad + math.radians(intent.yaw_rate_deg_s) * config.dt_sec)
    pitch = clamp(
        drone.camera_pitch_rad + math.radians(intent.camera_pitch_rate_deg_s) * config.dt_sec,
        math.radians(-85.0),
        math.radians(-10.0),
    )
    forward = clamp(intent.forward_mps, 0.0, config.max_forward_mps)
    delta = Vec3(math.cos(yaw) * forward * config.dt_sec, math.sin(yaw) * forward * config.dt_sec, 0.0)
    return DroneState(
        position=Vec3(drone.position.x + delta.x, drone.position.y + delta.y, config.drone_altitude_m),
        yaw_rad=yaw,
        camera_pitch_rad=pitch,
    )


def run_race_simulation(config: RaceConfig) -> dict[str, object]:
    rng = random.Random(config.seed)
    perception_rng = random.Random(config.seed + 50_000)
    base = sim_config(config)
    drone = DroneState(
        position=Vec3(config.drone_start_x_m, config.drone_start_y_m, config.drone_altitude_m),
        yaw_rad=math.radians(45.0),
        camera_pitch_rad=math.radians(config.camera_pitch_deg),
    )
    target = TargetState(position=Vec3(0.0, 0.0, 0.0), heading_rad=math.radians(20.0), speed_mps=config.target_speed_mps)
    rows: list[dict[str, object]] = []
    previous_box_size: float | None = None
    previous_offset: tuple[float, float] | None = None

    for step in range(config.steps):
        time_sec = step * config.dt_sec
        observation = observe_target(drone, target, base, perception_rng)
        intent = intent_from_observation(
            observation,
            previous_box_size=previous_box_size,
            previous_offset=previous_offset,
            tag_mode=config.tag_mode,
            config=config,
        )
        intent = ControlIntent(
            frame_index=step,
            timestamp_sec=time_sec,
            mode=intent.mode,
            reason=intent.reason,
            yaw_rate_deg_s=intent.yaw_rate_deg_s,
            camera_pitch_rate_deg_s=intent.camera_pitch_rate_deg_s,
            forward_mps=intent.forward_mps,
            normalized_offset=intent.normalized_offset,
            center_error=intent.center_error,
            tag_enabled=intent.tag_enabled,
            box_area_ratio=intent.box_area_ratio,
        )
        drone = step_drone_from_intent(drone, intent, config)
        target = step_race_target(target, config, rng, step=step)
        if observation.visible and observation.norm_x is not None and observation.norm_y is not None:
            previous_offset = (observation.norm_x, observation.norm_y)
        if observation.box_size is not None:
            previous_box_size = observation.box_size

        horizontal_distance = (target.position - drone.position).length_xy
        rows.append(
            {
                "step": step,
                "time_sec": time_sec,
                "mode": intent.mode,
                "reason": intent.reason,
                "visible": observation.visible,
                "confidence": observation.confidence,
                "norm_x": observation.norm_x,
                "norm_y": observation.norm_y,
                "box_size": observation.box_size,
                "box_area_ratio": intent.box_area_ratio,
                "center_error": intent.center_error,
                "yaw_rate_deg_s": intent.yaw_rate_deg_s,
                "camera_pitch_rate_deg_s": intent.camera_pitch_rate_deg_s,
                "forward_mps": intent.forward_mps,
                "drone_x": drone.position.x,
                "drone_y": drone.position.y,
                "drone_z": drone.position.z,
                "drone_yaw_deg": math.degrees(drone.yaw_rad),
                "camera_pitch_deg": math.degrees(drone.camera_pitch_rad),
                "target_x": target.position.x,
                "target_y": target.position.y,
                "distance_m": horizontal_distance,
            }
        )

    visible_count = sum(1 for row in rows if row["visible"])
    lost_count = len(rows) - visible_count
    errors = [float(row["center_error"]) for row in rows if row["center_error"] is not None]
    mode_counts: dict[str, int] = {}
    for row in rows:
        mode = str(row["mode"])
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    return {
        "config": asdict(config),
        "summary": {
            "frames": len(rows),
            "visible_ratio": visible_count / len(rows) if rows else 0.0,
            "lost_frames": lost_count,
            "average_center_error": sum(errors) / len(errors) if errors else None,
            "final_distance_m": rows[-1]["distance_m"] if rows else None,
            "mode_counts": mode_counts,
        },
        "trajectory": rows,
    }


def prepare_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def project_world(rows: list[dict[str, object]], *, width: int, height: int, padding: int):
    points = []
    for row in rows:
        points.append((float(row["drone_x"]), float(row["drone_y"])))
        points.append((float(row["target_x"]), float(row["target_y"])))
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    span = max(max_x - min_x, max_y - min_y, 1.0)

    def project(x: float, y: float) -> tuple[int, int]:
        px = padding + int((x - min_x) / span * (width - 2 * padding))
        py = height - padding - int((y - min_y) / span * (height - 2 * padding))
        return px, py

    return project


def draw_race_frame(rows: list[dict[str, object]], index: int, *, width: int = 1000, height: int = 560) -> Image.Image:
    image = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(image)
    row = rows[index]
    past = rows[: index + 1]
    camera_left, camera_top, camera_width, camera_height = 30, 70, 430, 300
    map_left, map_top, map_width, map_height = 520, 70, 420, 360

    draw.text((30, 24), "Closed-loop camera view", fill=(20, 25, 35))
    draw.text((520, 24), "Closed-loop world view", fill=(20, 25, 35))
    draw.rectangle((camera_left, camera_top, camera_left + camera_width, camera_top + camera_height), fill=(35, 38, 42), outline=(180, 185, 195), width=2)
    center = (camera_left + camera_width // 2, camera_top + camera_height // 2)
    draw.line((center[0] - 22, center[1], center[0] + 22, center[1]), fill=(230, 230, 230), width=1)
    draw.line((center[0], center[1] - 22, center[0], center[1] + 22), fill=(230, 230, 230), width=1)

    if row["norm_x"] is None or row["norm_y"] is None:
        draw.text((camera_left + 150, camera_top + 135), "TARGET LOST", fill=(255, 190, 80))
    else:
        norm_x = clamp(float(row["norm_x"]), -1.0, 1.0)
        norm_y = clamp(float(row["norm_y"]), -1.0, 1.0)
        target = (
            camera_left + int((norm_x + 1) * 0.5 * camera_width),
            camera_top + int((1 - (norm_y + 1) * 0.5) * camera_height),
        )
        color = (255, 120, 30) if row["mode"] == "tag" else (60, 220, 100)
        if row["mode"] == "search":
            color = (255, 190, 80)
        draw.ellipse((target[0] - 9, target[1] - 9, target[0] + 9, target[1] + 9), fill=color)
        draw.line((center, target), fill=color, width=2)

    project = project_world(rows, width=map_width, height=map_height, padding=25)
    draw.rectangle((map_left, map_top, map_left + map_width, map_top + map_height), outline=(185, 190, 200), width=1)
    drone_points = [(x + map_left, y + map_top) for x, y in (project(float(item["drone_x"]), float(item["drone_y"])) for item in past)]
    target_points = [(x + map_left, y + map_top) for x, y in (project(float(item["target_x"]), float(item["target_y"])) for item in past)]
    if len(target_points) > 1:
        draw.line(target_points, fill=(220, 60, 60), width=3)
    if len(drone_points) > 1:
        draw.line(drone_points, fill=(40, 90, 220), width=3)
    draw.ellipse((*offset(drone_points[-1], -7), *offset(drone_points[-1], 7)), fill=(40, 90, 220))
    draw.ellipse((*offset(target_points[-1], -7), *offset(target_points[-1], 7)), fill=(220, 60, 60))

    metrics_y = 400
    error_text = "n/a" if row["center_error"] is None else f"{float(row['center_error']):.3f}"
    draw.text((30, metrics_y), f"time: {float(row['time_sec']):.1f}s  mode: {row['mode']}  error: {error_text}", fill=(30, 35, 45))
    draw.text((30, metrics_y + 24), f"yaw: {float(row['yaw_rate_deg_s']):+.1f} deg/s  pitch: {float(row['camera_pitch_deg']):+.1f} deg", fill=(30, 35, 45))
    draw.text((30, metrics_y + 48), f"forward: {float(row['forward_mps']):.1f} m/s  visible: {row['visible']}", fill=(30, 35, 45))
    draw.text((520, metrics_y + 24), f"distance: {float(row['distance_m']):.1f} m", fill=(30, 35, 45))
    draw.text((520, metrics_y + 48), "blue=drone  red=target", fill=(30, 35, 45))
    return image


def offset(point: tuple[int, int], amount: int) -> tuple[int, int]:
    return point[0] + amount, point[1] + amount


def write_preview_gif(rows: list[dict[str, object]], output_path: Path, *, every_n: int = 3) -> None:
    if not rows:
        return
    frames = [draw_race_frame(rows, index) for index in range(0, len(rows), max(1, every_n))]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )


def run_and_write(config: RaceConfig, out_dir: Path) -> dict[str, object]:
    prepare_output_dir(out_dir)
    report = run_race_simulation(config)
    rows = report["trajectory"]
    (out_dir / "race_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(rows, out_dir / "race_trajectory.csv")
    write_preview_gif(rows, out_dir / "race_preview.gif")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a closed-loop drone race/tag camera simulator.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--dt-sec", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--scenario", choices=("wander", "swerve", "sharp_turns", "fast_break"), default="swerve")
    parser.add_argument("--tag-mode", action="store_true")
    parser.add_argument("--target-speed-mps", type=float, default=6.0)
    parser.add_argument("--drone-altitude-m", type=float, default=30.0)
    parser.add_argument("--drone-start-x-m", type=float, default=-10.0)
    parser.add_argument("--drone-start-y-m", type=float, default=-10.0)
    parser.add_argument("--horizontal-fov-deg", type=float, default=70.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=45.0)
    parser.add_argument("--max-forward-mps", type=float, default=12.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RaceConfig(
        duration_sec=args.duration_sec,
        dt_sec=args.dt_sec,
        seed=args.seed,
        scenario=args.scenario,
        tag_mode=args.tag_mode,
        target_speed_mps=args.target_speed_mps,
        drone_altitude_m=args.drone_altitude_m,
        drone_start_x_m=args.drone_start_x_m,
        drone_start_y_m=args.drone_start_y_m,
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
        max_forward_mps=args.max_forward_mps,
    )
    report = run_and_write(config, args.out)
    summary = report["summary"]
    print(f"report: {args.out / 'race_report.json'}")
    print(f"trajectory: {args.out / 'race_trajectory.csv'}")
    print(f"preview: {args.out / 'race_preview.gif'}")
    print(f"visible_ratio: {summary['visible_ratio']:.2%}")
    average_error = summary["average_center_error"]
    print(f"average_center_error: {average_error:.3f}" if average_error is not None else "average_center_error: n/a")
    print(f"lost_frames: {summary['lost_frames']}")
    print(f"mode_counts: {summary['mode_counts']}")
    print(f"final_distance_m: {summary['final_distance_m']:.2f}" if summary["final_distance_m"] is not None else "final_distance_m: n/a")


if __name__ == "__main__":
    main()
