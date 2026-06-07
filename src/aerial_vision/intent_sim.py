from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from aerial_vision.control_sim import Vec3, clamp, wrap_angle


@dataclass(frozen=True)
class MockDroneState:
    position: Vec3
    yaw_rad: float
    camera_pitch_deg: float


@dataclass(frozen=True)
class IntentFrame:
    frame_index: int
    timestamp_sec: float
    mode: str
    yaw_rate_deg_s: float
    camera_pitch_rate_deg_s: float
    forward_mps: float
    normalized_offset: tuple[float, float] | None
    center_error: float | None
    tag_enabled: bool

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> IntentFrame:
        offset_payload = payload.get("normalized_offset")
        offset = None
        if isinstance(offset_payload, (list, tuple)) and len(offset_payload) == 2:
            offset = (float(offset_payload[0]), float(offset_payload[1]))
        return cls(
            frame_index=int(payload["frame_index"]),
            timestamp_sec=float(payload["timestamp_sec"]),
            mode=str(payload["mode"]),
            yaw_rate_deg_s=float(payload["yaw_rate_deg_s"]),
            camera_pitch_rate_deg_s=float(payload["camera_pitch_rate_deg_s"]),
            forward_mps=float(payload["forward_mps"]),
            normalized_offset=offset,
            center_error=None if payload.get("center_error") is None else float(payload["center_error"]),
            tag_enabled=bool(payload.get("tag_enabled", False)),
        )


@dataclass(frozen=True)
class MockCommand:
    yaw_rate_deg_s: float
    camera_pitch_rate_deg_s: float
    forward_mps: float
    mode: str


@dataclass(frozen=True)
class MockBridgeConfig:
    max_yaw_rate_deg_s: float = 45.0
    max_pitch_rate_deg_s: float = 30.0
    max_forward_mps: float = 8.0
    max_yaw_accel_deg_s2: float = 120.0
    max_forward_accel_mps2: float = 4.0
    min_camera_pitch_deg: float = -85.0
    max_camera_pitch_deg: float = -10.0
    default_dt_sec: float = 1 / 30
    start_altitude_m: float = 30.0


def load_intents(path: Path) -> list[IntentFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("intents", payload.get("control_intents", []))
    return [IntentFrame.from_dict(row) for row in rows]


def dt_between(current: IntentFrame, previous: IntentFrame | None, config: MockBridgeConfig) -> float:
    if previous is None:
        return config.default_dt_sec
    dt_sec = current.timestamp_sec - previous.timestamp_sec
    if dt_sec <= 0:
        return config.default_dt_sec
    return min(dt_sec, 0.25)


def bridge_command(intent: IntentFrame, config: MockBridgeConfig) -> MockCommand:
    forward = intent.forward_mps
    if intent.mode in {"hold", "search", "center", "centered"}:
        forward = min(forward, 0.0)
    return MockCommand(
        yaw_rate_deg_s=clamp(intent.yaw_rate_deg_s, -config.max_yaw_rate_deg_s, config.max_yaw_rate_deg_s),
        camera_pitch_rate_deg_s=clamp(
            intent.camera_pitch_rate_deg_s,
            -config.max_pitch_rate_deg_s,
            config.max_pitch_rate_deg_s,
        ),
        forward_mps=clamp(forward, 0.0, config.max_forward_mps),
        mode=intent.mode,
    )


def limit_rate_change(current: float, desired: float, *, max_delta: float) -> float:
    return current + clamp(desired - current, -max_delta, max_delta)


def step_mock_drone(
    state: MockDroneState,
    command: MockCommand,
    *,
    dt_sec: float,
    previous_yaw_rate_deg_s: float,
    previous_forward_mps: float,
    config: MockBridgeConfig,
) -> tuple[MockDroneState, float, float]:
    yaw_rate = limit_rate_change(
        previous_yaw_rate_deg_s,
        command.yaw_rate_deg_s,
        max_delta=config.max_yaw_accel_deg_s2 * dt_sec,
    )
    forward_mps = limit_rate_change(
        previous_forward_mps,
        command.forward_mps,
        max_delta=config.max_forward_accel_mps2 * dt_sec,
    )
    yaw_rad = wrap_angle(state.yaw_rad + math.radians(yaw_rate) * dt_sec)
    camera_pitch = clamp(
        state.camera_pitch_deg + command.camera_pitch_rate_deg_s * dt_sec,
        config.min_camera_pitch_deg,
        config.max_camera_pitch_deg,
    )
    delta = Vec3(math.cos(yaw_rad) * forward_mps * dt_sec, math.sin(yaw_rad) * forward_mps * dt_sec, 0.0)
    return (
        MockDroneState(
            position=Vec3(state.position.x + delta.x, state.position.y + delta.y, state.position.z),
            yaw_rad=yaw_rad,
            camera_pitch_deg=camera_pitch,
        ),
        yaw_rate,
        forward_mps,
    )


def run_intent_simulation(intents: list[IntentFrame], config: MockBridgeConfig) -> dict[str, object]:
    state = MockDroneState(position=Vec3(0.0, 0.0, config.start_altitude_m), yaw_rad=0.0, camera_pitch_deg=-45.0)
    rows: list[dict[str, object]] = []
    previous_intent: IntentFrame | None = None
    previous_yaw_rate = 0.0
    previous_forward = 0.0

    for intent in intents:
        dt_sec = dt_between(intent, previous_intent, config)
        command = bridge_command(intent, config)
        state, applied_yaw_rate, applied_forward = step_mock_drone(
            state,
            command,
            dt_sec=dt_sec,
            previous_yaw_rate_deg_s=previous_yaw_rate,
            previous_forward_mps=previous_forward,
            config=config,
        )
        previous_yaw_rate = applied_yaw_rate
        previous_forward = applied_forward
        rows.append(
            {
                "frame_index": intent.frame_index,
                "timestamp_sec": intent.timestamp_sec,
                "dt_sec": dt_sec,
                "mode": intent.mode,
                "tag_enabled": intent.tag_enabled,
                "intent_yaw_rate_deg_s": intent.yaw_rate_deg_s,
                "applied_yaw_rate_deg_s": applied_yaw_rate,
                "intent_pitch_rate_deg_s": intent.camera_pitch_rate_deg_s,
                "camera_pitch_deg": state.camera_pitch_deg,
                "intent_forward_mps": intent.forward_mps,
                "applied_forward_mps": applied_forward,
                "drone_x": state.position.x,
                "drone_y": state.position.y,
                "drone_z": state.position.z,
                "drone_yaw_deg": math.degrees(state.yaw_rad),
                "normalized_offset_x": None if intent.normalized_offset is None else intent.normalized_offset[0],
                "normalized_offset_y": None if intent.normalized_offset is None else intent.normalized_offset[1],
                "center_error": intent.center_error,
            }
        )
        previous_intent = intent

    forward_frames = sum(1 for row in rows if float(row["applied_forward_mps"]) > 0)
    search_frames = sum(1 for row in rows if row["mode"] == "search")
    tag_frames = sum(1 for row in rows if row["mode"] == "tag")
    distance = rows[-1]["drone_x"] if rows else 0.0
    if rows:
        distance = math.hypot(float(rows[-1]["drone_x"]), float(rows[-1]["drone_y"]))

    return {
        "config": asdict(config),
        "summary": {
            "frames": len(rows),
            "duration_sec": rows[-1]["timestamp_sec"] - rows[0]["timestamp_sec"] if len(rows) > 1 else 0.0,
            "forward_frames": forward_frames,
            "search_frames": search_frames,
            "tag_frames": tag_frames,
            "final_distance_m": distance,
            "final_yaw_deg": rows[-1]["drone_yaw_deg"] if rows else 0.0,
        },
        "trajectory": rows,
    }


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_preview_frame(rows: list[dict[str, object]], index: int, *, width: int = 1000, height: int = 560) -> Image.Image:
    image = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(image)
    row = rows[index]
    past = rows[: index + 1]

    camera_left, camera_top, camera_width, camera_height = 30, 70, 430, 300
    map_left, map_top, map_width, map_height = 520, 70, 420, 360
    draw.text((30, 24), "Mock PX4 Bridge: camera intent view", fill=(20, 25, 35))
    draw.text((520, 24), "Simulated drone path from intent", fill=(20, 25, 35))
    draw.rectangle((camera_left, camera_top, camera_left + camera_width, camera_top + camera_height), fill=(35, 38, 42), outline=(180, 185, 195), width=2)
    center = (camera_left + camera_width // 2, camera_top + camera_height // 2)
    draw.line((center[0] - 22, center[1], center[0] + 22, center[1]), fill=(230, 230, 230), width=1)
    draw.line((center[0], center[1] - 22, center[0], center[1] + 22), fill=(230, 230, 230), width=1)

    if row["normalized_offset_x"] is None or row["normalized_offset_y"] is None:
        draw.text((camera_left + 150, camera_top + 135), "SEARCH", fill=(255, 190, 80))
    else:
        norm_x = clamp(float(row["normalized_offset_x"]), -1.0, 1.0)
        norm_y = clamp(float(row["normalized_offset_y"]), -1.0, 1.0)
        target = (
            camera_left + int((norm_x + 1) * 0.5 * camera_width),
            camera_top + int((1 - (norm_y + 1) * 0.5) * camera_height),
        )
        color = (255, 120, 30) if row["mode"] == "tag" else (60, 220, 100)
        if row["mode"] == "hold":
            color = (255, 90, 90)
        draw.ellipse((target[0] - 9, target[1] - 9, target[0] + 9, target[1] + 9), fill=color)
        draw.line((center, target), fill=color, width=2)

    points = [(float(item["drone_x"]), float(item["drone_y"])) for item in rows]
    span = max(
        max((abs(point[0]) for point in points), default=1.0),
        max((abs(point[1]) for point in points), default=1.0),
        5.0,
    )

    def project(point: tuple[float, float]) -> tuple[int, int]:
        px = map_left + map_width // 2 + int(point[0] / span * (map_width * 0.42))
        py = map_top + map_height // 2 - int(point[1] / span * (map_height * 0.42))
        return px, py

    draw.rectangle((map_left, map_top, map_left + map_width, map_top + map_height), outline=(185, 190, 200), width=1)
    path = [project((float(item["drone_x"]), float(item["drone_y"]))) for item in past]
    if len(path) > 1:
        draw.line(path, fill=(40, 90, 220), width=3)
    drone_xy = path[-1]
    draw.ellipse((drone_xy[0] - 7, drone_xy[1] - 7, drone_xy[0] + 7, drone_xy[1] + 7), fill=(40, 90, 220))
    yaw = math.radians(float(row["drone_yaw_deg"]))
    heading = (drone_xy[0] + int(math.cos(yaw) * 28), drone_xy[1] - int(math.sin(yaw) * 28))
    draw.line((drone_xy, heading), fill=(20, 40, 130), width=3)

    metrics_y = 400
    draw.text((30, metrics_y), f"time: {float(row['timestamp_sec']):.2f}s  mode: {row['mode']}", fill=(30, 35, 45))
    draw.text((30, metrics_y + 24), f"yaw intent/applied: {float(row['intent_yaw_rate_deg_s']):+.1f}/{float(row['applied_yaw_rate_deg_s']):+.1f} deg/s", fill=(30, 35, 45))
    draw.text((30, metrics_y + 48), f"pitch: {float(row['camera_pitch_deg']):+.1f} deg  forward: {float(row['applied_forward_mps']):.1f} m/s", fill=(30, 35, 45))
    draw.text((520, metrics_y + 24), f"x/y: {float(row['drone_x']):+.1f}, {float(row['drone_y']):+.1f} m", fill=(30, 35, 45))
    draw.text((520, metrics_y + 48), f"yaw: {float(row['drone_yaw_deg']):+.1f} deg", fill=(30, 35, 45))
    return image


def write_preview_gif(rows: list[dict[str, object]], output_path: Path, *, every_n: int = 3) -> None:
    if not rows:
        return
    frames = [draw_preview_frame(rows, index) for index in range(0, len(rows), max(1, every_n))]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )


def prepare_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def run_and_write(intent_path: Path, out_dir: Path, config: MockBridgeConfig) -> dict[str, object]:
    prepare_output_dir(out_dir)
    report = run_intent_simulation(load_intents(intent_path), config)
    rows = report["trajectory"]
    (out_dir / "mock_bridge_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(rows, out_dir / "mock_bridge_trajectory.csv")
    write_preview_gif(rows, out_dir / "mock_bridge_preview.gif")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a mock PX4-style drone response from control_intent.json.")
    parser.add_argument("intent", type=Path, help="Path to control_intent.json or live_session.json with control_intents.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for report, trajectory, and preview GIF.")
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=45.0)
    parser.add_argument("--max-pitch-rate-deg-s", type=float, default=30.0)
    parser.add_argument("--max-forward-mps", type=float, default=8.0)
    parser.add_argument("--max-yaw-accel-deg-s2", type=float, default=120.0)
    parser.add_argument("--max-forward-accel-mps2", type=float, default=4.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MockBridgeConfig(
        max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
        max_pitch_rate_deg_s=args.max_pitch_rate_deg_s,
        max_forward_mps=args.max_forward_mps,
        max_yaw_accel_deg_s2=args.max_yaw_accel_deg_s2,
        max_forward_accel_mps2=args.max_forward_accel_mps2,
    )
    report = run_and_write(args.intent, args.out, config)
    summary = report["summary"]
    print(f"report: {args.out / 'mock_bridge_report.json'}")
    print(f"trajectory: {args.out / 'mock_bridge_trajectory.csv'}")
    print(f"preview: {args.out / 'mock_bridge_preview.gif'}")
    print(f"frames: {summary['frames']}")
    print(f"forward_frames: {summary['forward_frames']}")
    print(f"search_frames: {summary['search_frames']}")
    print(f"tag_frames: {summary['tag_frames']}")
    print(f"final_distance_m: {summary['final_distance_m']:.2f}")


if __name__ == "__main__":
    main()
