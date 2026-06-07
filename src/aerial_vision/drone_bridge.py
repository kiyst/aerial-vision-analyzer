from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from aerial_vision.safety import SafetyLimits, safety_filter_intent
from aerial_vision.track_video import ControlIntent


@dataclass(frozen=True)
class BridgeConfig:
    max_setpoint_gap_sec: float = 0.5
    offboard_warmup_setpoints: int = 10


@dataclass(frozen=True)
class BridgeSetpoint:
    frame_index: int
    timestamp_sec: float
    mode: str
    reason: str
    yaw_rate_deg_s: float
    camera_pitch_rate_deg_s: float
    forward_mps: float
    center_error: float | None
    tag_enabled: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DroneBridge(Protocol):
    def arm(self) -> None:
        ...

    def disarm(self) -> None:
        ...

    def set_mode_offboard(self) -> None:
        ...

    def send_setpoint(self, setpoint: BridgeSetpoint) -> None:
        ...

    def stop(self) -> None:
        ...


@dataclass
class MockPX4Bridge:
    armed: bool = False
    offboard: bool = False
    stopped: bool = False

    def __post_init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.setpoints: list[BridgeSetpoint] = []

    def arm(self) -> None:
        self.armed = True
        self.events.append({"event": "arm"})

    def disarm(self) -> None:
        self.armed = False
        self.events.append({"event": "disarm"})

    def set_mode_offboard(self) -> None:
        if not self.armed:
            raise RuntimeError("Cannot enter offboard mode before arming.")
        self.offboard = True
        self.events.append({"event": "set_mode_offboard"})

    def send_setpoint(self, setpoint: BridgeSetpoint) -> None:
        if not self.armed or not self.offboard:
            raise RuntimeError("Cannot send setpoints until mock bridge is armed and in offboard mode.")
        if self.stopped:
            raise RuntimeError("Cannot send setpoints after stop.")
        self.setpoints.append(setpoint)

    def stop(self) -> None:
        self.stopped = True
        self.events.append({"event": "stop"})


def control_intent_from_dict(payload: dict[str, object]) -> ControlIntent:
    offset_payload = payload.get("normalized_offset")
    offset = None
    if isinstance(offset_payload, (list, tuple)) and len(offset_payload) == 2:
        offset = (float(offset_payload[0]), float(offset_payload[1]))
    return ControlIntent(
        frame_index=int(payload["frame_index"]),
        timestamp_sec=float(payload["timestamp_sec"]),
        mode=str(payload["mode"]),
        reason=str(payload.get("reason", "")),
        yaw_rate_deg_s=float(payload["yaw_rate_deg_s"]),
        camera_pitch_rate_deg_s=float(payload["camera_pitch_rate_deg_s"]),
        forward_mps=float(payload["forward_mps"]),
        normalized_offset=offset,
        center_error=None if payload.get("center_error") is None else float(payload["center_error"]),
        tag_enabled=bool(payload.get("tag_enabled", False)),
        box_area_ratio=None if payload.get("box_area_ratio") is None else float(payload["box_area_ratio"]),
    )


def load_control_intents(path: Path) -> list[ControlIntent]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("intents", payload.get("control_intents", []))
    if not isinstance(rows, list):
        raise ValueError("Expected 'intents' or 'control_intents' to be a list.")
    return [control_intent_from_dict(row) for row in rows if isinstance(row, dict)]


def setpoint_from_intent(intent: ControlIntent) -> BridgeSetpoint:
    return BridgeSetpoint(
        frame_index=intent.frame_index,
        timestamp_sec=intent.timestamp_sec,
        mode=intent.mode,
        reason=intent.reason,
        yaw_rate_deg_s=intent.yaw_rate_deg_s,
        camera_pitch_rate_deg_s=intent.camera_pitch_rate_deg_s,
        forward_mps=intent.forward_mps,
        center_error=intent.center_error,
        tag_enabled=intent.tag_enabled,
    )


def timing_gaps(intents: list[ControlIntent], config: BridgeConfig) -> list[dict[str, object]]:
    gaps: list[dict[str, object]] = []
    for previous, current in zip(intents, intents[1:]):
        gap = current.timestamp_sec - previous.timestamp_sec
        if gap > config.max_setpoint_gap_sec:
            gaps.append(
                {
                    "previous_frame_index": previous.frame_index,
                    "current_frame_index": current.frame_index,
                    "gap_sec": gap,
                }
            )
    return gaps


def run_bridge_session(
    intents: list[ControlIntent],
    bridge: DroneBridge,
    *,
    bridge_config: BridgeConfig = BridgeConfig(),
    safety_limits: SafetyLimits = SafetyLimits(),
) -> dict[str, object]:
    if not intents:
        return {
            "config": asdict(bridge_config),
            "safety_limits": asdict(safety_limits),
            "summary": {
                "input_intents": 0,
                "sent_setpoints": 0,
                "forward_setpoints": 0,
                "tag_setpoints": 0,
                "timing_gap_count": 0,
                "max_timing_gap_sec": 0.0,
                "ready_for_sitl": False,
            },
            "events": [],
            "setpoints": [],
            "timing_gaps": [],
        }

    bridge.arm()
    bridge.set_mode_offboard()

    filtered_intents = [safety_filter_intent(intent, safety_limits) for intent in intents]
    for intent in filtered_intents:
        bridge.send_setpoint(setpoint_from_intent(intent))
    bridge.stop()

    gaps = timing_gaps(filtered_intents, bridge_config)
    gap_values = [float(gap["gap_sec"]) for gap in gaps]
    setpoints = getattr(bridge, "setpoints", [])
    events = getattr(bridge, "events", [])
    ready_for_sitl = len(setpoints) >= bridge_config.offboard_warmup_setpoints and not gaps
    return {
        "config": asdict(bridge_config),
        "safety_limits": asdict(safety_limits),
        "summary": {
            "input_intents": len(intents),
            "sent_setpoints": len(setpoints),
            "forward_setpoints": sum(1 for setpoint in setpoints if setpoint.forward_mps > 0),
            "tag_setpoints": sum(1 for setpoint in setpoints if setpoint.mode == "tag"),
            "timing_gap_count": len(gaps),
            "max_timing_gap_sec": max(gap_values, default=0.0),
            "ready_for_sitl": ready_for_sitl,
        },
        "events": events,
        "setpoints": [setpoint.to_dict() for setpoint in setpoints],
        "timing_gaps": gaps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay control intents through the mock drone bridge.")
    parser.add_argument("intent_file", type=Path, help="control_intent.json or live_session.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for bridge_session.json.")
    parser.add_argument("--max-setpoint-gap-sec", type=float, default=0.5)
    parser.add_argument("--offboard-warmup-setpoints", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bridge_config = BridgeConfig(
        max_setpoint_gap_sec=args.max_setpoint_gap_sec,
        offboard_warmup_setpoints=args.offboard_warmup_setpoints,
    )
    report = run_bridge_session(
        load_control_intents(args.intent_file),
        MockPX4Bridge(),
        bridge_config=bridge_config,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / "bridge_session.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"bridge_session: {output_path}")
    print(f"ready_for_sitl: {report['summary']['ready_for_sitl']}")
    print(f"sent_setpoints: {report['summary']['sent_setpoints']}")
    print(f"timing_gap_count: {report['summary']['timing_gap_count']}")


if __name__ == "__main__":
    main()
