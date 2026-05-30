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


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float = 0.0

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> Vec3:
        return Vec3(self.x * factor, self.y * factor, self.z * factor)

    @property
    def length_xy(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass(frozen=True)
class DroneState:
    position: Vec3
    yaw_rad: float
    camera_pitch_rad: float


@dataclass(frozen=True)
class TargetState:
    position: Vec3
    heading_rad: float
    speed_mps: float


@dataclass(frozen=True)
class CameraObservation:
    visible: bool
    norm_x: float | None
    norm_y: float | None
    box_size: float | None
    distance_m: float | None
    confidence: float


@dataclass(frozen=True)
class CandidateObservation:
    object_id: str
    observation: CameraObservation
    ground_position: Vec3
    appearance: tuple[float, float, float]
    is_target: bool = False


@dataclass(frozen=True)
class ControlCommand:
    yaw_rate_rad_s: float
    camera_pitch_rate_rad_s: float
    forward_mps: float
    lateral_mps: float
    reason: str


@dataclass(frozen=True)
class SimConfig:
    duration_sec: float = 30.0
    dt_sec: float = 0.1
    latency_ms: float = 250.0
    seed: int = 7
    scenario: str = "wander"
    target_speed_mps: float = 2.5
    target_turn_noise_deg_s: float = 40.0
    target_bounds_m: float = 80.0
    drone_altitude_m: float = 45.0
    drone_start_x_m: float = -35.0
    drone_start_y_m: float = -35.0
    camera_pitch_deg: float = -45.0
    horizontal_fov_deg: float = 70.0
    vertical_fov_deg: float = 45.0
    desired_box_size: float = 0.16
    yaw_gain: float = 1.0
    pitch_gain: float = 0.9
    forward_gain: float = 35.0
    lateral_gain: float = 3.0
    max_yaw_rate_deg_s: float = 35.0
    max_pitch_rate_deg_s: float = 25.0
    max_forward_mps: float = 10.0
    max_lateral_mps: float = 4.0
    min_camera_pitch_deg: float = -85.0
    max_camera_pitch_deg: float = -20.0
    prediction_gain: float = 1.0
    max_prediction_sec: float = 0.6
    reacquire_timeout_sec: float = 2.0
    intercept_enabled: bool = True
    intercept_base_distance_m: float = 25.0
    intercept_speed_horizon_sec: float = 3.0
    intercept_lookahead_sec: float = 1.2
    intercept_delay_sec: float = 0.6
    intercept_yaw_gain: float = 2.0
    intercept_pitch_gain: float = 1.2
    intercept_forward_mps: float = 8.0
    intercept_velocity_smoothing: float = 0.35
    max_estimated_target_speed_mps: float = 18.0
    redetect_enabled: bool = True
    redetect_fov_multiplier: float = 1.8
    redetect_min_score: float = 0.68
    redetect_motion_weight: float = 0.55
    redetect_appearance_weight: float = 0.35
    redetect_confidence_weight: float = 0.10
    redetect_lookahead_sec: float = 1.0
    distractor_count: int = 5
    tracking_noise: float = 0.015
    confidence_noise: float = 0.06
    lost_confidence: float = 0.2

    @property
    def steps(self) -> int:
        return max(1, int(round(self.duration_sec / self.dt_sec)))

    @property
    def latency_steps(self) -> int:
        return max(0, int(round((self.latency_ms / 1000) / self.dt_sec)))


SCENARIOS = frozenset({"wander", "swerve", "sharp_turns", "fast_break"})
TARGET_APPEARANCE = (0.88, 0.22, 0.18)
DISTRACTOR_APPEARANCES = (
    (0.20, 0.58, 0.86),
    (0.72, 0.72, 0.20),
    (0.30, 0.78, 0.34),
    (0.70, 0.28, 0.78),
    (0.86, 0.54, 0.22),
    (0.42, 0.42, 0.45),
)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def dot(first: Vec3, second: Vec3) -> float:
    return first.x * second.x + first.y * second.y + first.z * second.z


def cross(first: Vec3, second: Vec3) -> Vec3:
    return Vec3(
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x,
    )


def normalize(vector: Vec3) -> Vec3:
    length = math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
    if length <= 1e-9:
        return Vec3(0.0, 0.0, 0.0)
    return vector.scale(1 / length)


def camera_basis(yaw_rad: float, pitch_rad: float) -> tuple[Vec3, Vec3, Vec3]:
    forward = normalize(
        Vec3(
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.cos(pitch_rad) * math.sin(yaw_rad),
            math.sin(pitch_rad),
        )
    )
    right = normalize(Vec3(-math.sin(yaw_rad), math.cos(yaw_rad), 0.0))
    up = normalize(cross(right, forward))
    return forward, right, up


def observe_target(
    drone: DroneState,
    target: TargetState,
    config: SimConfig,
    rng: random.Random,
) -> CameraObservation:
    return observe_position(drone, target.position, config, rng, fov_multiplier=1.0)


def observe_position(
    drone: DroneState,
    position: Vec3,
    config: SimConfig,
    rng: random.Random,
    *,
    fov_multiplier: float = 1.0,
) -> CameraObservation:
    forward, right, up = camera_basis(drone.yaw_rad, drone.camera_pitch_rad)
    relative = position - drone.position
    depth = dot(relative, forward)
    if depth <= 0:
        return CameraObservation(False, None, None, None, None, 0.0)

    cam_x = dot(relative, right) / depth
    cam_y = dot(relative, up) / depth
    half_h = math.tan(math.radians(config.horizontal_fov_deg) / 2)
    half_v = math.tan(math.radians(config.vertical_fov_deg) / 2)
    norm_x = cam_x / half_h
    norm_y = cam_y / half_v
    visible = abs(norm_x) <= fov_multiplier and abs(norm_y) <= fov_multiplier
    distance = math.sqrt(relative.x * relative.x + relative.y * relative.y + relative.z * relative.z)
    box_size = clamp(5.0 / max(distance, 1.0), 0.02, 0.8)

    if visible:
        norm_x = clamp(norm_x + rng.gauss(0, config.tracking_noise), -1.5, 1.5)
        norm_y = clamp(norm_y + rng.gauss(0, config.tracking_noise), -1.5, 1.5)
        edge_penalty = max(abs(norm_x), abs(norm_y)) * 0.25
        confidence = clamp(0.9 - edge_penalty + rng.gauss(0, config.confidence_noise), 0.0, 1.0)
    else:
        confidence = 0.0

    return CameraObservation(visible, norm_x if visible else None, norm_y if visible else None, box_size if visible else None, distance, confidence)


def noisy_appearance(appearance: tuple[float, float, float], rng: random.Random, *, noise: float = 0.035) -> tuple[float, float, float]:
    return tuple(clamp(channel + rng.gauss(0.0, noise), 0.0, 1.0) for channel in appearance)


def appearance_similarity(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    distance = math.sqrt(sum((a - b) * (a - b) for a, b in zip(first, second)))
    return clamp(1.0 - distance / math.sqrt(3.0), 0.0, 1.0)


def ray_from_observation(drone: DroneState, observation: CameraObservation, config: SimConfig) -> Vec3 | None:
    if not observation.visible or observation.norm_x is None or observation.norm_y is None:
        return None
    forward, right, up = camera_basis(drone.yaw_rad, drone.camera_pitch_rad)
    half_h = math.tan(math.radians(config.horizontal_fov_deg) / 2)
    half_v = math.tan(math.radians(config.vertical_fov_deg) / 2)
    return normalize(
        forward
        + right.scale(observation.norm_x * half_h)
        + up.scale(observation.norm_y * half_v)
    )


def estimate_ground_position_from_observation(
    drone: DroneState,
    observation: CameraObservation,
    config: SimConfig,
) -> Vec3 | None:
    ray = ray_from_observation(drone, observation, config)
    if ray is None or abs(ray.z) <= 1e-9:
        return None
    scale = -drone.position.z / ray.z
    if scale <= 0:
        return None
    return drone.position + ray.scale(scale)


def predicted_axis(
    current: float,
    previous: float | None,
    *,
    dt_sec: float,
    config: SimConfig,
) -> float:
    if previous is None or dt_sec <= 0:
        return current
    velocity = (current - previous) / dt_sec
    horizon = min(config.max_prediction_sec, (config.latency_ms / 1000) * config.prediction_gain)
    return clamp(current + velocity * horizon, -1.5, 1.5)


def command_from_screen_error(
    *,
    norm_x: float,
    norm_y: float,
    box_size: float,
    config: SimConfig,
    reason: str,
) -> ControlCommand:
    yaw_rate = clamp(
        config.yaw_gain * norm_x,
        -math.radians(config.max_yaw_rate_deg_s),
        math.radians(config.max_yaw_rate_deg_s),
    )
    camera_pitch_rate = clamp(
        -config.pitch_gain * norm_y,
        -math.radians(config.max_pitch_rate_deg_s),
        math.radians(config.max_pitch_rate_deg_s),
    )
    size_error = config.desired_box_size - box_size
    forward = clamp(
        config.forward_gain * size_error,
        -config.max_forward_mps,
        config.max_forward_mps,
    )
    lateral = clamp(
        config.lateral_gain * norm_x,
        -config.max_lateral_mps,
        config.max_lateral_mps,
    )
    return ControlCommand(yaw_rate, camera_pitch_rate, forward, lateral, reason)


def command_from_observation(
    observation: CameraObservation,
    config: SimConfig,
    *,
    previous_observation: CameraObservation | None = None,
    dt_sec: float | None = None,
) -> ControlCommand:
    if not observation.visible or observation.confidence < config.lost_confidence:
        return ControlCommand(0.0, 0.0, 0.0, 0.0, "hold_lost_target")

    assert observation.norm_x is not None
    assert observation.norm_y is not None
    assert observation.box_size is not None
    previous_x = None
    previous_y = None
    if (
        previous_observation is not None
        and previous_observation.visible
        and previous_observation.norm_x is not None
        and previous_observation.norm_y is not None
    ):
        previous_x = previous_observation.norm_x
        previous_y = previous_observation.norm_y
    predicted_x = predicted_axis(observation.norm_x, previous_x, dt_sec=dt_sec or config.dt_sec, config=config)
    predicted_y = predicted_axis(observation.norm_y, previous_y, dt_sec=dt_sec or config.dt_sec, config=config)
    return command_from_screen_error(
        norm_x=predicted_x,
        norm_y=predicted_y,
        box_size=observation.box_size,
        config=config,
        reason="track",
    )


def command_from_lost_prediction(
    *,
    last_norm_x: float | None,
    last_norm_y: float | None,
    last_box_size: float | None,
    velocity_x: float,
    velocity_y: float,
    lost_time_sec: float,
    config: SimConfig,
) -> ControlCommand:
    if (
        last_norm_x is None
        or last_norm_y is None
        or last_box_size is None
        or lost_time_sec > config.reacquire_timeout_sec
    ):
        return ControlCommand(0.0, 0.0, 0.0, 0.0, "hold_lost_target")

    prediction_horizon = min(config.max_prediction_sec, lost_time_sec + (config.latency_ms / 1000) * config.prediction_gain)
    predicted_x = clamp(last_norm_x + velocity_x * prediction_horizon, -1.5, 1.5)
    predicted_y = clamp(last_norm_y + velocity_y * prediction_horizon, -1.5, 1.5)
    return command_from_screen_error(
        norm_x=predicted_x,
        norm_y=predicted_y,
        box_size=last_box_size,
        config=config,
        reason="reacquire_prediction",
    )


def command_from_intercept(
    *,
    drone: DroneState,
    estimated_target_position: Vec3 | None,
    estimated_target_velocity: Vec3,
    estimated_target_speed_mps: float,
    lost_time_sec: float,
    config: SimConfig,
) -> ControlCommand | None:
    if not config.intercept_enabled or estimated_target_position is None:
        return None
    if lost_time_sec < config.intercept_delay_sec or lost_time_sec > config.reacquire_timeout_sec:
        return None

    lookahead = min(config.intercept_lookahead_sec + lost_time_sec, config.reacquire_timeout_sec)
    predicted_position = estimated_target_position + estimated_target_velocity.scale(lookahead)
    to_target = predicted_position - drone.position
    horizontal_distance = math.hypot(to_target.x, to_target.y)
    close_enough = config.intercept_base_distance_m + estimated_target_speed_mps * config.intercept_speed_horizon_sec
    if horizontal_distance > close_enough:
        return None

    target_bearing = math.atan2(to_target.y, to_target.x)
    yaw_error = wrap_angle(target_bearing - drone.yaw_rad)
    yaw_rate = clamp(
        config.intercept_yaw_gain * yaw_error,
        -math.radians(config.max_yaw_rate_deg_s),
        math.radians(config.max_yaw_rate_deg_s),
    )
    desired_pitch = math.atan2(-drone.position.z, max(horizontal_distance, 1.0))
    pitch_error = desired_pitch - drone.camera_pitch_rad
    camera_pitch_rate = clamp(
        config.intercept_pitch_gain * pitch_error,
        -math.radians(config.max_pitch_rate_deg_s),
        math.radians(config.max_pitch_rate_deg_s),
    )
    forward = clamp(config.intercept_forward_mps, 0.0, config.max_forward_mps)
    return ControlCommand(yaw_rate, camera_pitch_rate, forward, 0.0, "lost_intercept")


def scenario_turn_rate_deg_s(step: int, target: TargetState, config: SimConfig, rng: random.Random) -> float:
    if config.scenario == "wander":
        return rng.uniform(-config.target_turn_noise_deg_s, config.target_turn_noise_deg_s)
    if config.scenario == "swerve":
        period = max(1, int(round(2.0 / config.dt_sec)))
        direction = 1 if (step // period) % 2 == 0 else -1
        return direction * config.target_turn_noise_deg_s * 1.8
    if config.scenario == "sharp_turns":
        interval = max(1, int(round(4.0 / config.dt_sec)))
        if step % interval < max(1, int(round(0.8 / config.dt_sec))):
            return config.target_turn_noise_deg_s * 3.0
        return rng.uniform(-config.target_turn_noise_deg_s * 0.25, config.target_turn_noise_deg_s * 0.25)
    if config.scenario == "fast_break":
        return config.target_turn_noise_deg_s * math.sin(step * config.dt_sec * 3.0)
    raise ValueError(f"Unknown scenario: {config.scenario}")


def scenario_speed_mps(step: int, config: SimConfig) -> float:
    if config.scenario == "fast_break":
        return config.target_speed_mps * (1.0 + 0.5 * max(0.0, math.sin(step * config.dt_sec * 1.5)))
    return config.target_speed_mps


def step_target(target: TargetState, config: SimConfig, rng: random.Random, *, step: int) -> TargetState:
    speed = scenario_speed_mps(step, config)
    heading = wrap_angle(target.heading_rad + math.radians(scenario_turn_rate_deg_s(step, target, config, rng)) * config.dt_sec)
    position = Vec3(
        target.position.x + math.cos(heading) * speed * config.dt_sec,
        target.position.y + math.sin(heading) * speed * config.dt_sec,
        0.0,
    )
    bounds = config.target_bounds_m
    if abs(position.x) > bounds:
        heading = wrap_angle(math.pi - heading)
        position = Vec3(clamp(position.x, -bounds, bounds), position.y, 0.0)
    if abs(position.y) > bounds:
        heading = wrap_angle(-heading)
        position = Vec3(position.x, clamp(position.y, -bounds, bounds), 0.0)
    return TargetState(position=position, heading_rad=heading, speed_mps=speed)


def build_distractors(config: SimConfig) -> list[TargetState]:
    distractors: list[TargetState] = []
    count = max(0, config.distractor_count)
    for index in range(count):
        angle = (index / max(1, count)) * 2 * math.pi
        radius = 18.0 + 8.0 * (index % 3)
        distractors.append(
            TargetState(
                position=Vec3(math.cos(angle) * radius, math.sin(angle) * radius, 0.0),
                heading_rad=wrap_angle(angle + math.pi / 2),
                speed_mps=max(0.5, config.target_speed_mps * (0.45 + 0.08 * (index % 4))),
            )
        )
    return distractors


def step_distractor(distractor: TargetState, config: SimConfig, *, index: int, step: int) -> TargetState:
    turn_rate = math.sin(step * config.dt_sec * (0.8 + index * 0.07)) * 18.0
    heading = wrap_angle(distractor.heading_rad + math.radians(turn_rate) * config.dt_sec)
    position = Vec3(
        distractor.position.x + math.cos(heading) * distractor.speed_mps * config.dt_sec,
        distractor.position.y + math.sin(heading) * distractor.speed_mps * config.dt_sec,
        0.0,
    )
    bounds = config.target_bounds_m
    if abs(position.x) > bounds:
        heading = wrap_angle(math.pi - heading)
        position = Vec3(clamp(position.x, -bounds, bounds), position.y, 0.0)
    if abs(position.y) > bounds:
        heading = wrap_angle(-heading)
        position = Vec3(position.x, clamp(position.y, -bounds, bounds), 0.0)
    return TargetState(position=position, heading_rad=heading, speed_mps=distractor.speed_mps)


def candidate_observations(
    *,
    drone: DroneState,
    target: TargetState,
    distractors: list[TargetState],
    config: SimConfig,
    rng: random.Random,
) -> list[CandidateObservation]:
    candidates: list[CandidateObservation] = []
    actors = [("target", target, TARGET_APPEARANCE, True)]
    actors.extend(
        (
            f"distractor_{index}",
            distractor,
            DISTRACTOR_APPEARANCES[index % len(DISTRACTOR_APPEARANCES)],
            False,
        )
        for index, distractor in enumerate(distractors)
    )
    for object_id, actor, appearance, is_target in actors:
        observation = observe_position(
            drone,
            actor.position,
            config,
            rng,
            fov_multiplier=config.redetect_fov_multiplier,
        )
        if observation.visible and observation.confidence >= config.lost_confidence:
            candidates.append(
                CandidateObservation(
                    object_id=object_id,
                    observation=observation,
                    ground_position=actor.position,
                    appearance=noisy_appearance(appearance, rng),
                    is_target=is_target,
                )
            )
    return candidates


def redetection_score(
    candidate: CandidateObservation,
    *,
    predicted_position: Vec3,
    search_radius_m: float,
    target_appearance: tuple[float, float, float],
    config: SimConfig,
) -> float:
    distance = (candidate.ground_position - predicted_position).length_xy
    motion_score = clamp(1.0 - distance / max(search_radius_m, 1.0), 0.0, 1.0)
    visual_score = appearance_similarity(target_appearance, candidate.appearance)
    confidence_score = candidate.observation.confidence
    return (
        config.redetect_motion_weight * motion_score
        + config.redetect_appearance_weight * visual_score
        + config.redetect_confidence_weight * confidence_score
    )


def select_redetection_candidate(
    candidates: list[CandidateObservation],
    *,
    estimated_target_position: Vec3 | None,
    estimated_target_velocity: Vec3,
    estimated_target_speed_mps: float,
    target_appearance: tuple[float, float, float],
    lost_time_sec: float,
    config: SimConfig,
) -> tuple[CandidateObservation | None, float]:
    if not config.redetect_enabled or estimated_target_position is None or not candidates:
        return None, 0.0
    lookahead = min(config.redetect_lookahead_sec + lost_time_sec, config.reacquire_timeout_sec)
    predicted_position = estimated_target_position + estimated_target_velocity.scale(lookahead)
    search_radius = (
        config.intercept_base_distance_m
        + estimated_target_speed_mps * config.intercept_speed_horizon_sec
        + lost_time_sec * max(estimated_target_speed_mps, 1.0)
    )
    scored = [
        (
            redetection_score(
                candidate,
                predicted_position=predicted_position,
                search_radius_m=search_radius,
                target_appearance=target_appearance,
                config=config,
            ),
            candidate,
        )
        for candidate in candidates
    ]
    best_score, best_candidate = max(scored, key=lambda item: item[0])
    if best_score < config.redetect_min_score:
        return None, best_score
    return best_candidate, best_score


def step_drone(drone: DroneState, command: ControlCommand, config: SimConfig) -> DroneState:
    yaw = wrap_angle(drone.yaw_rad + command.yaw_rate_rad_s * config.dt_sec)
    camera_pitch = clamp(
        drone.camera_pitch_rad + command.camera_pitch_rate_rad_s * config.dt_sec,
        math.radians(config.min_camera_pitch_deg),
        math.radians(config.max_camera_pitch_deg),
    )
    forward = Vec3(math.cos(yaw), math.sin(yaw), 0.0)
    right = Vec3(-math.sin(yaw), math.cos(yaw), 0.0)
    delta = forward.scale(command.forward_mps * config.dt_sec) + right.scale(command.lateral_mps * config.dt_sec)
    return DroneState(
        position=Vec3(drone.position.x + delta.x, drone.position.y + delta.y, config.drone_altitude_m),
        yaw_rad=yaw,
        camera_pitch_rad=camera_pitch,
    )


def run_simulation(config: SimConfig) -> dict[str, object]:
    if config.scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {config.scenario}. Available scenarios: {', '.join(sorted(SCENARIOS))}.")

    motion_rng = random.Random(config.seed)
    perception_rng = random.Random(config.seed + 10_000)
    drone = DroneState(
        position=Vec3(config.drone_start_x_m, config.drone_start_y_m, config.drone_altitude_m),
        yaw_rad=math.radians(45.0),
        camera_pitch_rad=math.radians(config.camera_pitch_deg),
    )
    target = TargetState(position=Vec3(0.0, 0.0, 0.0), heading_rad=math.radians(25.0), speed_mps=config.target_speed_mps)
    distractors = build_distractors(config)
    hold = ControlCommand(0.0, 0.0, 0.0, 0.0, "startup_latency")
    command_buffer = [hold for _ in range(config.latency_steps + 1)]
    rows: list[dict[str, object]] = []
    previous_observation: CameraObservation | None = None
    last_norm_x: float | None = None
    last_norm_y: float | None = None
    last_box_size: float | None = None
    image_velocity_x = 0.0
    image_velocity_y = 0.0
    last_estimated_target_position: Vec3 | None = None
    estimated_target_velocity = Vec3(0.0, 0.0, 0.0)
    estimated_target_speed_mps = 0.0
    target_appearance = TARGET_APPEARANCE
    lost_steps = 0

    for step in range(config.steps):
        time_sec = step * config.dt_sec
        observation = observe_target(drone, target, config, perception_rng)
        selected_candidate_id: str | None = None
        selected_candidate_score = 0.0
        redetected = False
        false_redetect = False
        if observation.visible and observation.confidence >= config.lost_confidence:
            estimated_position = estimate_ground_position_from_observation(drone, observation, config)
            if estimated_position is not None:
                if last_estimated_target_position is not None and config.dt_sec > 0:
                    instant_velocity = Vec3(
                        (estimated_position.x - last_estimated_target_position.x) / config.dt_sec,
                        (estimated_position.y - last_estimated_target_position.y) / config.dt_sec,
                        0.0,
                    )
                    instant_speed = instant_velocity.length_xy
                    if instant_speed > config.max_estimated_target_speed_mps:
                        instant_velocity = instant_velocity.scale(config.max_estimated_target_speed_mps / instant_speed)
                    alpha = clamp(config.intercept_velocity_smoothing, 0.0, 1.0)
                    estimated_target_velocity = estimated_target_velocity.scale(1 - alpha) + instant_velocity.scale(alpha)
                    estimated_target_speed_mps = estimated_target_velocity.length_xy
                last_estimated_target_position = estimated_position
                target_appearance = tuple(
                    clamp(previous * 0.95 + current * 0.05, 0.0, 1.0)
                    for previous, current in zip(
                        target_appearance,
                        noisy_appearance(TARGET_APPEARANCE, perception_rng, noise=0.015),
                    )
                )
            if last_norm_x is not None and last_norm_y is not None and observation.norm_x is not None and observation.norm_y is not None:
                image_velocity_x = (observation.norm_x - last_norm_x) / config.dt_sec
                image_velocity_y = (observation.norm_y - last_norm_y) / config.dt_sec
            command = command_from_observation(
                observation,
                config,
                previous_observation=previous_observation,
                dt_sec=config.dt_sec,
            )
            last_norm_x = observation.norm_x
            last_norm_y = observation.norm_y
            last_box_size = observation.box_size
            lost_steps = 0
        else:
            lost_steps += 1
            lost_time_sec = lost_steps * config.dt_sec
            candidate, selected_candidate_score = select_redetection_candidate(
                candidate_observations(
                    drone=drone,
                    target=target,
                    distractors=distractors,
                    config=config,
                    rng=perception_rng,
                ),
                estimated_target_position=last_estimated_target_position,
                estimated_target_velocity=estimated_target_velocity,
                estimated_target_speed_mps=estimated_target_speed_mps,
                target_appearance=target_appearance,
                lost_time_sec=lost_time_sec,
                config=config,
            )
            if candidate is not None:
                redetected = True
                false_redetect = not candidate.is_target
                selected_candidate_id = candidate.object_id
                observation = candidate.observation
                if candidate.observation.norm_x is not None and candidate.observation.norm_y is not None:
                    command = command_from_screen_error(
                        norm_x=clamp(candidate.observation.norm_x, -1.5, 1.5),
                        norm_y=clamp(candidate.observation.norm_y, -1.5, 1.5),
                        box_size=candidate.observation.box_size or last_box_size or 0.08,
                        config=config,
                        reason="redetect",
                    )
                else:
                    command = ControlCommand(0.0, 0.0, 0.0, 0.0, "hold_lost_target")
                last_estimated_target_position = candidate.ground_position
                last_norm_x = candidate.observation.norm_x
                last_norm_y = candidate.observation.norm_y
                last_box_size = candidate.observation.box_size
                if candidate.is_target:
                    lost_steps = 0
            else:
                command = command_from_intercept(
                    drone=drone,
                    estimated_target_position=last_estimated_target_position,
                    estimated_target_velocity=estimated_target_velocity,
                    estimated_target_speed_mps=estimated_target_speed_mps,
                    lost_time_sec=lost_time_sec,
                    config=config,
                )
            if command is None:
                command = command_from_lost_prediction(
                    last_norm_x=last_norm_x,
                    last_norm_y=last_norm_y,
                    last_box_size=last_box_size,
                    velocity_x=image_velocity_x,
                    velocity_y=image_velocity_y,
                    lost_time_sec=lost_time_sec,
                    config=config,
                )
        command_buffer.append(command)
        delayed_command = command_buffer.pop(0)
        drone = step_drone(drone, delayed_command, config)
        target = step_target(target, config, motion_rng, step=step)
        distractors = [
            step_distractor(distractor, config, index=index, step=step)
            for index, distractor in enumerate(distractors)
        ]

        error = None
        if observation.visible and not false_redetect and observation.norm_x is not None and observation.norm_y is not None:
            error = math.hypot(observation.norm_x, observation.norm_y)

        rows.append(
            {
                "step": step,
                "time_sec": time_sec,
                "drone_x": drone.position.x,
                "drone_y": drone.position.y,
                "drone_z": drone.position.z,
                "drone_yaw_deg": math.degrees(drone.yaw_rad),
                "camera_pitch_deg": math.degrees(drone.camera_pitch_rad),
                "target_x": target.position.x,
                "target_y": target.position.y,
                "target_z": target.position.z,
                "visible": observation.visible and not false_redetect,
                "redetected": redetected,
                "false_redetect": false_redetect,
                "selected_candidate_id": selected_candidate_id,
                "selected_candidate_score": selected_candidate_score,
                "norm_x": observation.norm_x,
                "norm_y": observation.norm_y,
                "box_size": observation.box_size,
                "confidence": observation.confidence,
                "screen_error": error,
                "yaw_rate_deg_s": math.degrees(delayed_command.yaw_rate_rad_s),
                "camera_pitch_rate_deg_s": math.degrees(delayed_command.camera_pitch_rate_rad_s),
                "forward_mps": delayed_command.forward_mps,
                "lateral_mps": delayed_command.lateral_mps,
                "command_reason": delayed_command.reason,
                "lost_steps": lost_steps,
                "image_velocity_x": image_velocity_x,
                "image_velocity_y": image_velocity_y,
                "estimated_target_x": last_estimated_target_position.x if last_estimated_target_position else None,
                "estimated_target_y": last_estimated_target_position.y if last_estimated_target_position else None,
                "estimated_target_speed_mps": estimated_target_speed_mps,
                "distance_m": observation.distance_m,
            }
        )
        previous_observation = observation

    errors = [float(row["screen_error"]) for row in rows if row["screen_error"] is not None]
    visible_count = sum(1 for row in rows if row["visible"])
    lost_count = len(rows) - visible_count
    redetect_count = sum(1 for row in rows if row["redetected"])
    false_redetect_count = sum(1 for row in rows if row["false_redetect"])
    avg_error = sum(errors) / len(errors) if errors else None
    max_error = max(errors) if errors else None
    command_changes = [
        abs(float(rows[index]["yaw_rate_deg_s"]) - float(rows[index - 1]["yaw_rate_deg_s"]))
        for index in range(1, len(rows))
    ]
    pitch_command_changes = [
        abs(float(rows[index]["camera_pitch_rate_deg_s"]) - float(rows[index - 1]["camera_pitch_rate_deg_s"]))
        for index in range(1, len(rows))
    ]
    reacquisition = reacquisition_metrics(rows, dt_sec=config.dt_sec)
    report = {
        "config": asdict(config),
        "summary": {
            "steps": len(rows),
            "duration_sec": config.duration_sec,
            "visible_frames": visible_count,
            "lost_frames": lost_count,
            "redetect_frames": redetect_count,
            "false_redetect_frames": false_redetect_count,
            "visible_ratio": visible_count / len(rows) if rows else 0.0,
            "average_screen_error": avg_error,
            "max_screen_error": max_error,
            "average_yaw_rate_change_deg_s": sum(command_changes) / len(command_changes) if command_changes else 0.0,
            "average_pitch_rate_change_deg_s": sum(pitch_command_changes) / len(pitch_command_changes) if pitch_command_changes else 0.0,
            "final_distance_m": rows[-1]["distance_m"] if rows else None,
            **reacquisition,
        },
        "trajectory": rows,
    }
    report["summary"]["grade"] = scenario_grade(report["summary"])
    return report


def lost_streak_lengths(rows: list[dict[str, object]]) -> list[int]:
    streaks: list[int] = []
    current = 0
    for row in rows:
        if row["visible"]:
            if current:
                streaks.append(current)
                current = 0
        else:
            current += 1
    if current:
        streaks.append(current)
    return streaks


def reacquisition_metrics(rows: list[dict[str, object]], *, dt_sec: float) -> dict[str, object]:
    streaks = lost_streak_lengths(rows)
    ended_lost = bool(rows and not rows[-1]["visible"])
    reacquired_streaks = streaks[:-1] if ended_lost else streaks
    reacquisition_times = [streak * dt_sec for streak in reacquired_streaks]
    return {
        "lost_events": len(streaks),
        "reacquired_count": len(reacquired_streaks),
        "failed_reacquisition_count": 1 if ended_lost else 0,
        "longest_lost_streak_frames": max(streaks, default=0),
        "longest_lost_streak_sec": max(streaks, default=0) * dt_sec,
        "average_reacquisition_time_sec": sum(reacquisition_times) / len(reacquisition_times) if reacquisition_times else 0.0,
    }


def scenario_grade(summary: dict[str, object]) -> str:
    visible_ratio = float(summary["visible_ratio"])
    average_error = summary["average_screen_error"]
    longest_lost_sec = float(summary.get("longest_lost_streak_sec", 0.0))
    failed_reacquisitions = int(summary.get("failed_reacquisition_count", 0))
    if (
        visible_ratio >= 0.90
        and average_error is not None
        and float(average_error) <= 0.35
        and longest_lost_sec <= 1.0
        and failed_reacquisitions == 0
    ):
        return "stable"
    if (
        visible_ratio >= 0.70
        and average_error is not None
        and float(average_error) <= 0.65
        and longest_lost_sec <= 3.0
    ):
        return "marginal"
    return "failed"


def prepare_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def write_trajectory_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_top_down(rows: list[dict[str, object]], output_path: Path, *, size: int = 900, padding: int = 50) -> None:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    points = []
    for row in rows:
        points.append((float(row["drone_x"]), float(row["drone_y"])))
        points.append((float(row["target_x"]), float(row["target_y"])))
    if not points:
        image.save(output_path)
        return

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    span = max(max_x - min_x, max_y - min_y, 1.0)

    def project(x: float, y: float) -> tuple[int, int]:
        px = padding + int((x - min_x) / span * (size - 2 * padding))
        py = size - padding - int((y - min_y) / span * (size - 2 * padding))
        return px, py

    drone_points = [project(float(row["drone_x"]), float(row["drone_y"])) for row in rows]
    target_points = [project(float(row["target_x"]), float(row["target_y"])) for row in rows]
    if len(target_points) > 1:
        draw.line(target_points, fill=(220, 60, 60), width=3)
    if len(drone_points) > 1:
        draw.line(drone_points, fill=(40, 90, 220), width=3)

    draw.ellipse((*offset_point(target_points[0], -5), *offset_point(target_points[0], 5)), fill=(220, 60, 60))
    draw.ellipse((*offset_point(drone_points[0], -5), *offset_point(drone_points[0], 5)), fill=(40, 90, 220))
    draw.rectangle((padding, padding, size - padding, size - padding), outline=(210, 210, 210), width=1)
    draw.text((padding, 20), "blue=drone path  red=target path", fill=(30, 30, 30))
    image.save(output_path)


def world_projector(rows: list[dict[str, object]], *, width: int, height: int, padding: int) -> object:
    points = []
    for row in rows:
        points.append((float(row["drone_x"]), float(row["drone_y"])))
        points.append((float(row["target_x"]), float(row["target_y"])))
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    span = max(max_x - min_x, max_y - min_y, 1.0)

    def project(x: float, y: float) -> tuple[int, int]:
        px = padding + int((x - min_x) / span * (width - 2 * padding))
        py = height - padding - int((y - min_y) / span * (height - 2 * padding))
        return px, py

    return project


def camera_point(row: dict[str, object], *, left: int, top: int, width: int, height: int) -> tuple[int, int] | None:
    if not row["visible"] or row["norm_x"] is None or row["norm_y"] is None:
        return None
    norm_x = clamp(float(row["norm_x"]), -1.0, 1.0)
    norm_y = clamp(float(row["norm_y"]), -1.0, 1.0)
    x = left + int((norm_x + 1) * 0.5 * width)
    y = top + int((1 - (norm_y + 1) * 0.5) * height)
    return x, y


def draw_preview_frame(rows: list[dict[str, object]], frame_index: int, *, width: int = 1000, height: int = 560) -> Image.Image:
    image = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    row = rows[frame_index]

    map_left, map_top, map_width, map_height = 30, 50, 460, 460
    camera_left, camera_top, camera_width, camera_height = 530, 80, 420, 300
    project = world_projector(rows, width=map_width, height=map_height, padding=25)

    draw.text((30, 18), "Top-down world", fill=(20, 25, 35))
    draw.rectangle((map_left, map_top, map_left + map_width, map_top + map_height), outline=(185, 190, 200), width=1)
    past = rows[: frame_index + 1]
    target_points = [
        offset_point(project(float(item["target_x"]), float(item["target_y"])), 0)
        for item in past
    ]
    drone_points = [
        offset_point(project(float(item["drone_x"]), float(item["drone_y"])), 0)
        for item in past
    ]
    target_points = [(x + map_left, y + map_top) for x, y in target_points]
    drone_points = [(x + map_left, y + map_top) for x, y in drone_points]
    if len(target_points) > 1:
        draw.line(target_points, fill=(220, 60, 60), width=3)
    if len(drone_points) > 1:
        draw.line(drone_points, fill=(40, 90, 220), width=3)

    target_xy = target_points[-1]
    drone_xy = drone_points[-1]
    draw.ellipse((*offset_point(target_xy, -6), *offset_point(target_xy, 6)), fill=(220, 60, 60))
    draw.ellipse((*offset_point(drone_xy, -6), *offset_point(drone_xy, 6)), fill=(40, 90, 220))
    yaw = math.radians(float(row["drone_yaw_deg"]))
    heading_end = (drone_xy[0] + int(math.cos(yaw) * 28), drone_xy[1] - int(math.sin(yaw) * 28))
    draw.line((drone_xy, heading_end), fill=(30, 50, 130), width=3)
    draw.text((map_left, map_top + map_height + 10), "blue=drone  red=target", fill=(50, 55, 65))

    draw.text((camera_left, 48), "Simulated camera frame", fill=(20, 25, 35))
    draw.rectangle((camera_left, camera_top, camera_left + camera_width, camera_top + camera_height), fill=(35, 38, 42), outline=(185, 190, 200), width=2)
    center = (camera_left + camera_width // 2, camera_top + camera_height // 2)
    draw.line((center[0] - 20, center[1], center[0] + 20, center[1]), fill=(230, 230, 230), width=1)
    draw.line((center[0], center[1] - 20, center[0], center[1] + 20), fill=(230, 230, 230), width=1)

    point = camera_point(row, left=camera_left, top=camera_top, width=camera_width, height=camera_height)
    if point is not None:
        box_size = int(max(14, min(80, float(row["box_size"] or 0.08) * 260)))
        color = (80, 200, 255) if row.get("redetected") else (50, 230, 90)
        if float(row["confidence"]) < 0.45:
            color = (245, 180, 50)
        draw.rectangle(
            (point[0] - box_size, point[1] - box_size, point[0] + box_size, point[1] + box_size),
            outline=color,
            width=3,
        )
        draw.line((center, point), fill=color, width=2)
        if row.get("redetected"):
            draw.text((camera_left + 12, camera_top + 12), "REDETECTED", fill=color)
    else:
        draw.text((camera_left + 145, camera_top + 135), "TARGET LOST", fill=(255, 80, 80))

    metrics_y = camera_top + camera_height + 25
    draw.text((camera_left, metrics_y), f"time: {float(row['time_sec']):.1f}s", fill=(30, 35, 45))
    draw.text((camera_left, metrics_y + 24), f"visible: {row['visible']}  confidence: {float(row['confidence']):.2f}", fill=(30, 35, 45))
    error = row["screen_error"]
    error_text = "n/a" if error is None else f"{float(error):.3f}"
    draw.text((camera_left, metrics_y + 48), f"screen error: {error_text}", fill=(30, 35, 45))
    draw.text(
        (camera_left, metrics_y + 72),
        f"yaw: {float(row['yaw_rate_deg_s']):.1f} deg/s  pitch: {float(row['camera_pitch_rate_deg_s']):.1f} deg/s",
        fill=(30, 35, 45),
    )
    draw.text((camera_left, metrics_y + 96), f"forward: {float(row['forward_mps']):.1f} m/s", fill=(30, 35, 45))
    draw.text((camera_left, metrics_y + 120), f"mode: {row['command_reason']}", fill=(30, 35, 45))
    return image


def write_preview_gif(
    rows: list[dict[str, object]],
    output_path: Path,
    *,
    every_n: int = 3,
    frame_duration_ms: int = 80,
) -> None:
    if not rows:
        return
    frames = [
        draw_preview_frame(rows, index)
        for index in range(0, len(rows), max(1, every_n))
    ]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )


def offset_point(point: tuple[int, int], amount: int) -> tuple[int, int]:
    return point[0] + amount, point[1] + amount


def run_and_write(config: SimConfig, out_dir: Path) -> dict[str, object]:
    prepare_output_dir(out_dir)
    report = run_simulation(config)
    rows = report["trajectory"]
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_trajectory_csv(rows, out_dir / "trajectory.csv")
    draw_top_down(rows, out_dir / "top_down.png")
    write_preview_gif(rows, out_dir / "preview.gif")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a math-only 3D closed-loop drone target-follow simulator.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for report.json, trajectory.csv, top_down.png, and preview.gif.")
    parser.add_argument("--duration-sec", type=float, default=30.0, help="Simulation duration.")
    parser.add_argument("--dt-sec", type=float, default=0.1, help="Simulation timestep.")
    parser.add_argument("--latency-ms", type=float, default=250.0, help="Video/inference/command latency to simulate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="wander", help="Target motion scenario.")
    parser.add_argument("--target-speed-mps", type=float, default=2.5, help="Ground target speed.")
    parser.add_argument("--drone-altitude-m", type=float, default=45.0, help="Drone altitude.")
    parser.add_argument("--camera-pitch-deg", type=float, default=-45.0, help="Camera pitch angle; negative points downward.")
    parser.add_argument("--horizontal-fov-deg", type=float, default=70.0, help="Simulated camera horizontal field of view.")
    parser.add_argument("--vertical-fov-deg", type=float, default=45.0, help="Simulated camera vertical field of view.")
    parser.add_argument("--yaw-gain", type=float, default=1.0, help="Controller yaw gain.")
    parser.add_argument("--pitch-gain", type=float, default=0.9, help="Controller camera pitch gain.")
    parser.add_argument("--forward-gain", type=float, default=35.0, help="Controller forward gain.")
    parser.add_argument("--lateral-gain", type=float, default=3.0, help="Controller lateral gain.")
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=35.0, help="Maximum simulated drone yaw command.")
    parser.add_argument("--max-pitch-rate-deg-s", type=float, default=25.0, help="Maximum simulated gimbal pitch command.")
    parser.add_argument("--max-forward-mps", type=float, default=10.0, help="Maximum simulated drone forward speed.")
    parser.add_argument("--max-lateral-mps", type=float, default=4.0, help="Maximum simulated drone lateral speed.")
    parser.add_argument("--prediction-gain", type=float, default=1.0, help="Lead target screen position by this fraction of the latency.")
    parser.add_argument("--reacquire-timeout-sec", type=float, default=2.0, help="How long to steer using predicted target position after losing sight.")
    parser.add_argument("--no-intercept", action="store_true", help="Disable world-position lost-target intercept fallback.")
    parser.add_argument("--intercept-base-distance-m", type=float, default=25.0, help="Base distance for allowing lost-target intercept.")
    parser.add_argument("--intercept-speed-horizon-sec", type=float, default=3.0, help="Add target_speed * this horizon to the intercept distance gate.")
    parser.add_argument("--intercept-lookahead-sec", type=float, default=1.2, help="Predict this far ahead when steering toward a lost target.")
    parser.add_argument("--intercept-delay-sec", type=float, default=0.6, help="Wait this long after visual loss before using world-position intercept.")
    parser.add_argument("--intercept-yaw-gain", type=float, default=2.0, help="Yaw gain used by lost-target intercept.")
    parser.add_argument("--intercept-pitch-gain", type=float, default=1.2, help="Camera pitch gain used by lost-target intercept.")
    parser.add_argument("--intercept-forward-mps", type=float, default=8.0, help="Forward speed used by lost-target intercept.")
    parser.add_argument("--intercept-velocity-smoothing", type=float, default=0.35, help="Smoothing factor for estimated target velocity.")
    parser.add_argument("--max-estimated-target-speed-mps", type=float, default=18.0, help="Clamp noisy estimated target velocity before intercept.")
    parser.add_argument("--no-redetect", action="store_true", help="Disable appearance-aware lost-target re-detection.")
    parser.add_argument("--redetect-fov-multiplier", type=float, default=1.8, help="Search this many normal FOV widths/heights during re-detection.")
    parser.add_argument("--redetect-min-score", type=float, default=0.68, help="Minimum motion/appearance score to accept a re-detection.")
    parser.add_argument("--redetect-lookahead-sec", type=float, default=1.0, help="Predict this far ahead when scoring re-detection candidates.")
    parser.add_argument("--distractor-count", type=int, default=5, help="Number of simulated distractor objects for re-detection tests.")
    parser.add_argument("--tracking-noise", type=float, default=0.015, help="Normalized observation noise.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SimConfig(
        duration_sec=args.duration_sec,
        dt_sec=args.dt_sec,
        latency_ms=args.latency_ms,
        seed=args.seed,
        scenario=args.scenario,
        target_speed_mps=args.target_speed_mps,
        drone_altitude_m=args.drone_altitude_m,
        camera_pitch_deg=args.camera_pitch_deg,
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
        yaw_gain=args.yaw_gain,
        pitch_gain=args.pitch_gain,
        forward_gain=args.forward_gain,
        lateral_gain=args.lateral_gain,
        max_yaw_rate_deg_s=args.max_yaw_rate_deg_s,
        max_pitch_rate_deg_s=args.max_pitch_rate_deg_s,
        max_forward_mps=args.max_forward_mps,
        max_lateral_mps=args.max_lateral_mps,
        prediction_gain=args.prediction_gain,
        reacquire_timeout_sec=args.reacquire_timeout_sec,
        intercept_enabled=not args.no_intercept,
        intercept_base_distance_m=args.intercept_base_distance_m,
        intercept_speed_horizon_sec=args.intercept_speed_horizon_sec,
        intercept_lookahead_sec=args.intercept_lookahead_sec,
        intercept_delay_sec=args.intercept_delay_sec,
        intercept_yaw_gain=args.intercept_yaw_gain,
        intercept_pitch_gain=args.intercept_pitch_gain,
        intercept_forward_mps=args.intercept_forward_mps,
        intercept_velocity_smoothing=args.intercept_velocity_smoothing,
        max_estimated_target_speed_mps=args.max_estimated_target_speed_mps,
        redetect_enabled=not args.no_redetect,
        redetect_fov_multiplier=args.redetect_fov_multiplier,
        redetect_min_score=args.redetect_min_score,
        redetect_lookahead_sec=args.redetect_lookahead_sec,
        distractor_count=args.distractor_count,
        tracking_noise=args.tracking_noise,
    )
    report = run_and_write(config, args.out)
    summary = report["summary"]
    print(f"report: {args.out / 'report.json'}")
    print(f"trajectory: {args.out / 'trajectory.csv'}")
    print(f"top_down: {args.out / 'top_down.png'}")
    print(f"preview: {args.out / 'preview.gif'}")
    print(f"visible_ratio: {summary['visible_ratio']:.2%}")
    average_error = summary["average_screen_error"]
    print(f"average_screen_error: {average_error:.3f}" if average_error is not None else "average_screen_error: n/a")
    print(f"lost_frames: {summary['lost_frames']}")
    print(f"redetect_frames: {summary['redetect_frames']}")
    print(f"false_redetect_frames: {summary['false_redetect_frames']}")
    print(f"lost_events: {summary['lost_events']}")
    print(f"longest_lost_streak_sec: {summary['longest_lost_streak_sec']:.2f}")
    print(f"grade: {summary['grade']}")


if __name__ == "__main__":
    main()
