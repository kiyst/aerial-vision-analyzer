from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LivePlaybackStatus:
    frame_index: int
    video_time_sec: float
    wall_elapsed_sec: float
    latency_sec: float
    real_time_factor: float | None
    processed_frames: int
    dropped_frames: int

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "video_time_sec": self.video_time_sec,
            "wall_elapsed_sec": self.wall_elapsed_sec,
            "latency_sec": self.latency_sec,
            "real_time_factor": self.real_time_factor,
            "processed_frames": self.processed_frames,
            "dropped_frames": self.dropped_frames,
        }


def frame_time_sec(frame_index: int, fps: float) -> float:
    if frame_index < 0:
        raise ValueError("frame_index cannot be negative.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    return frame_index / fps


def realtime_delay_sec(*, video_time_sec: float, wall_elapsed_sec: float) -> float:
    if video_time_sec < 0:
        raise ValueError("video_time_sec cannot be negative.")
    if wall_elapsed_sec < 0:
        raise ValueError("wall_elapsed_sec cannot be negative.")
    return max(0.0, video_time_sec - wall_elapsed_sec)


def should_drop_frame(*, video_time_sec: float, wall_elapsed_sec: float, max_latency_sec: float) -> bool:
    if max_latency_sec < 0:
        raise ValueError("max_latency_sec cannot be negative.")
    return wall_elapsed_sec - video_time_sec > max_latency_sec


def playback_status(
    *,
    frame_index: int,
    fps: float,
    wall_elapsed_sec: float,
    processed_frames: int,
    dropped_frames: int,
) -> LivePlaybackStatus:
    if processed_frames < 0:
        raise ValueError("processed_frames cannot be negative.")
    if dropped_frames < 0:
        raise ValueError("dropped_frames cannot be negative.")

    video_time = frame_time_sec(frame_index, fps)
    latency = max(0.0, wall_elapsed_sec - video_time)
    real_time_factor = None if wall_elapsed_sec <= 0 else video_time / wall_elapsed_sec
    return LivePlaybackStatus(
        frame_index=frame_index,
        video_time_sec=video_time,
        wall_elapsed_sec=wall_elapsed_sec,
        latency_sec=latency,
        real_time_factor=real_time_factor,
        processed_frames=processed_frames,
        dropped_frames=dropped_frames,
    )
