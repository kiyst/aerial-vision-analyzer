from __future__ import annotations

import argparse
import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from aerial_vision.annotate import draw_detections
from aerial_vision.detection import ObjectDetection, normalize_label
from aerial_vision.pipeline import PROFILE_CONFIGS, analyze_image_with_detectors, build_detector_bundle, settings_from_profile


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    duration_sec: float


def format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}-{minutes:02d}-{secs:02d}"


def timestamp_display(seconds: float) -> str:
    return format_timestamp(seconds).replace("-", ":")


def prepare_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def frame_difference(first: Path, second: Path, *, size: tuple[int, int] = (64, 64)) -> float:
    with Image.open(first).convert("L").resize(size) as first_image:
        first_pixels = list(first_image.getdata())
    with Image.open(second).convert("L").resize(size) as second_image:
        second_pixels = list(second_image.getdata())

    total_delta = sum(abs(a - b) for a, b in zip(first_pixels, second_pixels))
    return total_delta / (len(first_pixels) * 255)


def detection_signature(detections: list[ObjectDetection], *, image_width: int, image_height: int) -> list[tuple[str, float, float, float, float]]:
    signatures = []
    for detection in detections:
        center_x, center_y = detection.box.center
        signatures.append(
            (
                normalize_label(detection.label),
                center_x / image_width,
                center_y / image_height,
                detection.box.width / image_width,
                detection.box.height / image_height,
            )
        )

    return sorted(signatures)


def detection_difference(
    current: list[ObjectDetection],
    previous: list[ObjectDetection],
    *,
    image_width: int,
    image_height: int,
) -> float:
    if not current and not previous:
        return 0.0
    if not current or not previous:
        return 1.0

    current_signature = detection_signature(current, image_width=image_width, image_height=image_height)
    previous_signature = detection_signature(previous, image_width=image_width, image_height=image_height)
    label_mismatch_penalty = 1.0
    unmatched_penalty = 0.5
    matched_scores: list[float] = []
    used_previous: set[int] = set()

    for current_item in current_signature:
        best_index = None
        best_score = label_mismatch_penalty

        for index, previous_item in enumerate(previous_signature):
            if index in used_previous:
                continue
            if current_item[0] != previous_item[0]:
                continue

            score = sum(abs(current_item[i] - previous_item[i]) for i in range(1, 5)) / 4
            if score < best_score:
                best_score = score
                best_index = index

        if best_index is None:
            matched_scores.append(unmatched_penalty)
        else:
            used_previous.add(best_index)
            matched_scores.append(best_score)

    unmatched_previous = len(previous_signature) - len(used_previous)
    matched_scores.extend([unmatched_penalty] * unmatched_previous)
    return sum(matched_scores) / len(matched_scores)


def render_review_html(report: dict[str, object]) -> str:
    video = report["video"]
    events_json = json.dumps(report["events"]).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aerial Vision Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, Arial, Helvetica, sans-serif;
      background: #0f1214;
      color: #eef2f4;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #0f1214;
      overflow: hidden;
    }}
    h1, h2, h3, p {{
      margin: 0;
    }}
    button, select {{
      font: inherit;
    }}
    button:focus-visible,
    select:focus-visible,
    a:focus-visible {{
      outline: 3px solid #f2c94c;
      outline-offset: 2px;
    }}
    .shell {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      height: 100vh;
      overflow: hidden;
    }}
    .sidebar {{
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      border-right: 1px solid #2b343c;
      background: #151a1f;
      height: 100vh;
      overflow: hidden;
    }}
    .brand {{
      padding: 18px;
      border-bottom: 1px solid #2b343c;
    }}
    .brand h1 {{
      font-size: 20px;
      line-height: 1.15;
      margin-bottom: 12px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .metric {{
      min-width: 0;
      padding: 9px 10px;
      background: #20282f;
      border: 1px solid #313c45;
      border-radius: 6px;
    }}
    .metric span {{
      display: block;
      color: #9ba9b3;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .04em;
      margin-bottom: 4px;
    }}
    .metric strong {{
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 14px;
    }}
    .controls {{
      display: grid;
      gap: 10px;
      padding: 14px 18px;
      border-bottom: 1px solid #2b343c;
      background: #11161a;
    }}
    .control-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .filter {{
      width: 100%;
      min-width: 0;
      color: #edf3f7;
      background: #20282f;
      border: 1px solid #3a4650;
      border-radius: 6px;
      padding: 9px 10px;
    }}
    .event-list {{
      overflow: auto;
      display: grid;
      align-content: start;
      gap: 8px;
      padding: 12px;
    }}
    .event-button {{
      width: 100%;
      display: grid;
      gap: 8px;
      text-align: left;
      color: #eef2f4;
      background: #1b2228;
      border: 1px solid #303a43;
      border-radius: 6px;
      padding: 11px;
      cursor: pointer;
    }}
    .event-button:hover,
    .event-button.active {{
      border-color: #79b8ff;
      background: #202b35;
    }}
    .event-button.active {{
      box-shadow: inset 3px 0 0 #79b8ff;
    }}
    .event-top {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }}
    .time {{
      font-size: 18px;
      font-weight: 700;
    }}
    .status {{
      flex: none;
      padding: 4px 6px;
      color: #c8d2d9;
      background: #29323a;
      border-radius: 4px;
      font-size: 12px;
    }}
    .status.confirmed {{
      color: #aef0bc;
      background: #173820;
    }}
    .status.dismissed {{
      color: #ffc1c1;
      background: #432022;
    }}
    .counts {{
      color: #b4c0c8;
      font-size: 13px;
      line-height: 1.35;
    }}
    .workspace {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
      height: 100vh;
      overflow: hidden;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      padding: 14px 18px;
      border-bottom: 1px solid #2b343c;
      background: #101519;
    }}
    .toolbar h2 {{
      font-size: 18px;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .actions button {{
      color: #edf3f7;
      background: #20282f;
      border: 1px solid #3a4650;
      border-radius: 6px;
      padding: 8px 11px;
      cursor: pointer;
    }}
    .actions button:hover {{
      border-color: #79b8ff;
    }}
    .viewer {{
      overflow: auto;
      padding: 18px;
      display: grid;
      align-content: start;
      justify-items: center;
      min-height: 0;
    }}
    .image-wrap {{
      width: min(100%, 980px);
      max-height: 68vh;
      background: #080a0c;
      border: 1px solid #2b343c;
      border-radius: 6px;
      padding: 8px;
      margin-bottom: 14px;
      display: grid;
      place-items: center;
      overflow: auto;
    }}
    img {{
      display: block;
      max-width: 100%;
      max-height: 64vh;
      width: auto;
      height: auto;
      border-radius: 3px;
    }}
    .details {{
      width: min(100%, 980px);
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .detail {{
      padding: 10px;
      border: 1px solid #2f3941;
      border-radius: 6px;
      background: #171d22;
    }}
    .detail span {{
      display: block;
      color: #9ba9b3;
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .detail strong {{
      font-size: 15px;
    }}
    .links {{
      width: min(100%, 980px);
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    a {{
      color: #8fc8ff;
    }}
    .empty {{
      padding: 24px;
      color: #bac5cc;
      border: 1px solid #2f3941;
      border-radius: 6px;
      background: #171d22;
    }}
    @media (max-width: 760px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        min-height: auto;
        border-right: 0;
      }}
      .event-list {{
        max-height: 320px;
      }}
      .details {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <section class="brand">
        <h1>Aerial Vision Dashboard</h1>
        <div class="summary">
          <div class="metric"><span>Video</span><strong title="{html.escape(video["path"])}">{html.escape(Path(video["path"]).name)}</strong></div>
          <div class="metric"><span>Profile</span><strong>{html.escape(str(report["profile"]))}</strong></div>
          <div class="metric"><span>Checked</span><strong>{report["frames_checked"]}</strong></div>
          <div class="metric"><span>Positive</span><strong>{report["positive_frames"]}</strong></div>
          <div class="metric"><span>Skipped</span><strong>{report.get("skipped_similar_positive_frames", 0)}</strong></div>
          <div class="metric"><span>Duration</span><strong>{video["duration_sec"]:.1f}s</strong></div>
        </div>
      </section>
      <section class="controls">
        <div class="control-row">
          <select id="labelFilter" class="filter" aria-label="Filter by label">
            <option value="all">All labels</option>
          </select>
          <select id="statusFilter" class="filter" aria-label="Filter by review status">
            <option value="all">All statuses</option>
            <option value="unreviewed">Unreviewed</option>
            <option value="confirmed">Confirmed</option>
            <option value="dismissed">Dismissed</option>
          </select>
        </div>
      </section>
      <nav id="eventList" class="event-list" aria-label="Positive events"></nav>
    </aside>
    <main class="workspace">
      <header class="toolbar">
        <h2 id="eventTitle">No event selected</h2>
        <div class="actions">
          <button type="button" data-action="unreviewed" title="Mark selected event as unreviewed">Unreviewed</button>
          <button type="button" data-action="confirmed" title="Confirm selected event">Confirm</button>
          <button type="button" data-action="dismissed" title="Dismiss selected event">Dismiss</button>
        </div>
      </header>
      <section id="viewer" class="viewer">
        <p class="empty">No positive frames found.</p>
      </section>
    </main>
  </div>
  <script>
    const events = {events_json};
    const runKey = "aerial-review:" + {json.dumps(str(video["path"]))} + ":" + {json.dumps(str(report["profile"]))};
    let statusStore = {{}};
    try {{
      statusStore = JSON.parse(localStorage.getItem(runKey) || "{{}}");
    }} catch (error) {{
      statusStore = {{}};
    }}
    let selectedIndex = 0;

    const eventList = document.getElementById("eventList");
    const viewer = document.getElementById("viewer");
    const eventTitle = document.getElementById("eventTitle");
    const labelFilter = document.getElementById("labelFilter");
    const statusFilter = document.getElementById("statusFilter");

    function eventId(event) {{
      return String(event.timestamp_sec ?? event.timestamp ?? event.frame_index);
    }}

    function getStatus(event) {{
      return statusStore[eventId(event)] || "unreviewed";
    }}

    function setStatus(status) {{
      const event = events[selectedIndex];
      if (!event) return;
      statusStore[eventId(event)] = status;
      try {{
        localStorage.setItem(runKey, JSON.stringify(statusStore));
      }} catch (error) {{
        // Status still updates in-page even if the browser blocks local storage.
      }}
      render();
    }}

    function countsText(event) {{
      return Object.entries(event.counts_by_label || {{}})
        .map(([label, count]) => `${{label}}=${{count}}`)
        .join(", ");
    }}

    function confidenceText(value) {{
      return value === null || value === undefined ? "n/a" : Number(value).toFixed(3);
    }}

    function labels() {{
      return [...new Set(events.flatMap(event => Object.keys(event.counts_by_label || {{}})))].sort();
    }}

    function filteredEvents() {{
      const label = labelFilter.value;
      const status = statusFilter.value;
      return events
        .map((event, index) => ({{ event, index }}))
        .filter(item => label === "all" || Object.prototype.hasOwnProperty.call(item.event.counts_by_label || {{}}, label))
        .filter(item => status === "all" || getStatus(item.event) === status);
    }}

    function populateLabels() {{
      for (const label of labels()) {{
        const option = document.createElement("option");
        option.value = label;
        option.textContent = label;
        labelFilter.appendChild(option);
      }}
    }}

    function renderList(items) {{
      eventList.innerHTML = "";
      if (items.length === 0) {{
        eventList.innerHTML = '<p class="empty">No matching events.</p>';
        return;
      }}

      for (const {{ event, index }} of items) {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "event-button" + (index === selectedIndex ? " active" : "");
        button.setAttribute("aria-current", index === selectedIndex ? "true" : "false");
        button.setAttribute("aria-label", `Select event at ${{event.timestamp}}, ${{event.total_detections}} detections, ${{getStatus(event)}}`);
        button.innerHTML = `
          <span class="event-top">
            <span class="time">${{event.timestamp}}</span>
            <span class="status ${{getStatus(event)}}">${{getStatus(event)}}</span>
          </span>
          <span class="counts">${{event.total_detections}} detections · ${{countsText(event)}}</span>
        `;
        button.addEventListener("click", () => {{
          selectedIndex = index;
          render();
        }});
        eventList.appendChild(button);
      }}
    }}

    function renderViewer(event) {{
      if (!event) {{
        eventTitle.textContent = "No event selected";
        viewer.innerHTML = '<p class="empty">No positive frames found.</p>';
        return;
      }}

      eventTitle.textContent = `Event ${{event.timestamp}} · ${{getStatus(event)}}`;
      const frameDelta = event.frame_difference_from_previous === null || event.frame_difference_from_previous === undefined
        ? "n/a"
        : Number(event.frame_difference_from_previous).toFixed(3);
      const detectionDelta = event.detection_difference_from_previous === null || event.detection_difference_from_previous === undefined
        ? "n/a"
        : Number(event.detection_difference_from_previous).toFixed(3);

      viewer.innerHTML = `
        <div class="image-wrap">
          <a href="${{event.annotated_image}}">
            <img src="${{event.annotated_image}}" alt="Annotated frame at ${{event.timestamp}}">
          </a>
        </div>
        <div class="details">
          <div class="detail"><span>Total detections</span><strong>${{event.total_detections}}</strong></div>
          <div class="detail"><span>Average confidence</span><strong>${{confidenceText(event.average_confidence)}}</strong></div>
          <div class="detail"><span>Frame delta</span><strong>${{frameDelta}}</strong></div>
          <div class="detail"><span>Detection delta</span><strong>${{detectionDelta}}</strong></div>
        </div>
        <div class="detail">
          <span>Counts</span>
          <strong>${{countsText(event) || "n/a"}}</strong>
        </div>
        <p class="links">
          <a href="${{event.detections_json}}">Detection JSON</a>
          <a href="${{event.annotated_image}}">Annotated image</a>
        </p>
      `;
    }}

    function focusSelectedEvent() {{
      const active = eventList.querySelector(".event-button.active");
      if (!active) return;
      active.focus({{ preventScroll: true }});
      active.scrollIntoView({{ block: "nearest" }});
    }}

    function resetScrollAtFirstEvent() {{
      const items = filteredEvents();
      if (items.length === 0 || items[0].index !== selectedIndex) return;
      eventList.scrollTop = 0;
    }}

    function selectFromVisibleItems(offset) {{
      const items = filteredEvents();
      if (items.length === 0) return;
      const currentPosition = Math.max(0, items.findIndex(item => item.index === selectedIndex));
      const nextPosition = Math.min(items.length - 1, Math.max(0, currentPosition + offset));
      selectedIndex = items[nextPosition].index;
      render();
      focusSelectedEvent();
      resetScrollAtFirstEvent();
    }}

    function selectEdgeFromVisibleItems(edge) {{
      const items = filteredEvents();
      if (items.length === 0) return;
      selectedIndex = edge === "start" ? items[0].index : items[items.length - 1].index;
      render();
      focusSelectedEvent();
      resetScrollAtFirstEvent();
    }}

    function render() {{
      const items = filteredEvents();
      if (items.length > 0 && !items.some(item => item.index === selectedIndex)) {{
        selectedIndex = items[0].index;
      }}
      renderList(items);
      renderViewer(events[selectedIndex]);
    }}

    document.querySelectorAll("[data-action]").forEach(button => {{
      button.addEventListener("click", () => setStatus(button.dataset.action));
    }});
    labelFilter.addEventListener("change", render);
    statusFilter.addEventListener("change", render);
    document.addEventListener("keydown", event => {{
      const tagName = event.target?.tagName;
      if (tagName === "SELECT" || tagName === "INPUT" || tagName === "TEXTAREA") return;

      if (event.key === "ArrowDown" || event.key === "ArrowRight") {{
        event.preventDefault();
        selectFromVisibleItems(1);
      }} else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {{
        event.preventDefault();
        selectFromVisibleItems(-1);
      }} else if (event.key === "Home") {{
        event.preventDefault();
        selectEdgeFromVisibleItems("start");
      }} else if (event.key === "End") {{
        event.preventDefault();
        selectEdgeFromVisibleItems("end");
      }} else if (event.key.toLowerCase() === "q") {{
        event.preventDefault();
        setStatus("confirmed");
      }} else if (event.key.toLowerCase() === "w") {{
        event.preventDefault();
        setStatus("dismissed");
      }} else if (event.key.toLowerCase() === "e") {{
        event.preventDefault();
        setStatus("unreviewed");
      }}
    }});

    populateLabels();
    render();
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan a video for positive aerial detections.")
    parser.add_argument("video", type=Path, help="Path to a video file.")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_CONFIGS),
        default="general",
        help="Detection profile to run on sampled frames.",
    )
    parser.add_argument("--sample-every-sec", type=float, default=1.0, help="Seconds between sampled frames.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the field review run.")
    parser.add_argument(
        "--min-frame-difference",
        type=float,
        default=0.0,
        help="Skip positive frames visually similar to the last kept positive. Range 0-1.",
    )
    parser.add_argument(
        "--min-detection-difference",
        type=float,
        default=0.0,
        help="Skip positive frames with detections similar to the last kept positive. Range 0-1.",
    )
    parser.add_argument("--save-raw-frames", action="store_true", help="Keep raw sampled frames that had detections.")
    parser.add_argument("--save-empty", action="store_true", help="Also save raw sampled frames with no detections.")
    return parser


def scan_video(
    video_path: Path,
    *,
    profile: str,
    sample_every_sec: float,
    out_dir: Path,
    min_frame_difference: float = 0.0,
    min_detection_difference: float = 0.0,
    save_raw_frames: bool = False,
    save_empty: bool = False,
) -> dict[str, object]:
    if sample_every_sec <= 0:
        raise ValueError("sample_every_sec must be positive.")
    if not 0 <= min_frame_difference <= 1:
        raise ValueError("min_frame_difference must be between 0 and 1.")
    if not 0 <= min_detection_difference <= 1:
        raise ValueError("min_detection_difference must be between 0 and 1.")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video scanning. Install with: python3 -m pip install -e '.[ml]'") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        raise RuntimeError("Video FPS could not be determined.")

    duration_sec = frame_count / fps if frame_count else 0
    frame_step = max(1, int(round(fps * sample_every_sec)))
    settings = settings_from_profile(profile)
    detectors = build_detector_bundle(settings)

    prepare_output_dir(out_dir)
    frames_dir = out_dir / "frames"
    positives_dir = out_dir / "positives"
    detections_dir = out_dir / "detections"
    for directory in (positives_dir, detections_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if save_raw_frames or save_empty:
        frames_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict[str, object]] = []
    frames_checked = 0
    frame_index = 0
    last_positive_frame_path: Path | None = None
    last_positive_detections: list[ObjectDetection] | None = None
    skipped_similar_positive_frames = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        timestamp_sec = frame_index / fps
        timestamp = format_timestamp(timestamp_sec)
        frame_path = out_dir / f".frame_{timestamp}.jpg"
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb_frame).save(frame_path)

        analysis = analyze_image_with_detectors(frame_path, settings, detectors)
        frames_checked += 1

        if analysis.total_detections > 0:
            frame_delta = None
            detection_delta = None
            skip_as_similar = False
            if last_positive_frame_path is not None:
                frame_delta = frame_difference(last_positive_frame_path, frame_path)
                skip_as_similar = min_frame_difference > 0 and frame_delta < min_frame_difference

            if last_positive_detections is not None:
                detection_delta = detection_difference(
                    analysis.detections,
                    last_positive_detections,
                    image_width=analysis.image_width,
                    image_height=analysis.image_height,
                )
                if min_detection_difference > 0:
                    skip_as_similar = skip_as_similar or detection_delta < min_detection_difference

            if skip_as_similar:
                skipped_similar_positive_frames += 1
                frame_path.unlink()
                frame_index += 1
                continue

            annotated_path = positives_dir / f"{timestamp}.jpg"
            detection_path = detections_dir / f"{timestamp}.json"
            draw_detections(frame_path, analysis.detections, annotated_path)
            detection_path.write_text(json.dumps(analysis.to_dict(), indent=2) + "\n", encoding="utf-8")
            raw_frame_path = None
            if save_raw_frames:
                raw_frame_path = frames_dir / f"{timestamp}.jpg"
                frame_path.replace(raw_frame_path)
                last_positive_frame_path = raw_frame_path
            else:
                last_positive_frame_path = annotated_path
                frame_path.unlink()
            events.append(
                {
                    "timestamp": timestamp_display(timestamp_sec),
                    "timestamp_sec": timestamp_sec,
                    "frame_index": frame_index,
                    "counts_by_label": analysis.counts_by_label,
                    "total_detections": analysis.total_detections,
                    "average_confidence": analysis.average_confidence,
                    "frame_difference_from_previous": frame_delta,
                    "detection_difference_from_previous": detection_delta,
                    "annotated_image": str(annotated_path.relative_to(out_dir)),
                    "detections_json": str(detection_path.relative_to(out_dir)),
                    "raw_frame": str(raw_frame_path.relative_to(out_dir)) if raw_frame_path else None,
                }
            )
            last_positive_detections = analysis.detections
        elif save_empty:
            frame_path.replace(frames_dir / f"{timestamp}.jpg")
        else:
            frame_path.unlink()

        frame_index += 1

    capture.release()

    report = {
        "video": {
            "path": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
        },
        "profile": profile,
        "sample_every_sec": sample_every_sec,
        "min_frame_difference": min_frame_difference,
        "min_detection_difference": min_detection_difference,
        "frames_checked": frames_checked,
        "positive_frames": len(events),
        "skipped_similar_positive_frames": skipped_similar_positive_frames,
        "events": events,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "review.html").write_text(render_review_html(report), encoding="utf-8")
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = scan_video(
        args.video,
        profile=args.profile,
        sample_every_sec=args.sample_every_sec,
        out_dir=args.out,
        min_frame_difference=args.min_frame_difference,
        min_detection_difference=args.min_detection_difference,
        save_raw_frames=args.save_raw_frames,
        save_empty=args.save_empty,
    )
    print(f"video: {args.video}")
    print(f"profile: {args.profile}")
    print(f"frames_checked: {report['frames_checked']}")
    print(f"positive_frames: {report['positive_frames']}")
    print(f"skipped_similar_positive_frames: {report['skipped_similar_positive_frames']}")
    print(f"report: {args.out / 'report.json'}")
    print(f"review: {args.out / 'review.html'}")


if __name__ == "__main__":
    main()
