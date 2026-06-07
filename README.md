# Aerial Vision Analyzer

Aerial Vision Analyzer is a Python computer-vision project for aerial/drone
imagery, video review, selected-target tracking, and pre-autopilot control
simulation.

The current goal is **not** to fly a drone autonomously yet. The project is
building and testing the software layers that would eventually feed a drone or
gimbal controller:

```text
image/video input
→ object detection
→ operator selects a target
→ tracker keeps target identity
→ control intent is generated
→ simulators test whether that intent is stable
→ safety gate decides whether autopilot work is allowed
```

## Current Status

Working:

- Aerial image object detection
- Video field-review dashboard
- Vehicle/human/animal-like detection profiles
- Target selection by clicking boxes in a live-like preview
- Selected-target tracking with optical flow between detector refreshes
- Control intent modes:
  - `center`
  - `centered`
  - `follow`
  - `tag`
  - `hold`
  - `search`
- Mock drone intent replay
- Closed-loop race/tag simulator
- Race benchmark runner
- Pre-autopilot safety gate
- Control-simulation readiness benchmark

Not done yet:

- Real PX4/MAVLink autopilot bridge
- Gazebo/PX4 SITL integration
- Real drone hardware testing
- True species-level animal detection
- Reliable operation in every high-speed/narrow-FOV case

The latest control-simulation benchmark says the project is ready for
**PX4/Gazebo SITL work**, but still not ready for real drone autonomy:

```text
passed: True
pass_ratio: 98.61%
passed_runs: 71 / 72
```

The remaining failed case is a fast-break target at 10 m/s, 70 degree FOV, and
400 ms latency. The target stays visible, but average screen-centering error is
slightly above the current gate.

## Installation

Install the project:

```bash
python3 -m pip install -e .
```

Install ML/video dependencies:

```bash
python3 -m pip install -e '.[ml]'
```

Download the recommended aerial vehicle model:

```bash
python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='mshamrai/yolov8x-visdrone', filename='best.pt', local_dir='models'))"
```

## Detection Profiles

```text
general   vehicles + humans + animal-like detections
vehicles  vehicles only
humans    pedestrian/people classes
animals   generic animal-like detections
```

Main vehicle/human model:

```text
models/best.pt
source: mshamrai/yolov8x-visdrone
```

Animal-like detections currently use generic YOLO classes such as horse, cow,
sheep, dog, cat, and bird, then collapse them into `animal` for review.

## Image Detection

Run the general profile:

```bash
python3 -m aerial_vision.classify images/image04.png \
  --profile general \
  --json-out outputs/detections.json \
  --annotated-out outputs/annotated.jpg
```

Outputs:

```text
outputs/detections.json
outputs/annotated.jpg
```

Vehicle-only example:

```bash
python3 -m aerial_vision.classify images/image04.png \
  --profile vehicles \
  --json-out outputs/detections.json \
  --annotated-out outputs/annotated.jpg
```

## Video Field Review

Scan a video and keep only interesting moments:

```bash
python3 -m aerial_vision.scan_video videos/testCity1.mp4 \
  --profile general \
  --sample-every-sec 1 \
  --min-detection-difference 0.04 \
  --out field_runs/testCity1
```

Outputs:

```text
field_runs/testCity1/
  report.json
  review.html
  positives/
  detections/
```

Open `review.html` to review positives. Keyboard controls:

```text
Arrow keys  move through events
Q           confirm
W           dismiss
E           mark unreviewed
D           download review.html + review_decisions.json
```

## Interactive Target Tracking

Use a video file like a live source:

```bash
PYTHONPATH=src python3 -m aerial_vision.live_track videos/highway.mp4 \
  --profile vehicles \
  --out live_runs/highway_interactive \
  --interactive-select \
  --select-labels truck bus car \
  --resize-width 640 \
  --imgsz 512 \
  --max-det 50 \
  --detect-every 10
```

Controls:

```text
click box   select or switch target
T           toggle tag mode
Space       show dense boxes again
C           clear target
Q           quit
```

Normal selection keeps the camera centered on the target. Tag mode is a
separate operator action that enables forward tag intent.

The run writes:

```text
live_runs/highway_interactive/live_session.json
```

## Control Intent

When a target is selected, the tracker produces intent rows:

```text
mode
yaw_rate_deg_s
camera_pitch_rate_deg_s
forward_mps
center_error
normalized_offset
tag_enabled
box_area_ratio
```

Intent modes:

```text
center    target visible; center camera/gimbal
centered  target centered; hold position
follow    target appears to be moving away; low forward intent
tag       tag mode active; higher forward intent when centered enough
hold      identity risk or weak lock
search    target lost; rotate toward last known side
```

Offline control-intent example:

```bash
PYTHONPATH=src python3 -m aerial_vision.track_video videos/highway.mp4 \
  --profile vehicles \
  --out tracking_runs/control_intent_highway \
  --target-label truck \
  --resize-width 640 \
  --imgsz 512 \
  --max-frames 80 \
  --detect-every 5
```

Outputs:

```text
tracking_runs/control_intent_highway/
  tracked_preview.mp4
  tracks.json
  target_lock.json
  control_intent.json
```

## Mock Drone Intent Replay

This replays a real `control_intent.json` into a simple mock drone model.

```bash
PYTHONPATH=src python3 -m aerial_vision.intent_sim \
  tracking_runs/control_intent_tag_mode/control_intent.json \
  --out control_runs/mock_px4_tag_mode
```

Outputs:

```text
control_runs/mock_px4_tag_mode/
  mock_bridge_report.json
  mock_bridge_trajectory.csv
  mock_bridge_preview.gif
```

This is not PX4 yet. It is a visual bridge test for command interpretation.

## Mock PX4 Bridge

The bridge layer is the first PX4/Gazebo preparation step. It converts existing
control intent into safety-filtered setpoints and sends them through a mock
PX4-style bridge.

Run it on a saved intent file:

```bash
PYTHONPATH=src python3 -m aerial_vision.drone_bridge \
  tracking_runs/control_intent_highway/control_intent.json \
  --out control_runs/mock_bridge_session
```

Output:

```text
control_runs/mock_bridge_session/bridge_session.json
```

The report checks:

```text
setpoints were sent
center/search/hold modes do not move forward
tag forward motion is safety-filtered
setpoint timing gaps are detected
ready_for_sitl is true only when the stream is continuous enough
```

This is still a mock. The real PX4/MAVSDK adapter should plug into the same
bridge interface after Gazebo/PX4 SITL is installed.

## Closed-Loop Race/Tag Simulator

The race simulator closes the feedback loop:

```text
target moves
drone camera sees target
controller creates intent
drone moves/yaws/pitches
camera view changes
repeat
```

Center-only run:

```bash
PYTHONPATH=src python3 -m aerial_vision.race_sim \
  --out control_runs/race_center_sharp \
  --scenario sharp_turns \
  --duration-sec 30 \
  --target-speed-mps 6
```

Tag-mode run:

```bash
PYTHONPATH=src python3 -m aerial_vision.race_sim \
  --out control_runs/race_tag_sharp \
  --scenario sharp_turns \
  --duration-sec 30 \
  --target-speed-mps 6 \
  --tag-mode
```

Outputs:

```text
race_report.json
race_trajectory.csv
race_preview.gif
```

Recent result:

```text
center mode:
visible_ratio: 100.00%
average_center_error: 0.277
lost_frames: 0
final_distance_m: 15.01

tag mode:
visible_ratio: 100.00%
average_center_error: 0.449
lost_frames: 0
final_distance_m: 10.18
```

Interpretation: tag mode closes distance, but makes camera centering harder.

## Race Benchmark

The older race benchmark is useful for broad tag/follow comparison:

```bash
PYTHONPATH=src python3 -m aerial_vision.benchmark_race \
  --out control_runs/race_benchmark \
  --durations 120 \
  --target-speeds 3,6,10 \
  --horizontal-fovs 45,70,110 \
  --modes center,tag \
  --trials 1
```

Output:

```text
control_runs/race_benchmark/race_benchmark.json
```

Latest finding:

```text
Average score by FOV:
45 deg   59.63
70 deg   87.49
110 deg  94.28
```

Takeaway: wider FOV is currently the biggest stability lever. Tag mode can work
well in moderate cases, but is more fragile overall.

## Control-Sim Benchmark

The current pre-PX4 gate uses the richer control simulator. It checks target
visibility, screen-centering error, lost-target recovery, false re-detection,
latency tolerance, and FOV limits.

Run it:

```bash
PYTHONPATH=src python3 -m aerial_vision.benchmark_control_sim \
  --out control_runs/control_benchmark \
  --durations 60 \
  --target-speeds 3,6,10 \
  --horizontal-fovs 70,110 \
  --latencies-ms 100,250,400 \
  --trials 1
```

Outputs:

```text
control_runs/control_benchmark/control_benchmark.json
control_runs/control_benchmark/control_benchmark.html
```

Latest result:

```text
passed: True
pass_ratio: 98.61%
passed_runs: 71 / 72
```

Meaning: the software is ready to move into PX4/Gazebo simulation. It should
still be treated as simulation-only until SITL proves the bridge, timing,
failsafes, and manual override behavior.

## Safety Gate

For the older race benchmark, run:

```bash
PYTHONPATH=src python3 -m aerial_vision.safety \
  control_runs/race_benchmark/race_benchmark.json \
  --out control_runs/race_benchmark/safety_report.json
```

Default gates:

```text
visible_ratio >= 95%
lost_ratio <= 2%
average_center_error <= 0.35
at least 80% of benchmark runs must pass
```

Older race-benchmark result:

```text
passed: False
pass_ratio: 57.41%
passed_runs: 31 / 54
```

Meaning: the older benchmark is stricter on tag/follow race behavior and still
shows weak areas. The newer control-sim benchmark is the current gate for
starting PX4/Gazebo SITL, not real hardware.

## Recommended Real Drone Direction

For first real hardware, use a stable PX4 development quad rather than a tiny
racing drone:

```text
500-class PX4 quad, such as Holybro X500-style platform
Pixhawk 6C / Pixhawk 6X class flight controller
wide-FOV camera
ground-laptop processing first
onboard compute later, likely Jetson Orin class rather than old Jetson Nano
```

First live drone tests should be camera/gimbal centering only. Movement modes
should wait until PX4/Gazebo SITL and safety gates pass.

## Tests

Run the full test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Current suite:

```text
93 tests
```

## Ignored Local Artifacts

These folders are ignored by git because they contain local runs, models, or
large media:

```text
outputs/
field_runs/
tracking_runs/
live_runs/
live_benchmarks/
control_runs/
videos/
models/
```
