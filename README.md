# Aerial Vision Analyzer

Aerial Vision Analyzer is a Python computer vision project for reviewing
top-down aerial, drone, and satellite-style imagery. It detects objects of
interest, writes machine-readable JSON, and creates annotated images for human
review.

The current project is focused on **review and alerting**, not fully autonomous
identification. The system is meant to flag possible objects of interest so a
person can inspect the image or frame.

## Current Capabilities

- Detect vehicles in overhead imagery:
  - `car`
  - `van`
  - `truck`
  - `bus`
  - `motor`
  - `bicycle`
- Detect human-like classes:
  - `pedestrian`
  - `people`
- Detect animal-like objects for review:
  - `horse`
  - `cow`
  - `sheep`
  - `dog`
  - `cat`
  - `bird`
- Collapse animal proxy labels into a single `animal` label.
- Run tiled inference for small objects in large aerial images.
- Draw annotated bounding boxes on output images.
- Export JSON with image metadata, counts, confidence scores, boxes, centers,
  and pixel areas.
- Clear previous files from `outputs/` on each run.
- Scan video files by sampling frames at a chosen interval.
- Keep only positive frames by default.
- Skip repeated positive frames when detections look too similar.
- Generate a self-contained field review dashboard at `review.html`.
- Track vehicles through video with persistent track IDs.
- Log basic appearance consistency using color and shape scores.
- Run live-like target tracking from a video file, camera index, or stream URL.
- Benchmark live tracking settings across resize width, detector refresh rate,
  YOLO image size, and detection limits.

## Project Philosophy

This project is built around a practical field-review workflow:

```text
aerial image or drone frame
→ run object detection
→ save annotated output
→ human reviews positives
```

For video, the workflow is:

```text
video file
→ sample frames every N seconds
→ run the selected detection profile
→ keep positive moments
→ open review.html
→ confirm, dismiss, or leave events unreviewed
```

For wildlife and security use cases, the model should be treated as a screening
tool. It can flag “something interesting is here,” but final identification
should be done by a person.

## Models

The best current model for vehicles and humans is:

```text
mshamrai/yolov8x-visdrone
```

It is saved locally as:

```text
models/best.pt
```

This model is trained on VisDrone-style aerial/drone imagery and works much
better for overhead road scenes than generic COCO YOLO models.

For animal-like detections, the project currently uses:

```text
yolo11x.pt
```

This is a generic COCO model. It does **not** have a deer class, so animal
detection is intentionally treated as “animal-like positive for review,” not
exact species identification.

## Installation

Install the base project:

```bash
python3 -m pip install -e .
```

Install YOLO/Ultralytics dependencies:

```bash
python3 -m pip install -e '.[ml]'
```

Download the recommended VisDrone model:

```bash
python3 -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='mshamrai/yolov8x-visdrone', filename='best.pt', local_dir='models'))"
```

## Quick Start

Run the automatic general profile:

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

## Detection Profiles

Profiles configure models, classes, thresholds, tiling, and post-processing.

```text
general   vehicles + humans + animal-like detections
vehicles  vehicles only
humans    pedestrians/people only
animals   animal-like detections only, collapsed to animal
```

Profile thresholds:

```text
general   vehicles/humans at 0.15, animals at 0.05
vehicles  0.15
humans    0.20
animals   0.05
```

Example commands:

```bash
python3 -m aerial_vision.classify images/image04.png \
  --profile vehicles \
  --json-out outputs/detections.json \
  --annotated-out outputs/annotated.jpg
```

```bash
python3 -m aerial_vision.classify images/image08.png \
  --profile humans \
  --json-out outputs/detections.json \
  --annotated-out outputs/annotated.jpg
```

```bash
python3 -m aerial_vision.classify images/image09.png \
  --profile animals \
  --json-out outputs/detections.json \
  --annotated-out outputs/annotated.jpg
```

## Manual Advanced Command

The general profile expands roughly to this:

```bash
python3 -m aerial_vision.classify images/image04.png \
  --model models/best.pt \
  --min-confidence 0.15 \
  --objects car van truck bus motor bicycle pedestrian people \
  --extra-model yolo11x.pt \
  --extra-min-confidence 0.05 \
  --extra-objects horse cow sheep dog cat bird \
  --tile-size 800 \
  --tile-overlap 240 \
  --geometry-preset overhead-vehicles \
  --collapse-animals \
  --json-out outputs/detections.json \
  --annotated-out outputs/annotated.jpg
```

## Tiled Inference

Aerial objects are often small. The project can split an image into overlapping
tiles, run detection on each tile, shift the boxes back into full-image
coordinates, and suppress duplicate boxes.

Important flags:

```text
--tile-size 800
--tile-overlap 240
--nms-iou 0.45
```

## Geometry Filtering

The `overhead-vehicles` preset removes unrealistic boxes while allowing long
vehicles like trucks, vans, and buses to be larger than cars.

```bash
--geometry-preset overhead-vehicles
```

This helps reduce false positives like:

```text
building roof → truck
large rectangle → vehicle
shadow patch → person
```

## JSON Output

The JSON output includes:

- source image path
- image width and height
- total detection count
- counts by label
- average confidence
- largest detection
- full detection list
- bounding boxes
- centers
- pixel areas
- source model

## Current Limitations

- Animal detection is generic and proxy-based.
- Deer are currently detected only as animal-like shapes, not as a true `deer`
  class.
- People can false-positive in scenes where no people exist.
- Buildings, roads, paths, and nature features are not properly segmented yet.
- Camera and stream input support depends on OpenCV being able to open the
  source on the field machine.
- Review decisions can be exported from the browser as JSON, but the static
  HTML page cannot silently write that file back into the run folder.

## Video Field Review

Scan a video file:

```bash
python3 -m aerial_vision.scan_video videos/flight01.mp4 \
  --profile general \
  --sample-every-sec 1 \
  --min-detection-difference 0.04 \
  --out field_runs/flight01
```

Behavior:

```text
video file
→ sample frames every N seconds
→ run detection profile
→ save positive frames
→ save annotated frames
→ save per-frame JSON
→ write report.json
→ write review.html
```

Output structure:

```text
field_runs/flight01/
  report.json
  review.html
  positives/
    00-01-24.jpg
  detections/
    00-01-24.json
```

By default, raw sampled frames are temporary and only annotated positives are
kept. Use `--save-raw-frames` to keep raw positive frames, or `--save-empty` to
also keep sampled frames that had no detections.

Each run clears and rebuilds the target output folder passed to `--out`.

The scanner loads the selected model profile once per run and reuses those
model objects for each sampled frame.

Open `review.html` in a browser to review the positive moments visually. The
dashboard is designed for field triage:

- left-side event list
- selected annotated frame viewer
- label and status filters
- per-event detection counts, confidence, and difference scores
- links to each annotated frame and JSON file
- status buttons for `Confirm`, `Dismiss`, and `Unreviewed`
- a `Download` button that downloads `review.html` and
  `review_decisions.json`

Keyboard controls:

```text
Arrow Down / Arrow Right  next event
Arrow Up / Arrow Left     previous event
Home                      first visible event
End                       last visible event
Q                         confirm selected event
W                         dismiss selected event
E                         mark selected event unreviewed
D                         download review.html and review_decisions.json
```

The downloaded review decisions file includes run metadata, status counts, and
every event with its current review status. Save that downloaded file next to
the run if you want the folder to contain the final field decisions:

```text
field_runs/flight01/review_decisions.json
```

If you do not press `D`, closing the review tab is the abandon/reject path. No
portable review package is created.

Export confirmed events only:

```bash
python3 -m aerial_vision.export_review field_runs/flight01/review_decisions.json \
  --run-dir field_runs/flight01 \
  --out field_runs/flight01/confirmed_export
```

Output structure:

```text
field_runs/flight01/confirmed_export/
  confirmed_events.json
  images/
    00-01-24.jpg
  detections/
    00-01-24.json
```

Use `--min-detection-difference` to skip repeated positives whose detections
look similar to the last kept positive. This compares labels, box centers, and
box sizes. A starting value of `0.04` works well for collapsing stationary-object
repeats.

Use `--min-frame-difference` if you want to compare the whole frame visually
instead. Detection difference is usually better for drone footage because it
focuses on the detected objects rather than the entire background.

## Target Tracking Prototype

Track vehicles through a video and write persistent track IDs:

```bash
python3 -m aerial_vision.track_video videos/highway.mp4 \
  --profile vehicles \
  --out tracking_runs/highway \
  --max-frames 90 \
  --resize-width 960 \
  --min-track-frames 5
```

Output structure:

```text
tracking_runs/highway/
  tracks.json
  target_lock.json
  tracked_preview.mp4
```

`tracks.json` includes each exported track's boxes, timestamps, class labels,
confidence scores, color histogram, shape data, and basic identity scores.
The preview video draws track IDs on top of the source footage.

## Operator Target Picker

Create a numbered target-selection image from an early video frame:

```bash
python3 -m aerial_vision.pick_target videos/highway.mp4 \
  --profile vehicles \
  --out tracking_runs/highway_picker \
  --target-label truck \
  --resize-width 640 \
  --imgsz 512 \
  --max-det 50
```

Outputs:

```text
tracking_runs/highway_picker/
  target_choices.jpg
  target_choices.json
```

Open `target_choices.jpg`, look at the numbered boxes, then track the selected
choice:

```bash
python3 -m aerial_vision.pick_target videos/highway.mp4 \
  --profile vehicles \
  --out tracking_runs/highway_picker_tracked \
  --target-label truck \
  --choice 1 \
  --detect-every 15 \
  --resize-width 640 \
  --imgsz 512 \
  --max-det 50 \
  --max-frames 90
```

That writes:

```text
tracking_runs/highway_picker_tracked/tracking/tracked_preview.mp4
tracking_runs/highway_picker_tracked/tracking/target_lock.json
```

Use `--timestamp-sec` or `--frame-index` if the object you want is not visible
in the first frame.

Lock onto a specific track ID:

```bash
python3 -m aerial_vision.track_video videos/highway.mp4 \
  --profile vehicles \
  --out tracking_runs/highway_target_1 \
  --max-frames 90 \
  --resize-width 960 \
  --min-track-frames 5 \
  --target-id 1 \
  --detect-every 10
```

When `--target-id` is provided, the preview highlights that target and
`target_lock.json` records:

- lock state: `locked`, `weak_lock`, `id_switch_risk`, or `lost`
- target box and confidence
- appearance identity score
- overlap risk with similar nearby objects
- pixel offset from frame center
- normalized offset from frame center

Those offsets are the future bridge into gimbal/drone control logic.

Useful flags:

```text
--tracker bytetrack.yaml       default Ultralytics tracker
--tracker botsort.yaml         stronger tracker option to compare
--resize-width 960             faster tracking while preserving useful detail
--imgsz 512                    smaller YOLO inference tensor for faster refreshes
--max-det 50                   cap detections per refresh in dense scenes
--device cpu                   choose an Ultralytics device, such as cpu, mps, 0
--half                         use FP16 on supported GPU devices
--sample-every-sec 1           track sampled frames instead of every frame
--min-track-frames 5           only export tracks seen at least 5 frames
--max-frames 90                quick smoke test
--target-id 1                  highlight/log lock state for one track
--detect-every 15              detector refresh interval for real-time target lock
```

When `--target-id` and `--detect-every` are used together, the detector runs
only every N processed frames and optical flow tracks the selected target
between detector refreshes. This is the real-time-oriented path:

```text
YOLO/ByteTrack refresh frame
→ optical flow target updates
→ YOLO/ByteTrack refresh frame
→ repeat
```

On `videos/highway.mp4`, target `1` at 640-960px for 90 frames improved from
about `0.66 FPS` with detector-every-frame tracking to roughly `16-20 FPS` in
fast-loop tests by using sparse detector refreshes, class filtering, smaller
YOLO inference size, and optical flow between detector frames.

## Live-Like Tracking

Use a video file as a live source:

```bash
python3 -m aerial_vision.live_track videos/highway.mp4 \
  --profile vehicles \
  --out live_runs/highway_live \
  --target-id 1 \
  --detect-every 15 \
  --resize-width 640 \
  --imgsz 512 \
  --max-det 50 \
  --max-frames 90
```

This plays the file on a real-time clock and writes:

```text
live_runs/highway_live/live_session.json
```

The live session report includes:

- processed frames
- dropped frames
- detector frames
- optical-flow frames
- latency
- real-time factor
- target lock states

Use `--display` to show an OpenCV live preview window. Use `--no-realtime` to
benchmark the tracking loop as fast as possible.

Interactive live target selection:

```bash
python3 -m aerial_vision.live_track videos/highway.mp4 \
  --profile vehicles \
  --out live_runs/highway_interactive \
  --interactive-select \
  --resize-width 640 \
  --imgsz 512 \
  --max-det 50
```

Before the preview opens, the terminal asks which labels to show:

```text
all
car
van
truck
bus
motor
bicycle
```

Type `all`, or type a smaller set such as:

```text
truck bus
```

You can skip the prompt and pass the filter directly:

```bash
python3 -m aerial_vision.live_track videos/highway.mp4 \
  --profile vehicles \
  --out live_runs/highway_interactive_trucks \
  --interactive-select \
  --select-labels truck bus \
  --resize-width 640 \
  --imgsz 512 \
  --max-det 50
```

Controls:

```text
left click a detected box   select or switch active target
Space                       show dense boxes again for switching
C                           clear target and return to selection mode
Q                           quit
```

Interactive selection intentionally runs the detector on every displayed frame
only while choosing a target. After you click a target, it switches back to the
faster detector-plus-optical-flow tracking path. Press `Space` to show the
dense selectable boxes again if you want to switch targets. The selected target
is drawn green; other selectable boxes are yellow.

Useful performance flags:

```text
--resize-width 640             faster field default for common laptops
--resize-width 960             quality-biased option for smaller/farther targets
--imgsz 512                    faster YOLO inference than the default 640
--max-det 50                   faster post-processing in dense scenes
--detect-every 15              run detector twice per second on 30 FPS video
--device mps                   try Apple Silicon GPU acceleration
--device 0 --half              try CUDA GPU with FP16 inference
```

On the current development machine, repeated 90-frame tests on
`videos/highway.mp4` found `--resize-width 640 --detect-every 15 --imgsz 512
--max-det 50` to be the best speed/stability tradeoff: about `18.5 FPS`
fast-loop processing, `96.7%` locked frames, and zero dropped frames when not
paced to real time. When paced as a live 30 FPS source, the code may still drop
frames on CPU-only machines; the report records dropped frames and latency so
you can see when the machine is falling behind.

## Live Tracking Benchmark

Benchmark candidate live settings:

```bash
python3 -m aerial_vision.benchmark_live videos/highway.mp4 \
  --profile vehicles \
  --target-id 1 \
  --out live_benchmarks/highway \
  --max-frames 90 \
  --resize-widths 960,640 \
  --detect-everies 15 \
  --imgsizes 512 \
  --max-dets 50 \
  --trials 2
```

The benchmark writes:

```text
live_benchmarks/highway/benchmark.json
```

The score rewards processed FPS and stable target lock, while penalizing lost
target states and dropped frames. Use `--realtime` when you want to benchmark
against the source FPS clock and measure frame dropping. Use the default
fast-loop mode when you want raw tracking throughput.

This is an early prototype for operator-selected target lock. It does not
control a drone yet.

`field_runs/`, `tracking_runs/`, `live_runs/`, and `videos/` are ignored by git
so large local run artifacts are not committed accidentally.

## Recommended Next Milestone

The next best milestone is a small local review server:

```text
review.html
→ save decisions directly into the run folder
→ avoid manual browser downloads
```

That would make field review smoother while keeping the current static HTML
workflow as the portable fallback.

## Tests

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
