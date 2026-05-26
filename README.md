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
- The current video workflow scans saved video files, not live camera streams.
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
  tracked_preview.mp4
```

`tracks.json` includes each exported track's boxes, timestamps, class labels,
confidence scores, color histogram, shape data, and basic identity scores.
The preview video draws track IDs on top of the source footage.

Useful flags:

```text
--tracker bytetrack.yaml       default Ultralytics tracker
--tracker botsort.yaml         stronger tracker option to compare
--resize-width 960             faster tracking while preserving useful detail
--sample-every-sec 1           track sampled frames instead of every frame
--min-track-frames 5           only export tracks seen at least 5 frames
--max-frames 90                quick smoke test
```

On `videos/highway.mp4`, a 90-frame comparison kept nearly the same useful
long-running tracks at 960px wide as the full 1920px source. In that test,
960px preserved the label mix and long-track counts better than dropping to
720px, while producing much smaller preview files. Use full resolution for
final/high-confidence analysis, but start tracking experiments at 960px.

This is an early prototype for operator-selected target lock. It does not
control a drone yet.

`field_runs/`, `tracking_runs/`, and `videos/` are ignored by git so large
local run artifacts are not committed accidentally.

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
