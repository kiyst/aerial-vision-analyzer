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

## Project Philosophy

This project is built around a practical field-review workflow:

```text
aerial image or drone frame
→ run object detection
→ save annotated output
→ human reviews positives
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
- The current system works on images, not video streams yet.

## Recommended Next Milestone

The next major software step is video scanning:

```bash
python3 -m aerial_vision.scan_video videos/flight01.mp4 \
  --profile general \
  --sample-every-sec 1 \
  --out field_runs/flight01
```

Planned behavior:

```text
video file
→ sample frames every N seconds
→ run detection profile
→ save positive frames
→ save annotated frames
→ save per-frame JSON
→ write a review report with timestamps
```

That would turn this from a single-image tool into a field-review workflow for
drone footage.

## Tests

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

