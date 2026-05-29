import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import aerial_vision.classify as classify_module
from aerial_vision.analysis import ImageAnalysis
from aerial_vision.annotate import draw_detections
from aerial_vision.classify import apply_profile_defaults, clean_previous_outputs
from aerial_vision.detection import (
    BoundingBox,
    GeometryRule,
    ObjectDetection,
    collapse_labels,
    filter_by_geometry,
    filter_detections,
)
from aerial_vision.export_review import export_confirmed, safe_run_path
from aerial_vision.benchmark_live import live_score, parse_int_list, parse_optional_int_list, summarize_result
from aerial_vision.live_track import observation_contains_point, parse_source, resolve_label_filter, select_observation_at_point
from aerial_vision.live_source import frame_time_sec, playback_status, realtime_delay_sec, should_drop_frame
from aerial_vision.pick_target import selected_track_id
from aerial_vision.pipeline import DetectionSettings, DetectorBundle, analyze_image_with_detectors
from aerial_vision.scan_video import (
    detection_difference,
    format_timestamp,
    frame_difference,
    prepare_output_dir,
    render_review_html,
    timestamp_display,
)
from aerial_vision.track_video import (
    class_ids_for_model,
    color_histogram,
    cosine_similarity,
    identity_score,
    observation_from_box,
    points_for_box,
    profile_min_confidence,
    select_target_observation,
    target_center_offsets,
    target_lock_state,
    track_video,
    yolo_track_kwargs,
)
from aerial_vision.tiling import bbox_iou, generate_tiles, non_max_suppression, shift_detections


class FakeDetector:
    def __init__(self, detections: list[ObjectDetection]) -> None:
        self.detections = detections
        self.calls = 0

    def detect(
        self,
        image_path: str | Path,
        *,
        min_confidence: float = 0.25,
        objects_of_interest: set[str] | None = None,
    ) -> list[ObjectDetection]:
        self.calls += 1
        wanted = objects_of_interest or {detection.label for detection in self.detections}
        return filter_detections(self.detections, objects_of_interest=wanted, min_confidence=min_confidence)


class DetectionTest(unittest.TestCase):
    def test_bounding_box_geometry(self) -> None:
        box = BoundingBox(x_min=10, y_min=20, x_max=40, y_max=80)

        self.assertEqual(box.width, 30)
        self.assertEqual(box.height, 60)
        self.assertEqual(box.area, 1800)
        self.assertEqual(box.center, (25, 50))

    def test_filters_objects_of_interest(self) -> None:
        detections = [
            ObjectDetection("car", 0.91, BoundingBox(0, 0, 10, 10)),
            ObjectDetection("airplane", 0.88, BoundingBox(10, 10, 20, 20)),
            ObjectDetection("person", 0.10, BoundingBox(20, 20, 30, 30)),
        ]

        filtered = filter_detections(detections, objects_of_interest={"car", "person"}, min_confidence=0.25)

        self.assertEqual([detection.label for detection in filtered], ["car"])

    def test_detection_serializes_to_dict(self) -> None:
        detection = ObjectDetection(
            "truck",
            0.8,
            BoundingBox(1, 2, 11, 22),
            source_model="example.pt",
            polygon=((1, 2), (11, 2), (11, 22), (1, 22)),
        )

        self.assertEqual(
            detection.to_dict(),
            {
                "label": "truck",
                "confidence": 0.8,
                "box": {"x_min": 1, "y_min": 2, "x_max": 11, "y_max": 22},
                "source_model": "example.pt",
                "polygon": ((1, 2), (11, 2), (11, 22), (1, 22)),
                "center": (6.0, 12.0),
                "area_px": 200,
            },
        )

    def test_image_analysis_summarizes_detections(self) -> None:
        analysis = ImageAnalysis(
            image_path="sample.png",
            image_width=100,
            image_height=80,
            detections=[
                ObjectDetection("Car", 0.9, BoundingBox(0, 0, 10, 10)),
                ObjectDetection("car", 0.7, BoundingBox(0, 0, 20, 20)),
                ObjectDetection("person", 0.5, BoundingBox(0, 0, 5, 5)),
            ],
        )

        payload = analysis.to_dict()

        self.assertEqual(payload["image"]["width_px"], 100)
        self.assertEqual(payload["summary"]["total_detections"], 3)
        self.assertEqual(payload["summary"]["counts_by_label"], {"car": 2, "person": 1})
        self.assertAlmostEqual(payload["summary"]["average_confidence"], 0.7)
        self.assertEqual(payload["summary"]["largest_detection"]["label"], "car")

    def test_draw_detections_writes_annotated_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "input.png"
            output_path = temp_path / "annotated.png"
            Image.new("RGB", (100, 100), "white").save(image_path)

            draw_detections(
                image_path,
                [ObjectDetection("car", 0.9, BoundingBox(10, 10, 50, 50))],
                output_path,
            )

            self.assertTrue(output_path.exists())

    def test_classify_main_uses_analysis_detections_for_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "input.png"
            output_path = temp_path / "annotated.jpg"
            Image.new("RGB", (100, 100), "white").save(image_path)
            detections = [ObjectDetection("car", 0.9, BoundingBox(10, 10, 20, 20))]
            analysis = ImageAnalysis(
                image_path=str(image_path),
                image_width=100,
                image_height=100,
                detections=detections,
            )

            with (
                patch.object(classify_module, "analyze_image", return_value=analysis),
                patch.object(classify_module, "draw_detections") as draw_mock,
                patch("builtins.print"),
                patch("sys.argv", ["classify", str(image_path), "--annotated-out", str(output_path)]),
            ):
                classify_module.main()

            draw_mock.assert_called_once_with(image_path, detections, output_path)

    def test_clean_previous_outputs_removes_files_from_outputs_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs_dir = Path(temp_dir) / "outputs"
            outputs_dir.mkdir()
            stale_json = outputs_dir / "old.json"
            stale_image = outputs_dir / "old.jpg"
            stale_json.write_text("{}", encoding="utf-8")
            stale_image.write_text("image", encoding="utf-8")

            clean_previous_outputs([outputs_dir / "new.json"])

            self.assertFalse(stale_json.exists())
            self.assertFalse(stale_image.exists())

    def test_clean_previous_outputs_ignores_non_outputs_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            important_file = temp_path / "important.json"
            important_file.write_text("{}", encoding="utf-8")

            clean_previous_outputs([temp_path / "new.json"])

            self.assertTrue(important_file.exists())

    def test_generate_tiles_covers_edges(self) -> None:
        tiles = generate_tiles(image_width=1000, image_height=700, tile_size=400, overlap=100)

        self.assertEqual(tiles[0].x, 0)
        self.assertEqual(tiles[0].y, 0)
        self.assertEqual(tiles[-1].x + tiles[-1].width, 1000)
        self.assertEqual(tiles[-1].y + tiles[-1].height, 700)

    def test_shift_detections_offsets_boxes_and_polygons(self) -> None:
        detections = [
            ObjectDetection(
                "car",
                0.8,
                BoundingBox(1, 2, 11, 22),
                polygon=((1, 2), (11, 2), (11, 22), (1, 22)),
            )
        ]

        shifted = shift_detections(detections, x_offset=100, y_offset=200)

        self.assertEqual(shifted[0].box.to_xyxy(), (101, 202, 111, 222))
        self.assertEqual(shifted[0].polygon, ((101, 202), (111, 202), (111, 222), (101, 222)))

    def test_bbox_iou(self) -> None:
        first = BoundingBox(0, 0, 10, 10)
        second = BoundingBox(5, 5, 15, 15)

        self.assertAlmostEqual(bbox_iou(first, second), 25 / 175)

    def test_non_max_suppression_keeps_highest_confidence_duplicate(self) -> None:
        detections = [
            ObjectDetection("car", 0.7, BoundingBox(0, 0, 10, 10)),
            ObjectDetection("car", 0.9, BoundingBox(1, 1, 11, 11)),
            ObjectDetection("person", 0.6, BoundingBox(1, 1, 11, 11)),
        ]

        kept = non_max_suppression(detections, iou_threshold=0.4)

        self.assertEqual([detection.label for detection in kept], ["car", "person"])
        self.assertEqual(kept[0].confidence, 0.9)

    def test_filter_by_geometry_drops_unrealistic_large_boxes(self) -> None:
        detections = [
            ObjectDetection("car", 0.8, BoundingBox(10, 10, 30, 30)),
            ObjectDetection("truck", 0.8, BoundingBox(0, 0, 900, 900)),
        ]

        filtered = filter_by_geometry(
            detections,
            image_width=1000,
            image_height=1000,
            max_area_ratio=0.1,
        )

        self.assertEqual([detection.label for detection in filtered], ["car"])

    def test_filter_by_geometry_allows_per_label_overrides(self) -> None:
        detections = [
            ObjectDetection("car", 0.8, BoundingBox(0, 0, 300, 300)),
            ObjectDetection("truck", 0.8, BoundingBox(0, 0, 300, 300)),
        ]

        filtered = filter_by_geometry(
            detections,
            image_width=1000,
            image_height=1000,
            max_area_ratio=0.05,
            per_label_rules={"truck": GeometryRule(max_area_ratio=0.1)},
        )

        self.assertEqual([detection.label for detection in filtered], ["truck"])

    def test_collapse_labels_renames_grouped_labels(self) -> None:
        detections = [
            ObjectDetection("horse", 0.8, BoundingBox(0, 0, 10, 10)),
            ObjectDetection("bird", 0.7, BoundingBox(10, 10, 20, 20)),
            ObjectDetection("car", 0.9, BoundingBox(20, 20, 30, 30)),
        ]

        collapsed = collapse_labels(detections, {"animal": {"horse", "bird"}})

        self.assertEqual([detection.label for detection in collapsed], ["animal", "animal", "car"])

    def test_apply_profile_defaults_sets_general_workflow(self) -> None:
        args = Namespace(
            profile="general",
            model="yolo11n.pt",
            min_confidence=0.25,
            objects=None,
            tile_size=None,
            tile_overlap=128,
            geometry_preset=None,
            extra_model=None,
            extra_objects=None,
            extra_min_confidence=None,
            collapse_animals=False,
        )

        updated = apply_profile_defaults(args)

        self.assertEqual(updated.model, "models/best.pt")
        self.assertEqual(updated.min_confidence, 0.15)
        self.assertIn("pedestrian", updated.objects)
        self.assertEqual(updated.extra_model, "yolo11x.pt")
        self.assertEqual(updated.extra_min_confidence, 0.05)
        self.assertTrue(updated.collapse_animals)

    def test_apply_profile_defaults_preserves_explicit_overrides(self) -> None:
        args = Namespace(
            profile="animals",
            model="custom.pt",
            min_confidence=0.2,
            objects=None,
            tile_size=None,
            tile_overlap=128,
            geometry_preset=None,
            extra_model=None,
            extra_objects=None,
            extra_min_confidence=None,
            collapse_animals=False,
        )

        updated = apply_profile_defaults(args)

        self.assertEqual(updated.model, "custom.pt")
        self.assertEqual(updated.min_confidence, 0.2)
        self.assertEqual(updated.objects, ["horse", "cow", "sheep", "dog", "cat", "bird"])

    def test_analyze_image_with_detectors_reuses_supplied_detector_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            primary = FakeDetector([ObjectDetection("car", 0.8, BoundingBox(10, 10, 20, 20))])
            extra = FakeDetector([ObjectDetection("horse", 0.8, BoundingBox(30, 30, 45, 45))])
            settings = DetectionSettings(
                objects=["car"],
                min_confidence=0.25,
                extra_model="unused-by-test.pt",
                extra_objects=["horse"],
                extra_min_confidence=0.25,
                collapse_animals=True,
            )

            analysis = analyze_image_with_detectors(
                image_path,
                settings,
                DetectorBundle(primary=primary, extra=extra),
            )

            self.assertEqual(primary.calls, 1)
            self.assertEqual(extra.calls, 1)
            self.assertEqual(analysis.counts_by_label, {"animal": 1, "car": 1})

    def test_video_timestamp_formatting(self) -> None:
        self.assertEqual(format_timestamp(0), "00-00-00")
        self.assertEqual(format_timestamp(65), "00-01-05")
        self.assertEqual(format_timestamp(3661), "01-01-01")
        self.assertEqual(timestamp_display(3661), "01:01:01")

    def test_prepare_output_dir_clears_previous_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "run"
            stale_dir = out_dir / "frames"
            stale_dir.mkdir(parents=True)
            stale_file = stale_dir / "stale.jpg"
            stale_file.write_text("old", encoding="utf-8")

            prepare_output_dir(out_dir)

            self.assertTrue(out_dir.exists())
            self.assertFalse(stale_file.exists())

    def test_render_review_html_contains_event_links(self) -> None:
        html = render_review_html(
            {
                "video": {"path": "videos/test.mp4", "duration_sec": 10.0},
                "profile": "animals",
                "sample_every_sec": 1,
                "frames_checked": 3,
                "positive_frames": 1,
                "events": [
                    {
                        "timestamp": "00:00:01",
                        "counts_by_label": {"animal": 2},
                        "total_detections": 2,
                        "average_confidence": 0.5,
                        "annotated_image": "positives/00-00-01.jpg",
                        "detections_json": "detections/00-00-01.json",
                    }
                ],
            }
        )

        self.assertIn("Aerial Vision Dashboard", html)
        self.assertIn("positives/00-00-01.jpg", html)
        self.assertIn('"animal": 2', html)
        self.assertIn("review_decisions.json", html)
        self.assertIn("review.html", html)
        self.assertIn("buildDecisionPayload", html)
        self.assertIn("downloadReviewPackage", html)

    def test_frame_difference_detects_visual_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            black = temp_path / "black.jpg"
            white = temp_path / "white.jpg"
            Image.new("RGB", (10, 10), "black").save(black)
            Image.new("RGB", (10, 10), "white").save(white)

            self.assertEqual(frame_difference(black, black), 0)
            self.assertGreater(frame_difference(black, white), 0.9)

    def test_detection_difference_compares_labels_and_boxes(self) -> None:
        first = [ObjectDetection("animal", 0.8, BoundingBox(10, 10, 20, 20))]
        similar = [ObjectDetection("animal", 0.7, BoundingBox(11, 10, 21, 20))]
        different = [ObjectDetection("car", 0.7, BoundingBox(80, 80, 100, 100))]

        self.assertLess(
            detection_difference(first, similar, image_width=100, image_height=100),
            0.01,
        )
        self.assertGreater(
            detection_difference(first, different, image_width=100, image_height=100),
            0.4,
        )

    def test_export_confirmed_copies_only_confirmed_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            positives_dir = run_dir / "positives"
            detections_dir = run_dir / "detections"
            positives_dir.mkdir(parents=True)
            detections_dir.mkdir()
            (positives_dir / "confirmed.jpg").write_text("confirmed-image", encoding="utf-8")
            (positives_dir / "dismissed.jpg").write_text("dismissed-image", encoding="utf-8")
            (detections_dir / "confirmed.json").write_text("{}", encoding="utf-8")
            (detections_dir / "dismissed.json").write_text("{}", encoding="utf-8")
            decisions_path = run_dir / "review_decisions.json"
            decisions_path.write_text(
                """
                {
                  "video": {"path": "videos/test.mp4"},
                  "profile": "general",
                  "events": [
                    {
                      "id": "1",
                      "status": "confirmed",
                      "timestamp": "00:00:01",
                      "annotated_image": "positives/confirmed.jpg",
                      "detections_json": "detections/confirmed.json"
                    },
                    {
                      "id": "2",
                      "status": "dismissed",
                      "timestamp": "00:00:02",
                      "annotated_image": "positives/dismissed.jpg",
                      "detections_json": "detections/dismissed.json"
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            output = export_confirmed(decisions_path)

            self.assertEqual(output["confirmed_count"], 1)
            self.assertTrue((run_dir / "confirmed_export" / "images" / "confirmed.jpg").exists())
            self.assertTrue((run_dir / "confirmed_export" / "detections" / "confirmed.json").exists())
            self.assertFalse((run_dir / "confirmed_export" / "images" / "dismissed.jpg").exists())
            exported = json.loads((run_dir / "confirmed_export" / "confirmed_events.json").read_text(encoding="utf-8"))
            self.assertEqual(exported["events"][0]["exported_annotated_image"], "images/confirmed.jpg")

    def test_safe_run_path_rejects_paths_outside_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                safe_run_path(Path(temp_dir), "../outside.jpg")

    def test_color_histogram_and_similarity_match_same_crop(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[:, :] = [10, 120, 240]
        box = BoundingBox(0, 0, 20, 20)

        histogram = color_histogram(frame, box)

        self.assertAlmostEqual(sum(histogram), 1.0)
        self.assertAlmostEqual(cosine_similarity(histogram, histogram), 1.0)

    def test_observation_identity_scores_consistent_shape_and_color(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:30, 10:50] = [200, 200, 200]
        first = observation_from_box(
            frame=frame,
            frame_index=0,
            timestamp_sec=0,
            track_id=1,
            label="car",
            confidence=0.9,
            box=BoundingBox(10, 10, 50, 30),
        )
        second = observation_from_box(
            frame=frame,
            frame_index=1,
            timestamp_sec=1,
            track_id=1,
            label="car",
            confidence=0.9,
            box=BoundingBox(12, 10, 52, 30),
            previous=first,
        )

        self.assertIsNotNone(second.identity_score)
        self.assertGreater(second.identity_score, 0.9)
        self.assertGreater(identity_score(1.0, 1.0), 0.99)

    def test_track_video_rejects_invalid_resize_width(self) -> None:
        with self.assertRaises(ValueError):
            track_video(
                Path("missing.mp4"),
                profile="vehicles",
                out_dir=Path("tracking_runs/test"),
                resize_width=0,
            )

    def test_track_video_requires_target_for_sparse_detection(self) -> None:
        with self.assertRaises(ValueError):
            track_video(
                Path("missing.mp4"),
                profile="vehicles",
                out_dir=Path("tracking_runs/test"),
                detect_every=10,
            )

    def test_class_ids_for_model_maps_requested_names(self) -> None:
        class Model:
            names = {0: "person", 1: "car", 2: "truck"}

        self.assertEqual(class_ids_for_model(Model(), {"car", "truck"}), [1, 2])

    def test_yolo_track_kwargs_includes_speed_controls(self) -> None:
        kwargs = yolo_track_kwargs(
            tracker="bytetrack.yaml",
            class_ids=[1, 2],
            min_confidence=0.2,
            imgsz=512,
            max_det=50,
            device="cpu",
            half=True,
        )

        self.assertEqual(kwargs["classes"], [1, 2])
        self.assertEqual(kwargs["conf"], 0.2)
        self.assertEqual(kwargs["imgsz"], 512)
        self.assertEqual(kwargs["max_det"], 50)
        self.assertEqual(kwargs["device"], "cpu")
        self.assertTrue(kwargs["half"])

    def test_profile_min_confidence_uses_pipeline_profiles(self) -> None:
        self.assertEqual(profile_min_confidence("vehicles"), 0.15)

    def test_select_target_observation_prefers_requested_label_confidence(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        car = observation_from_box(
            frame=frame,
            frame_index=0,
            timestamp_sec=0,
            track_id=1,
            label="car",
            confidence=0.99,
            box=BoundingBox(0, 0, 10, 10),
        )
        weak_truck = observation_from_box(
            frame=frame,
            frame_index=0,
            timestamp_sec=0,
            track_id=2,
            label="truck",
            confidence=0.5,
            box=BoundingBox(0, 0, 20, 20),
        )
        strong_truck = observation_from_box(
            frame=frame,
            frame_index=0,
            timestamp_sec=0,
            track_id=3,
            label="truck",
            confidence=0.8,
            box=BoundingBox(0, 0, 15, 15),
        )

        selected = select_target_observation([car, weak_truck, strong_truck], "truck")

        self.assertIsNotNone(selected)
        self.assertEqual(selected.track_id, 3)

    def test_live_source_frame_time_and_delay(self) -> None:
        self.assertEqual(frame_time_sec(60, 30), 2.0)
        self.assertEqual(realtime_delay_sec(video_time_sec=2.0, wall_elapsed_sec=1.25), 0.75)
        self.assertEqual(realtime_delay_sec(video_time_sec=2.0, wall_elapsed_sec=2.25), 0.0)

    def test_live_source_drop_decision(self) -> None:
        self.assertFalse(should_drop_frame(video_time_sec=10.0, wall_elapsed_sec=10.4, max_latency_sec=0.5))
        self.assertTrue(should_drop_frame(video_time_sec=10.0, wall_elapsed_sec=10.6, max_latency_sec=0.5))

    def test_live_source_playback_status_reports_latency(self) -> None:
        status = playback_status(
            frame_index=30,
            fps=30,
            wall_elapsed_sec=1.25,
            processed_frames=20,
            dropped_frames=10,
        )

        self.assertEqual(status.video_time_sec, 1.0)
        self.assertEqual(status.latency_sec, 0.25)
        self.assertAlmostEqual(status.real_time_factor, 0.8)
        self.assertEqual(status.to_dict()["dropped_frames"], 10)

    def test_live_track_parse_source_accepts_camera_index_or_path(self) -> None:
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source("videos/highway.mp4"), "videos/highway.mp4")

    def test_live_track_selects_smallest_box_at_clicked_point(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        large = observation_from_box(
            frame=frame,
            frame_index=0,
            timestamp_sec=0,
            track_id=1,
            label="car",
            confidence=0.8,
            box=BoundingBox(10, 10, 80, 80),
        )
        small = observation_from_box(
            frame=frame,
            frame_index=0,
            timestamp_sec=0,
            track_id=2,
            label="truck",
            confidence=0.8,
            box=BoundingBox(20, 20, 40, 40),
        )

        self.assertTrue(observation_contains_point(small, 25, 25))
        self.assertEqual(select_observation_at_point([large, small], 25, 25), small)
        self.assertIsNone(select_observation_at_point([large, small], 90, 90))

    def test_live_track_resolves_interactive_label_filter(self) -> None:
        available = {"car", "truck", "bus"}

        self.assertEqual(resolve_label_filter(available, None), available)
        self.assertEqual(resolve_label_filter(available, ["all"]), available)
        self.assertEqual(resolve_label_filter(available, ["Truck", "bus"]), {"truck", "bus"})
        with self.assertRaises(ValueError):
            resolve_label_filter(available, ["plane"])

    def test_benchmark_live_parses_comma_lists(self) -> None:
        self.assertEqual(parse_int_list("10, 15"), [10, 15])
        self.assertEqual(parse_optional_int_list("native,512"), [None, 512])

    def test_benchmark_live_scores_fast_stable_runs_higher(self) -> None:
        stable = {
            "processed_fps": 20.0,
            "processed_frames": 100,
            "dropped_frames": 0,
            "state_counts": {"locked": 95, "lost": 5},
            "observations": [{} for _ in range(100)],
        }
        unstable = {
            "processed_fps": 20.0,
            "processed_frames": 100,
            "dropped_frames": 20,
            "state_counts": {"locked": 40, "lost": 60},
            "observations": [{} for _ in range(100)],
        }

        self.assertGreater(live_score(stable), live_score(unstable))
        self.assertEqual(summarize_result(stable)["locked_ratio"], 0.95)

    def test_selected_track_id_returns_choice_track_id(self) -> None:
        choices = [
            {"choice": 1, "track_id": 10},
            {"choice": 2, "track_id": 22},
        ]

        self.assertEqual(selected_track_id(choices, 2), 22)
        with self.assertRaises(ValueError):
            selected_track_id(choices, 3)

    def test_points_for_box_generates_grid_inside_box(self) -> None:
        points = points_for_box(BoundingBox(10, 20, 50, 60), frame_width=100, frame_height=100, grid_size=3)

        self.assertEqual(points.shape, (9, 1, 2))
        self.assertGreater(float(points[:, :, 0].min()), 10)
        self.assertLess(float(points[:, :, 0].max()), 50)

    def test_target_center_offsets_are_normalized(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        observation = observation_from_box(
            frame=frame,
            frame_index=0,
            timestamp_sec=0,
            track_id=1,
            label="car",
            confidence=0.9,
            box=BoundingBox(60, 40, 80, 60),
        )

        raw_offset, normalized_offset = target_center_offsets(observation, frame_width=100, frame_height=100)

        self.assertEqual(raw_offset, (20.0, 0.0))
        self.assertEqual(normalized_offset, (0.4, 0.0))

    def test_target_lock_state_marks_lost_target(self) -> None:
        lock = target_lock_state(
            None,
            [],
            target_id=7,
            frame_index=10,
            timestamp_sec=1.0,
            frame_width=100,
            frame_height=100,
        )

        self.assertEqual(lock.state, "lost")
        self.assertIsNone(lock.center_offset)

    def test_target_lock_state_marks_overlap_risk(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        target = observation_from_box(
            frame=frame,
            frame_index=0,
            timestamp_sec=0,
            track_id=1,
            label="car",
            confidence=0.9,
            box=BoundingBox(10, 10, 50, 50),
        )
        nearby = observation_from_box(
            frame=frame,
            frame_index=0,
            timestamp_sec=0,
            track_id=2,
            label="car",
            confidence=0.9,
            box=BoundingBox(20, 20, 60, 60),
        )

        lock = target_lock_state(
            target,
            [target, nearby],
            target_id=1,
            frame_index=0,
            timestamp_sec=0,
            frame_width=100,
            frame_height=100,
            overlap_threshold=0.1,
        )

        self.assertEqual(lock.state, "id_switch_risk")
        self.assertGreater(lock.overlap_risk, 0.1)


if __name__ == "__main__":
    unittest.main()
