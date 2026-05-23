import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image

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
from aerial_vision.tiling import bbox_iou, generate_tiles, non_max_suppression, shift_detections


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


if __name__ == "__main__":
    unittest.main()
