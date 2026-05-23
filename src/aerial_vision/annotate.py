from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from aerial_vision.detection import ObjectDetection


def draw_detections(
    image_path: Path,
    detections: list[ObjectDetection],
    output_path: Path,
) -> None:
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for detection in detections:
            box = detection.box.to_xyxy()
            label = f"{detection.label} {detection.confidence:.2f}"
            if detection.polygon:
                draw.line(detection.polygon + (detection.polygon[0],), fill="red", width=3)
            else:
                draw.rectangle(box, outline="red", width=3)
            text_bbox = draw.textbbox((box[0], box[1]), label, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((box[0], box[1]), label, fill="white", font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
