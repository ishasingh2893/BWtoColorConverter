"""Run with .venv/bin/python -m unittest discover -s tests."""
import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image

import app


def upload(image):
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    buffer.seek(0)
    return buffer, "portrait.png"


class ManualCropTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_prepare_composites_transparency_and_preserves_aspect(self):
        response = self.client.post("/api/prepare", data={
            "photo": upload(Image.new("RGBA", (400, 600), (0, 0, 0, 0))),
        })
        self.assertEqual(response.status_code, 200)
        image = Image.open(io.BytesIO(base64.b64decode(response.json["image"].split(",")[1])))
        self.assertEqual(image.size, (400, 600))
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_manual_crop_reaches_every_model_without_detection(self):
        crop = Image.new("RGB", (240, 240), (120, 120, 120))
        crop.putpixel((30, 40), (10, 20, 30))
        with patch("knn_lookup_color_model.detect_largest_face", side_effect=AssertionError("Detection ran")):
            response = self.client.post("/api/colorize", data={
                "photo": upload(crop), "manual_crop": "true",
            })
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(set(response.json["outputs"]), {"linear", "knn", "hybrid"})
        result = Image.open(io.BytesIO(base64.b64decode(response.json["input"].split(",")[1])))
        self.assertEqual(result.size, crop.size)
        self.assertEqual(result.tobytes(), crop.tobytes())

    def test_rejects_uncropped_or_invalid_images(self):
        self.assertEqual(self.client.post("/api/colorize").status_code, 400)
        response = self.client.post("/api/colorize", data={
            "photo": upload(Image.new("RGB", (400, 600))), "manual_crop": "true",
        })
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/prepare", data={"photo": (io.BytesIO(b"invalid"), "bad.png")})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
