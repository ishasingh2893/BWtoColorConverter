import base64
import mimetypes
import json
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

from knn_lookup_color_model import colorize as colorize_with_lookup
from linear_color_model import colorize as colorize_with_linear


BASE_DIR = Path(__file__).resolve().parent
LOOKUP_MODEL_PATH = BASE_DIR / "knn_lookup_color_model.npz"
LINEAR_MODEL_PATH = BASE_DIR / "linear_color_model.npz"
MAX_IMAGE_BYTES = 12 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 1024 * 1024


class ImageLinkParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.candidate = ""

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        values = dict(attrs)
        if tag == "meta" and values.get("property") in {"og:image", "twitter:image"}:
            content = values.get("content")
            if content and not self.candidate:
                self.candidate = urljoin(self.base_url, content)
        if tag == "img" and not self.candidate:
            src = values.get("src")
            if src:
                self.candidate = urljoin(self.base_url, src)


def data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def download_image(url: str) -> bytes:
    response = requests.get(url, timeout=15, headers={"user-agent": "BWColorizer/1.0"})
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";")[0].lower()

    if content_type.startswith("image/"):
        return response.content

    if "html" in content_type:
        parser = ImageLinkParser(url)
        parser.feed(response.text)
        if parser.candidate:
            return download_image(parser.candidate)

    raise ValueError("That link did not resolve to an image.")


def save_request_image(temp_dir: Path) -> Path:
    upload = request.files.get("photo")
    image_url = request.form.get("image_url", "").strip()
    input_path = temp_dir / "input_image"

    if upload and upload.filename:
        image_bytes = upload.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("Image is too large. Use a file under 12 MB.")
        input_path.write_bytes(image_bytes)
        return input_path

    if image_url:
        image_bytes = download_image(image_url)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("Linked image is too large. Use an image under 12 MB.")
        input_path.write_bytes(image_bytes)
        return input_path

    raise ValueError("Upload a portrait or provide an image link.")


@app.get("/")
def index():
    available_models = []
    if LINEAR_MODEL_PATH.exists():
        available_models.append("linear regression")
    if LOOKUP_MODEL_PATH.exists():
        available_models.extend(["KNN lookup", "hybrid KNN"])

    return render_template(
        "index.html",
        model_ready=LOOKUP_MODEL_PATH.exists() or LINEAR_MODEL_PATH.exists(),
        model_name=", ".join(available_models),
        portrait_guide=json.loads((BASE_DIR / "static" / "portrait-guide.json").read_text()),
    )


@app.errorhandler(413)
def image_too_large(error):
    return jsonify({"error": "Image is too large. Use a file under 12 MB."}), 413


@app.post("/api/prepare")
def api_prepare():
    """Resolve links and normalize orientation before the user positions the image."""
    try:
        with tempfile.TemporaryDirectory(prefix="bw_prepare_") as tmp:
            path = save_request_image(Path(tmp))
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGBA")
                image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
                background = Image.new("RGBA", image.size, "white")
                background.alpha_composite(image)
                output = Path(tmp) / "prepared.png"
                background.convert("RGB").save(output)
            return jsonify({"image": data_uri(output)})
    except requests.RequestException:
        return jsonify({"error": "Could not download the image from that link."}), 400
    except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify({"error": "Unable to open this image. Choose a PNG, JPG, or WebP portrait."}), 400


@app.post("/api/colorize")
def api_colorize():
    if not LOOKUP_MODEL_PATH.exists() and not LINEAR_MODEL_PATH.exists():
        return jsonify({"error": "Train a color model before generating results."}), 400

    try:
        with tempfile.TemporaryDirectory(prefix="bw_colorizer_") as tmp:
            temp_dir = Path(tmp)
            upload = request.files.get("photo")
            if not upload or request.form.get("manual_crop") != "true":
                raise ValueError("Position your portrait in the crop tool before colorizing.")
            input_path = save_request_image(temp_dir)
            portrait_path = temp_dir / "portrait.png"
            with Image.open(input_path) as image:
                if image.size != (240, 240):
                    raise ValueError("The crop must be 240 × 240 pixels. Please use the crop tool.")
                image.convert("RGB").save(portrait_path)

            # The entire manually aligned frame is foreground. Supplying this mask
            # also prevents hybrid KNN from invoking its automatic face detector.
            mask_path = temp_dir / "full_frame.png"
            Image.new("L", (240, 240), 255).save(mask_path)
            outputs = {}
            if LINEAR_MODEL_PATH.exists():
                output = temp_dir / "linear.png"
                colorize_with_linear(portrait_path, LINEAR_MODEL_PATH, output, mask_path)
                outputs["linear"] = data_uri(output)
            if LOOKUP_MODEL_PATH.exists():
                for name, hybrid in (("knn", False), ("hybrid", True)):
                    output = temp_dir / f"{name}.png"
                    colorize_with_lookup(
                        portrait_path, LOOKUP_MODEL_PATH, output,
                        use_hybrid=hybrid, subject_mask_path=mask_path,
                    )
                    outputs[name] = data_uri(output)
            return jsonify({"input": data_uri(portrait_path), "outputs": outputs})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
