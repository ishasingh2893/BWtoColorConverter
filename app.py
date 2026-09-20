import base64
import mimetypes
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests
from flask import Flask, jsonify, render_template, request

from knn_lookup_color_model import colorize as colorize_with_lookup
from linear_color_model import colorize as colorize_with_linear
from portrait_preprocess import (
    crop_face_portrait,
    save_edge_detection_preview,
    save_outermost_layer_preview,
)


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
    )


@app.post("/api/colorize")
def api_colorize():
    if not LOOKUP_MODEL_PATH.exists() and not LINEAR_MODEL_PATH.exists():
        return (
            jsonify(
                {
                    "error": (
                        "Model artifact is missing. Train knn_lookup_color_model.npz "
                        "or linear_color_model.npz before generating results."
                    )
                }
            ),
            400,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="bw_colorizer_") as tmp:
            temp_dir = Path(tmp)
            input_path = save_request_image(temp_dir)
            portrait_path = temp_dir / "portrait.png"
            edge_path = temp_dir / "edges.png"
            outer_layer_path = temp_dir / "outer_layer.png"
            output_path = temp_dir / "colorized.png"
            crop_face_portrait(input_path, portrait_path)
            save_edge_detection_preview(portrait_path, edge_path)
            save_outermost_layer_preview(portrait_path, outer_layer_path)
            outputs = {}

            if LINEAR_MODEL_PATH.exists():
                linear_output_path = temp_dir / "linear.png"
                colorize_with_linear(portrait_path, LINEAR_MODEL_PATH, linear_output_path)
                outputs["linear"] = data_uri(linear_output_path)

            if LOOKUP_MODEL_PATH.exists():
                knn_output_path = temp_dir / "knn.png"
                hybrid_output_path = temp_dir / "hybrid.png"
                colorize_with_lookup(
                    portrait_path,
                    LOOKUP_MODEL_PATH,
                    knn_output_path,
                    use_hybrid=False,
                )
                colorize_with_lookup(
                    portrait_path,
                    LOOKUP_MODEL_PATH,
                    hybrid_output_path,
                    use_hybrid=True,
                )
                outputs["knn"] = data_uri(knn_output_path)
                outputs["hybrid"] = data_uri(hybrid_output_path)

            return jsonify(
                {
                    "input": data_uri(portrait_path),
                    "edges": data_uri(edge_path),
                    "outer_layer": data_uri(outer_layer_path),
                    "outputs": outputs,
                }
            )
    except requests.RequestException:
        return jsonify({"error": "Could not download the image from that link."}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
