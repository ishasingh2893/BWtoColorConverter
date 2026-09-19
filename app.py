import base64
import mimetypes
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import requests
from flask import Flask, jsonify, render_template, request

from linear_color_model import colorize


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "linear_color_model.npz"
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
    return render_template("index.html", model_ready=MODEL_PATH.exists())


@app.post("/api/colorize")
def api_colorize():
    if not MODEL_PATH.exists():
        return (
            jsonify(
                {
                    "error": (
                        "Model file linear_color_model.npz is missing. "
                        "Run transformtrainingdata.py, then train the linear model."
                    )
                }
            ),
            400,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="bw_colorizer_") as tmp:
            temp_dir = Path(tmp)
            input_path = save_request_image(temp_dir)
            output_path = temp_dir / "colorized.png"
            colorize(input_path, MODEL_PATH, output_path)
            return jsonify({"output": data_uri(output_path)})
    except requests.RequestException:
        return jsonify({"error": "Could not download the image from that link."}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
