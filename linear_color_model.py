import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


IMAGE_SIZE = (240, 240)


def add_white_background(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    combined = Image.alpha_composite(white_bg, img)
    return combined.convert("RGB")


def image_to_lab_channels(image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = Image.open(image_path)
    img = add_white_background(img)
    img = img.resize(IMAGE_SIZE)
    lab = img.convert("LAB")
    l_channel, a_channel, b_channel = lab.split()
    return (
        np.asarray(l_channel, dtype=np.float32).reshape(-1),
        np.asarray(a_channel, dtype=np.float32).reshape(-1),
        np.asarray(b_channel, dtype=np.float32).reshape(-1),
    )


def iter_image_paths(data_folder: Path) -> list:
    extensions = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    if not data_folder.exists():
        raise FileNotFoundError(f"Training folder does not exist: {data_folder}")
    image_paths = [
        path
        for path in sorted(data_folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
    ]
    if not image_paths:
        raise FileNotFoundError(f"No training images found in: {data_folder}")
    return image_paths


def load_training_channels(data_folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l_values = []
    a_values = []
    b_values = []

    for image_path in iter_image_paths(data_folder):
        l_channel, a_channel, b_channel = image_to_lab_channels(image_path)
        l_values.append(l_channel)
        a_values.append(a_channel)
        b_values.append(b_channel)

    return (
        np.vstack(l_values),
        np.vstack(a_values),
        np.vstack(b_values),
    )


def fit_streaming_pixelwise_linear_model(
    data_folder: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    num_pixels = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    sum_l = np.zeros(num_pixels, dtype=np.float64)
    sum_l2 = np.zeros(num_pixels, dtype=np.float64)
    sum_a = np.zeros(num_pixels, dtype=np.float64)
    sum_b = np.zeros(num_pixels, dtype=np.float64)
    sum_la = np.zeros(num_pixels, dtype=np.float64)
    sum_lb = np.zeros(num_pixels, dtype=np.float64)

    image_count = 0
    for image_path in iter_image_paths(data_folder):
        l_channel, a_channel, b_channel = image_to_lab_channels(image_path)
        l_values = l_channel.astype(np.float64)
        a_values = a_channel.astype(np.float64)
        b_values = b_channel.astype(np.float64)

        sum_l += l_values
        sum_l2 += l_values * l_values
        sum_a += a_values
        sum_b += b_values
        sum_la += l_values * a_values
        sum_lb += l_values * b_values
        image_count += 1

        if image_count % 1000 == 0:
            print(f"Processed {image_count} training images")

    mean_l = sum_l / image_count
    variance_l = (sum_l2 / image_count) - (mean_l * mean_l)

    mean_a = sum_a / image_count
    covariance_a = (sum_la / image_count) - (mean_l * mean_a)
    a_slope = np.divide(
        covariance_a,
        variance_l,
        out=np.zeros_like(covariance_a, dtype=np.float64),
        where=variance_l > 1e-6,
    )
    a_intercept = mean_a - a_slope * mean_l

    mean_b = sum_b / image_count
    covariance_b = (sum_lb / image_count) - (mean_l * mean_b)
    b_slope = np.divide(
        covariance_b,
        variance_l,
        out=np.zeros_like(covariance_b, dtype=np.float64),
        where=variance_l > 1e-6,
    )
    b_intercept = mean_b - b_slope * mean_l

    return (
        a_slope.astype(np.float32),
        a_intercept.astype(np.float32),
        b_slope.astype(np.float32),
        b_intercept.astype(np.float32),
        image_count,
    )


def load_transformed_channels(
    transformed_data_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not transformed_data_path.exists():
        raise FileNotFoundError(
            f"Transformed training data does not exist: {transformed_data_path}"
        )

    with np.load(transformed_data_path) as data:
        return (
            data["L"].astype(np.float32),
            data["A"].astype(np.float32),
            data["B"].astype(np.float32),
        )


def fit_pixelwise_linear_model(
    l_values: np.ndarray,
    target_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = l_values.mean(axis=0)
    y_mean = target_values.mean(axis=0)
    x_centered = l_values - x_mean
    y_centered = target_values - y_mean

    variance = np.mean(x_centered * x_centered, axis=0)
    covariance = np.mean(x_centered * y_centered, axis=0)

    slope = np.divide(
        covariance,
        variance,
        out=np.zeros_like(covariance, dtype=np.float32),
        where=variance > 1e-6,
    )
    intercept = y_mean - slope * x_mean
    return slope.astype(np.float32), intercept.astype(np.float32)


def train(
    output_path: Path,
    transformed_data_path: Optional[Path] = None,
    data_folder: Optional[Path] = None,
) -> None:
    if transformed_data_path is not None:
        l_values, a_values, b_values = load_transformed_channels(transformed_data_path)
        training_count = l_values.shape[0]
        a_slope, a_intercept = fit_pixelwise_linear_model(l_values, a_values)
        b_slope, b_intercept = fit_pixelwise_linear_model(l_values, b_values)
    elif data_folder is not None:
        (
            a_slope,
            a_intercept,
            b_slope,
            b_intercept,
            training_count,
        ) = fit_streaming_pixelwise_linear_model(data_folder)
    else:
        raise FileNotFoundError("Provide --transformed-data or --data for training")

    np.savez_compressed(
        output_path,
        a_slope=a_slope,
        a_intercept=a_intercept,
        b_slope=b_slope,
        b_intercept=b_intercept,
        width=np.array([IMAGE_SIZE[0]], dtype=np.int32),
        height=np.array([IMAGE_SIZE[1]], dtype=np.int32),
        training_count=np.array([training_count], dtype=np.int32),
    )
    print(f"Saved linear color model to {output_path}")
    print(f"Training images: {training_count}")


def colorize(input_path: Path, model_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    with np.load(model_path) as model:
        a_slope = model["a_slope"].astype(np.float32)
        a_intercept = model["a_intercept"].astype(np.float32)
        b_slope = model["b_slope"].astype(np.float32)
        b_intercept = model["b_intercept"].astype(np.float32)

    l_channel, _, _ = image_to_lab_channels(input_path)
    predicted_a = np.clip(a_slope * l_channel + a_intercept, 0, 255).astype(np.uint8)
    predicted_b = np.clip(b_slope * l_channel + b_intercept, 0, 255).astype(np.uint8)
    l_uint8 = np.clip(l_channel, 0, 255).astype(np.uint8)

    lab_image = Image.merge(
        "LAB",
        (
            Image.fromarray(l_uint8.reshape(IMAGE_SIZE), mode="L"),
            Image.fromarray(predicted_a.reshape(IMAGE_SIZE), mode="L"),
            Image.fromarray(predicted_b.reshape(IMAGE_SIZE), mode="L"),
        ),
    )
    lab_image.convert("RGB").save(output_path, "PNG")
    print(f"Saved colorized image to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a pixel-wise linear BW-to-color model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Learn parameters from images.")
    train_parser.add_argument(
        "--transformed-data",
        default="transformed_data.npz",
        help="Existing transformed training data from transformtrainingdata.py.",
    )
    train_parser.add_argument(
        "--data",
        help="Folder of training images. Used only when --transformed-data is not set.",
    )
    train_parser.add_argument(
        "--output",
        default="linear_color_model.npz",
        help="Path for the learned parameter file.",
    )

    colorize_parser = subparsers.add_parser("colorize", help="Colorize one image.")
    colorize_parser.add_argument("--input", default="inputphoto2.jpg", help="Input image.")
    colorize_parser.add_argument(
        "--model",
        default="linear_color_model.npz",
        help="Learned parameter file.",
    )
    colorize_parser.add_argument(
        "--output",
        default="linear_output.png",
        help="Output PNG image.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            data_folder = Path(args.data) if args.data else None
            transformed_data_path = None
            if data_folder is None and args.transformed_data:
                transformed_data_path = Path(args.transformed_data)
            train(Path(args.output), transformed_data_path, data_folder)
        elif args.command == "colorize":
            colorize(Path(args.input), Path(args.model), Path(args.output))
        else:
            parser.error(f"Unknown command: {args.command}")
    except FileNotFoundError as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
