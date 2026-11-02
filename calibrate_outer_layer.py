import argparse
import json
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from linear_color_model import IMAGE_SIZE, iter_image_paths
from portrait_preprocess import detect_largest_face, load_image_for_cv


DEFAULT_OUTPUT = "outer_layer_calibration.json"
DEFAULT_BINS = 72
DEFAULT_MIN_FOREGROUND_RATIO = 0.08
DEFAULT_MAX_FOREGROUND_RATIO = 0.95
Y_RATIO_MIN = -0.45
Y_RATIO_MAX = 2.25


def training_foreground_mask(image_path: Path) -> Optional[np.ndarray]:
    image = Image.open(image_path).convert("RGBA").resize(IMAGE_SIZE)
    rgba = np.asarray(image, dtype=np.uint8)
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]

    if np.unique(alpha).size > 1:
        mask = alpha > 16
    else:
        mask = np.any(rgb < 245, axis=2)

    ratio = float(mask.mean())
    if ratio < DEFAULT_MIN_FOREGROUND_RATIO or ratio > DEFAULT_MAX_FOREGROUND_RATIO:
        return None

    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask_uint8 > 0


def collect_boundary_samples(
    image_path: Path,
) -> Optional[List[Tuple[float, float, float]]]:
    image = load_image_for_cv(image_path)
    face = detect_largest_face(image)
    foreground = training_foreground_mask(image_path)
    if face is None or foreground is None:
        return None

    x, y, width, height = face
    face_size = max(width, height)
    face_center_x = x + width / 2
    min_row_width = max(12, int(face_size * 0.18))
    samples: List[Tuple[float, float, float]] = []

    for row_y in range(foreground.shape[0]):
        xs = np.flatnonzero(foreground[row_y])
        if xs.size < min_row_width:
            continue

        left = float(np.quantile(xs, 0.02))
        right = float(np.quantile(xs, 0.98))
        if right - left < min_row_width:
            continue

        y_ratio = (row_y - y) / face_size
        if y_ratio < Y_RATIO_MIN or y_ratio > Y_RATIO_MAX:
            continue

        samples.append(
            (
                y_ratio,
                (left - face_center_x) / face_size,
                (right - face_center_x) / face_size,
            )
        )

    if len(samples) < 20:
        return None
    return samples


def smooth(values: np.ndarray) -> np.ndarray:
    kernel = np.array([1, 3, 6, 8, 6, 3, 1], dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(values, (3, 3), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fit_outer_layer(
    source_dir: Path,
    bins: int,
) -> Dict[str, object]:
    y_grid = np.linspace(Y_RATIO_MIN, Y_RATIO_MAX, bins, dtype=np.float32)
    half_step = float((Y_RATIO_MAX - Y_RATIO_MIN) / max(bins - 1, 1) / 2)
    left_by_bin: List[List[float]] = [[] for _ in range(bins)]
    right_by_bin: List[List[float]] = [[] for _ in range(bins)]
    image_count = 0
    usable_count = 0

    for image_path in iter_image_paths(source_dir):
        image_count += 1
        samples = collect_boundary_samples(image_path)
        if not samples:
            continue

        usable_count += 1
        for y_ratio, left_ratio, right_ratio in samples:
            bin_index = int(np.argmin(np.abs(y_grid - y_ratio)))
            if abs(float(y_grid[bin_index]) - y_ratio) <= half_step * 1.35:
                left_by_bin[bin_index].append(left_ratio)
                right_by_bin[bin_index].append(right_ratio)

        if usable_count % 1000 == 0:
            print(f"Collected outer-layer boundaries from {usable_count} images")

    raw_left = np.full(bins, np.nan, dtype=np.float32)
    raw_right = np.full(bins, np.nan, dtype=np.float32)
    counts = []
    for index in range(bins):
        counts.append(min(len(left_by_bin[index]), len(right_by_bin[index])))
        if left_by_bin[index] and right_by_bin[index]:
            raw_left[index] = float(median(left_by_bin[index]))
            raw_right[index] = float(median(right_by_bin[index]))

    known = np.flatnonzero(~np.isnan(raw_left) & ~np.isnan(raw_right))
    if known.size < 8:
        raise RuntimeError(f"Could not learn enough outer-layer rows from {source_dir}")

    all_indices = np.arange(bins)
    left = np.interp(all_indices, known, raw_left[known])
    right = np.interp(all_indices, known, raw_right[known])

    left = smooth(left.astype(np.float32))
    right = smooth(right.astype(np.float32))

    # Keep the learned envelope physically plausible after smoothing.
    min_width = 0.28
    widths = right - left
    too_narrow = widths < min_width
    if np.any(too_narrow):
        centers = (left + right) / 2
        left[too_narrow] = centers[too_narrow] - min_width / 2
        right[too_narrow] = centers[too_narrow] + min_width / 2

    top_ratio = -0.30
    top_ratios = []
    upper_lefts = []
    upper_rights = []
    shoulder_lefts = []
    shoulder_rights = []
    for image_path in iter_image_paths(source_dir):
        samples = collect_boundary_samples(image_path)
        if not samples:
            continue

        sample_array = np.asarray(samples, dtype=np.float32)
        top_ratios.append(float(sample_array[:, 0].min()))

        upper = sample_array[
            (sample_array[:, 0] >= -0.42) & (sample_array[:, 0] <= 1.0)
        ]
        if upper.size:
            upper_lefts.append(float(upper[:, 1].min()))
            upper_rights.append(float(upper[:, 2].max()))

        shoulder = sample_array[
            (sample_array[:, 0] >= 1.02) & (sample_array[:, 0] <= 1.34)
        ]
        if shoulder.size:
            shoulder_lefts.append(float(np.quantile(shoulder[:, 1], 0.10)))
            shoulder_rights.append(float(np.quantile(shoulder[:, 2], 0.90)))

    if top_ratios:
        top_ratio = float(np.quantile(top_ratios, 0.35))
    else:
        top_ratio = -0.30

    upper_half_width = max(
        abs(float(median(upper_lefts))) if upper_lefts else 0.52,
        abs(float(median(upper_rights))) if upper_rights else 0.52,
    )
    shoulder_half_width = max(
        abs(float(median(shoulder_lefts))) if shoulder_lefts else 0.76,
        abs(float(median(shoulder_rights))) if shoulder_rights else 0.76,
    )

    head_bottom_ratio = 1.12
    head_axis_y_ratio = (head_bottom_ratio - top_ratio) / 2
    head_center_y_ratio = top_ratio + head_axis_y_ratio
    head_axis_x_ratio = float(np.clip(upper_half_width * 1.08, 0.52, 0.68))
    shoulder_half_width = float(np.clip(shoulder_half_width, 0.66, 0.96))
    torso_bottom_half_width = float(np.clip(shoulder_half_width * 1.12, 0.78, 1.08))
    parameters = {
        "head_center_y_ratio": round(head_center_y_ratio, 5),
        "head_axis_x_ratio": round(head_axis_x_ratio, 5),
        "head_axis_y_ratio": round(head_axis_y_ratio, 5),
        "neck_top_ratio": 0.82,
        "neck_bottom_ratio": 1.16,
        "neck_half_width_ratio": 0.22,
        "shoulder_start_y_ratio": 1.02,
        "shoulder_end_y_ratio": 1.25,
        "shoulder_half_width_ratio": round(shoulder_half_width, 5),
        "torso_bottom_half_width_ratio": round(torso_bottom_half_width, 5),
    }

    return {
        "dataset_images": image_count,
        "usable_images": usable_count,
        "shape_type": "oval_shoulders",
        "parameters": parameters,
        "y_ratio_min": Y_RATIO_MIN,
        "y_ratio_max": Y_RATIO_MAX,
        "face_y_ratios": y_grid.round(5).tolist(),
        "left_x_ratios": left.round(5).tolist(),
        "right_x_ratios": right.round(5).tolist(),
        "samples_per_row": counts,
        "description": (
            "Median training-set foreground envelope normalized by detected face box. "
            "Y is relative to face top and X is relative to face center, both scaled "
            "by detected face size."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Learn a smooth outer-layer portrait mask from training data."
    )
    parser.add_argument(
        "--source-dir",
        default="data",
        help="Folder of aligned training portraits.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path for the learned outer-layer calibration JSON.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of row samples in the learned boundary model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bins < 12:
        raise SystemExit("--bins must be at least 12")

    calibration = fit_outer_layer(Path(args.source_dir), args.bins)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    print(f"Saved outer-layer calibration to {output_path}")
    print(json.dumps({k: calibration[k] for k in ("dataset_images", "usable_images")}, indent=2))


if __name__ == "__main__":
    main()
