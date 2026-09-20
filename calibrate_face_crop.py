import argparse
import json
from pathlib import Path
from statistics import median
from typing import Dict, List

import cv2
import numpy as np

from download_training_data import DEFAULT_DATASET, download_dataset, image_paths
from portrait_preprocess import detect_largest_face, load_image_for_cv


DEFAULT_OUTPUT = "face_crop_calibration.json"


def robust_quantile(values: List[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def collect_face_stats(source_dir: Path, no_op_regularization: float) -> Dict[str, float]:
    face_width_ratios: List[float] = []
    face_center_x_ratios: List[float] = []
    face_center_y_ratios: List[float] = []
    image_count = 0
    detected_count = 0

    for image_path in image_paths(source_dir):
        image_count += 1
        image = load_image_for_cv(image_path)
        face = detect_largest_face(image)
        if face is None:
            continue

        x, y, width, height = face
        detected_count += 1
        image_height, image_width = image.shape[:2]
        face_size = max(width, height)

        face_width_ratios.append(face_size / min(image_width, image_height))
        face_center_x_ratios.append((x + width / 2) / image_width)
        face_center_y_ratios.append((y + height / 2) / image_height)

        if detected_count % 1000 == 0:
            print(f"Detected faces in {detected_count} training images")

    if not face_width_ratios:
        raise RuntimeError(f"No faces detected in {source_dir}")

    median_width = float(median(face_width_ratios))
    no_op_width = robust_quantile(face_width_ratios, 0.05)
    target_width = (
        (1 - no_op_regularization) * median_width
        + no_op_regularization * no_op_width
    )

    # The primary objective is to match the training-set face geometry. The no-op
    # term is only a regularizer, nudging the crop slightly wider so already
    # aligned training portraits are not over-cropped by the detector.
    return {
        "dataset_images": image_count,
        "detected_faces": detected_count,
        "no_op_regularization": no_op_regularization,
        "target_face_width_ratio": target_width,
        "target_face_center_x": float(median(face_center_x_ratios)),
        "target_face_center_y": float(median(face_center_y_ratios)),
        "median_face_width_ratio": float(median(face_width_ratios)),
        "median_face_center_x": float(median(face_center_x_ratios)),
        "median_face_center_y": float(median(face_center_y_ratios)),
        "no_op_face_width_ratio": no_op_width,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Learn face-crop calibration from the training dataset."
    )
    parser.add_argument(
        "--source-dir",
        help="Local extracted training dataset folder. Defaults to KaggleHub cache.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path for the learned face-crop calibration JSON.",
    )
    parser.add_argument(
        "--no-op-regularization",
        type=float,
        default=0.25,
        help=(
            "Blend weight from median training geometry toward a wider no-op crop. "
            "0 means pure median training geometry; 1 means strong no-op pressure."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_dir = Path(args.source_dir) if args.source_dir else download_dataset(DEFAULT_DATASET)
    calibration = collect_face_stats(source_dir, args.no_op_regularization)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    print(f"Saved face-crop calibration to {output_path}")
    print(json.dumps(calibration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
