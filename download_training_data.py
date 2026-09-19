import argparse
import shutil
from pathlib import Path
from typing import List, Optional

import kagglehub
from PIL import Image


DEFAULT_DATASET = "saiharim/fifa-player-faces"
DEFAULT_LIMIT = 200
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def download_dataset(dataset: str) -> Path:
    path = kagglehub.dataset_download(dataset)
    print("Path to dataset files:", path)
    return Path(path)


def image_paths(root: Path) -> List[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_valid_images(
    source_dir: Path, output_dir: Path, limit: Optional[int] = None
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for source_path in image_paths(source_dir):
        if limit is not None and copied >= limit:
            break

        try:
            with Image.open(source_path) as img:
                img.verify()
        except Exception:
            continue

        copied += 1
        output_path = output_dir / f"football_face_{copied:05d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, output_path)

    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a Kaggle face dataset and populate the data/ folder."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"KaggleHub dataset slug. Defaults to {DEFAULT_DATASET}.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Flat folder of training images consumed by transformtrainingdata.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "Maximum number of images to copy into the training folder. "
            "Use 0 to copy every image."
        ),
    )
    parser.add_argument(
        "--source-dir",
        help="Reuse a local extracted dataset folder instead of downloading.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    source_dir = Path(args.source_dir) if args.source_dir else download_dataset(args.dataset)
    limit = None if args.limit == 0 else args.limit

    copied = copy_valid_images(source_dir, output_dir, limit)
    if copied == 0:
        raise SystemExit(f"No valid images found in {source_dir}.")

    print(f"Copied {copied} training images into {output_dir}")
    print("Next steps:")
    print("  python3 transformtrainingdata.py")
    print(
        "  python3 linear_color_model.py train "
        "--transformed-data transformed_data.npz --output linear_color_model.npz"
    )


if __name__ == "__main__":
    main()
