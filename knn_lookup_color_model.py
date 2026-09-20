import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from linear_color_model import IMAGE_SIZE, add_white_background, image_to_lab_channels, iter_image_paths
from portrait_preprocess import (
    detect_largest_face,
    foreground_mask_from_face,
    load_image_for_cv,
)


DEFAULT_K = 25
WEAK_CHROMA_THRESHOLD = 12
FOREGROUND_FALLBACK_BLEND = 0.85


def image_to_lab_channels_and_foreground(
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img = Image.open(image_path)
    rgba = img.convert("RGBA").resize(IMAGE_SIZE)
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8).reshape(-1)
    rgb = add_white_background(rgba)
    lab = rgb.convert("LAB")
    l_channel, a_channel, b_channel = lab.split()

    foreground = alpha > 16
    if foreground.mean() > 0.98:
        rgb_values = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
        foreground = np.any(rgb_values < 245, axis=1)

    return (
        np.asarray(l_channel, dtype=np.float32).reshape(-1),
        np.asarray(a_channel, dtype=np.float32).reshape(-1),
        np.asarray(b_channel, dtype=np.float32).reshape(-1),
        foreground,
    )


def build_binned_statistics(
    data_folder: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    num_pixels = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    counts = np.zeros((num_pixels, 256), dtype=np.uint16)
    a_sums = np.zeros((num_pixels, 256), dtype=np.uint32)
    b_sums = np.zeros((num_pixels, 256), dtype=np.uint32)
    foreground_counts = np.zeros(256, dtype=np.uint32)
    foreground_a_sums = np.zeros(256, dtype=np.uint64)
    foreground_b_sums = np.zeros(256, dtype=np.uint64)

    image_count = 0
    pixel_indices = np.arange(num_pixels)

    for image_path in iter_image_paths(data_folder):
        l_channel, a_channel, b_channel, foreground = image_to_lab_channels_and_foreground(image_path)
        l_bins = np.clip(l_channel, 0, 255).astype(np.uint8)
        a_values = np.clip(a_channel, 0, 255).astype(np.uint8)
        b_values = np.clip(b_channel, 0, 255).astype(np.uint8)

        np.add.at(counts, (pixel_indices, l_bins), 1)
        np.add.at(a_sums, (pixel_indices, l_bins), a_values)
        np.add.at(b_sums, (pixel_indices, l_bins), b_values)
        np.add.at(foreground_counts, l_bins[foreground], 1)
        np.add.at(foreground_a_sums, l_bins[foreground], a_values[foreground])
        np.add.at(foreground_b_sums, l_bins[foreground], b_values[foreground])

        image_count += 1
        if image_count % 1000 == 0:
            print(f"Processed {image_count} training images")

    return (
        counts,
        a_sums,
        b_sums,
        foreground_counts,
        foreground_a_sums,
        foreground_b_sums,
        image_count,
    )


def build_lookup_tables(
    counts: np.ndarray,
    a_sums: np.ndarray,
    b_sums: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_pixels = counts.shape[0]
    lookup_a = np.empty((num_pixels, 256), dtype=np.uint8)
    lookup_b = np.empty((num_pixels, 256), dtype=np.uint8)

    for query_bin in range(256):
        total_count = counts[:, query_bin].astype(np.uint32)
        total_a = a_sums[:, query_bin].astype(np.uint32)
        total_b = b_sums[:, query_bin].astype(np.uint32)

        for radius in range(1, 256):
            active = total_count < k
            if not np.any(active):
                break

            left = query_bin - radius
            right = query_bin + radius

            if left >= 0:
                total_count[active] += counts[active, left]
                total_a[active] += a_sums[active, left]
                total_b[active] += b_sums[active, left]
            if right <= 255:
                total_count[active] += counts[active, right]
                total_a[active] += a_sums[active, right]
                total_b[active] += b_sums[active, right]

            if left <= 0 and right >= 255:
                break

        total_count = np.maximum(total_count, 1)
        lookup_a[:, query_bin] = np.rint(total_a / total_count).clip(0, 255)
        lookup_b[:, query_bin] = np.rint(total_b / total_count).clip(0, 255)

        if (query_bin + 1) % 32 == 0:
            print(f"Built lookup bins through L={query_bin}")

    return lookup_a, lookup_b


def build_global_lookup(
    counts: np.ndarray,
    a_sums: np.ndarray,
    b_sums: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    lookup_a = np.empty(256, dtype=np.uint8)
    lookup_b = np.empty(256, dtype=np.uint8)

    for query_bin in range(256):
        total_count = int(counts[query_bin])
        total_a = int(a_sums[query_bin])
        total_b = int(b_sums[query_bin])

        for radius in range(1, 256):
            if total_count >= k:
                break

            left = query_bin - radius
            right = query_bin + radius
            if left >= 0:
                total_count += int(counts[left])
                total_a += int(a_sums[left])
                total_b += int(b_sums[left])
            if right <= 255:
                total_count += int(counts[right])
                total_a += int(a_sums[right])
                total_b += int(b_sums[right])
            if left <= 0 and right >= 255:
                break

        if total_count == 0:
            lookup_a[query_bin] = 128
            lookup_b[query_bin] = 128
        else:
            lookup_a[query_bin] = np.clip(round(total_a / total_count), 0, 255)
            lookup_b[query_bin] = np.clip(round(total_b / total_count), 0, 255)

    return lookup_a, lookup_b


def train_lookup_model(data_folder: Path, output_path: Path, k: int = DEFAULT_K) -> None:
    (
        counts,
        a_sums,
        b_sums,
        foreground_counts,
        foreground_a_sums,
        foreground_b_sums,
        training_count,
    ) = build_binned_statistics(data_folder)
    lookup_a, lookup_b = build_lookup_tables(counts, a_sums, b_sums, k)
    global_lookup_a, global_lookup_b = build_global_lookup(
        foreground_counts,
        foreground_a_sums,
        foreground_b_sums,
        k * 100,
    )

    np.savez_compressed(
        output_path,
        lookup_a=lookup_a,
        lookup_b=lookup_b,
        global_lookup_a=global_lookup_a,
        global_lookup_b=global_lookup_b,
        width=np.array([IMAGE_SIZE[0]], dtype=np.int32),
        height=np.array([IMAGE_SIZE[1]], dtype=np.int32),
        k=np.array([k], dtype=np.int32),
        training_count=np.array([training_count], dtype=np.int32),
    )
    print(f"Saved KNN lookup color model to {output_path}")
    print(f"Training images: {training_count}")
    print(f"K: {k}")


def foreground_mask_for_input(input_path: Path) -> np.ndarray:
    try:
        image = load_image_for_cv(input_path)
        face = detect_largest_face(image)
        if face is None:
            return np.zeros(IMAGE_SIZE[0] * IMAGE_SIZE[1], dtype=bool)
        mask = foreground_mask_from_face(image, face)
        mask = np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize(IMAGE_SIZE)
        )
        return mask.reshape(-1) > 96
    except Exception:
        return np.zeros(IMAGE_SIZE[0] * IMAGE_SIZE[1], dtype=bool)


def apply_foreground_fallback(
    l_uint8: np.ndarray,
    predicted_a: np.ndarray,
    predicted_b: np.ndarray,
    foreground: np.ndarray,
    global_lookup_a: np.ndarray,
    global_lookup_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    chroma_distance = np.sqrt(
        np.square(predicted_a.astype(np.float32) - 128)
        + np.square(predicted_b.astype(np.float32) - 128)
    )
    weak_foreground = foreground & (chroma_distance < WEAK_CHROMA_THRESHOLD)
    if not np.any(weak_foreground):
        return predicted_a, predicted_b

    fallback_a = global_lookup_a[l_uint8].astype(np.float32)
    fallback_b = global_lookup_b[l_uint8].astype(np.float32)
    blended_a = (
        predicted_a.astype(np.float32) * (1 - FOREGROUND_FALLBACK_BLEND)
        + fallback_a * FOREGROUND_FALLBACK_BLEND
    )
    blended_b = (
        predicted_b.astype(np.float32) * (1 - FOREGROUND_FALLBACK_BLEND)
        + fallback_b * FOREGROUND_FALLBACK_BLEND
    )
    predicted_a = predicted_a.copy()
    predicted_b = predicted_b.copy()
    predicted_a[weak_foreground] = np.clip(np.rint(blended_a[weak_foreground]), 0, 255)
    predicted_b[weak_foreground] = np.clip(np.rint(blended_b[weak_foreground]), 0, 255)
    return predicted_a, predicted_b


def colorize(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    use_hybrid: bool = True,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    with np.load(model_path) as model:
        lookup_a = model["lookup_a"]
        lookup_b = model["lookup_b"]
        global_lookup_a = model["global_lookup_a"] if "global_lookup_a" in model else None
        global_lookup_b = model["global_lookup_b"] if "global_lookup_b" in model else None

    l_channel, _, _ = image_to_lab_channels(input_path)
    l_uint8 = np.clip(l_channel, 0, 255).astype(np.uint8)
    pixel_indices = np.arange(l_uint8.size)
    predicted_a = lookup_a[pixel_indices, l_uint8].astype(np.uint8)
    predicted_b = lookup_b[pixel_indices, l_uint8].astype(np.uint8)
    if use_hybrid and global_lookup_a is not None and global_lookup_b is not None:
        foreground = foreground_mask_for_input(input_path)
        predicted_a, predicted_b = apply_foreground_fallback(
            l_uint8,
            predicted_a,
            predicted_b,
            foreground,
            global_lookup_a,
            global_lookup_b,
        )

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
        description="Train and run a pixel-wise KNN lookup BW-to-color model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Build lookup from images.")
    train_parser.add_argument("--data", required=True, help="Folder of training images.")
    train_parser.add_argument(
        "--output",
        default="knn_lookup_color_model.npz",
        help="Path for the learned lookup file.",
    )
    train_parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of nearest brightness samples to average.",
    )

    colorize_parser = subparsers.add_parser("colorize", help="Colorize one image.")
    colorize_parser.add_argument("--input", required=True, help="Input image.")
    colorize_parser.add_argument(
        "--model",
        default="knn_lookup_color_model.npz",
        help="Learned lookup file.",
    )
    colorize_parser.add_argument(
        "--output",
        default="knn_lookup_output.png",
        help="Output PNG image.",
    )
    colorize_parser.add_argument(
        "--plain",
        action="store_true",
        help="Use the pixel lookup only, without foreground fallback.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        if args.k < 1:
            parser.error("--k must be at least 1")
        train_lookup_model(Path(args.data), Path(args.output), args.k)
    elif args.command == "colorize":
        colorize(
            Path(args.input),
            Path(args.model),
            Path(args.output),
            use_hybrid=not args.plain,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
