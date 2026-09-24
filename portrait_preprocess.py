from pathlib import Path
from typing import Optional, Tuple
import io
import json
import urllib.request

import cv2
import numpy as np
from PIL import Image


IMAGE_SIZE = (240, 240)
CALIBRATION_PATH = Path(__file__).resolve().parent / "face_crop_calibration.json"
MEDIAPIPE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "selfie_segmenter.tflite"
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)
TARGET_FACE_WIDTH_RATIO = 0.5417
TARGET_FACE_CENTER_X = 0.4938
TARGET_FACE_CENTER_Y = 0.5333
MIN_TOP_MARGIN_FACE_RATIO = 0.65
EDGE_REFINEMENT_STEPS = 7
REMBG_SESSION = None
MEDIAPIPE_IMAGE_SEGMENTER = None


def load_crop_calibration() -> Tuple[float, float, float]:
    if not CALIBRATION_PATH.exists():
        return TARGET_FACE_WIDTH_RATIO, TARGET_FACE_CENTER_X, TARGET_FACE_CENTER_Y

    data = json.loads(CALIBRATION_PATH.read_text())
    return (
        float(data.get("target_face_width_ratio", TARGET_FACE_WIDTH_RATIO)),
        float(data.get("target_face_center_x", TARGET_FACE_CENTER_X)),
        float(data.get("target_face_center_y", TARGET_FACE_CENTER_Y)),
    )


def load_image_for_cv(input_path: Path) -> np.ndarray:
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Unable to read this image.")

    if len(image.shape) == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3] / 255.0
        bgr = image[:, :, :3]
        white = np.full_like(bgr, 255)
        image = (bgr * alpha[..., None] + white * (1 - alpha[..., None])).astype(
            np.uint8
        )

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def detect_largest_face(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None

    return tuple(max(faces, key=lambda face: face[2] * face[3]))


def crop_face_portrait(input_path: Path, output_path: Path) -> Path:
    image = load_image_for_cv(input_path)
    face = detect_largest_face(image)
    if face is None:
        raise ValueError("No face detected. Please upload a clear front-facing portrait.")

    x, y, width, height = face
    width_ratio, center_x_ratio, center_y_ratio = load_crop_calibration()
    crop = square_crop_bounds(
        image_width=image.shape[1],
        image_height=image.shape[0],
        center_x=x + width // 2,
        center_y=y + height // 2,
        face_size=max(width, height),
        target_face_width_ratio=width_ratio,
        target_face_center_x=center_x_ratio,
        target_face_center_y=center_y_ratio,
    )
    crop = refine_crop_with_edges(
        image=image,
        face=face,
        initial_crop=crop,
        target_face_width_ratio=width_ratio,
        target_face_center_x=center_x_ratio,
        target_face_center_y=center_y_ratio,
    )
    left, top, right, bottom = crop
    portrait = crop_with_padding(image, left, top, right, bottom)
    portrait = cv2.resize(portrait, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), portrait)
    return output_path


def edge_detection_sliding_window(image: np.ndarray) -> np.ndarray:
    """
    Applies a 3x3 Sobel horizontal edge detection kernel to a grayscale image
    using a custom sliding window loop.
    """
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    img_height, img_width = image.shape
    output_image = np.zeros((img_height, img_width))

    for y in range(1, img_height - 1):
        for x in range(1, img_width - 1):
            window = image[y - 1 : y + 2, x - 1 : x + 2]
            pixel_gradient = np.sum(window * kernel)
            output_image[y, x] = min(abs(pixel_gradient), 255)

    return output_image.astype(np.uint8)


def sobel_gradient_edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(image, (5, 5), 0)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def save_edge_detection_preview(input_path: Path, output_path: Path) -> Path:
    image = load_image_for_cv(input_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = edge_detection_sliding_window(grayscale)
    cv2.imwrite(str(output_path), edge_image)
    return output_path


def save_subject_mask_preview(
    input_path: Path,
    output_path: Path,
    method: str = "rembg",
) -> Path:
    image = load_image_for_cv(input_path)
    mask = subject_mask_for_image(image, method=method)
    preview = np.clip(mask * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), preview)
    return output_path


def save_edge_overlay_preview(input_path: Path, output_path: Path) -> Path:
    image = load_image_for_cv(input_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = edge_detection_sliding_window(grayscale)
    _, strong_edges = cv2.threshold(edge_image, 80, 255, cv2.THRESH_BINARY)

    overlay = image.copy()
    overlay[strong_edges > 0] = (60, 60, 255)
    blended = cv2.addWeighted(image, 0.72, overlay, 0.28, 0)
    cv2.imwrite(str(output_path), blended)
    return output_path


def boundary_from_strong_edges(
    edge_image: np.ndarray,
    prior_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    edge_image = edge_image.copy()
    border_margin = 8
    edge_image[:border_margin, :] = 0
    edge_image[-border_margin:, :] = 0
    edge_image[:, :border_margin] = 0
    edge_image[:, -border_margin:] = 0

    if prior_mask is not None:
        edge_image = edge_image * (prior_mask > 0)

    _, boundary = cv2.threshold(edge_image, 80, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.morphologyEx(boundary, cv2.MORPH_OPEN, kernel, iterations=1)
    return boundary


def save_boundary_preview(input_path: Path, output_path: Path) -> Path:
    image = load_image_for_cv(input_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = edge_detection_sliding_window(grayscale)
    face = detect_largest_face(image)
    prior_mask = None
    if face is not None:
        prior_mask = geometric_foreground_mask(image, face)
        prior_mask = cv2.dilate(
            prior_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
    boundary = boundary_from_strong_edges(edge_image, prior_mask)
    cv2.imwrite(str(output_path), boundary)
    return output_path


def save_connected_boundary_preview(input_path: Path, output_path: Path) -> Path:
    image = load_image_for_cv(input_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = edge_detection_sliding_window(grayscale)
    face = detect_largest_face(image)
    prior_mask = None
    if face is not None:
        prior_mask = geometric_foreground_mask(image, face)
        prior_mask = cv2.dilate(
            prior_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )

    strong_boundary = boundary_from_strong_edges(edge_image, prior_mask)
    connected_boundary = outermost_layer_from_edges(strong_boundary, prior_mask)
    cv2.imwrite(str(output_path), connected_boundary)
    return output_path


def outermost_layer_from_edges(
    edge_image: np.ndarray,
    prior_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    image_height, image_width = edge_image.shape
    output = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    left_points, right_points, binary_edges, allowed_mask = outer_layer_boundary_points(
        edge_image,
        prior_mask,
    )
    draw_boundary_polyline(output, left_points)
    draw_boundary_polyline(output, right_points)
    draw_top_boundary_bridge(output, left_points, right_points, binary_edges, allowed_mask)
    return output


def outer_layer_boundary_points(
    edge_image: np.ndarray,
    prior_mask: Optional[np.ndarray] = None,
) -> tuple[list[Tuple[int, int]], list[Tuple[int, int]], np.ndarray, np.ndarray]:
    edge_image = edge_image.copy()
    border_margin = 8
    edge_image[:border_margin, :] = 0
    edge_image[-border_margin:, :] = 0
    edge_image[:, :border_margin] = 0
    edge_image[:, -border_margin:] = 0

    allowed_mask = np.ones(edge_image.shape, dtype=bool)
    if prior_mask is not None:
        allowed_mask = prior_mask > 0
        edge_image = edge_image * allowed_mask

    _, binary_edges = cv2.threshold(edge_image, 80, 255, cv2.THRESH_BINARY)
    left_points = smooth_boundary_points(
        boundary_points_by_row(binary_edges, allowed_mask, "left")
    )
    right_points = smooth_boundary_points(
        boundary_points_by_row(binary_edges, allowed_mask, "right")
    )
    return left_points, right_points, binary_edges, allowed_mask


def outer_layer_mask_from_edges(
    edge_image: np.ndarray,
    prior_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    left_points, right_points, _, _ = outer_layer_boundary_points(edge_image, prior_mask)
    mask = np.zeros(edge_image.shape, dtype=np.uint8)
    if len(left_points) < 2 or len(right_points) < 2:
        return mask

    polygon = np.array(left_points + list(reversed(right_points)), dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def outer_layer_mask_for_image(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = edge_detection_sliding_window(grayscale)
    face = detect_largest_face(image)
    prior_mask = None
    if face is not None:
        prior_mask = geometric_foreground_mask(image, face)
        prior_mask = cv2.dilate(
            prior_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
    return outer_layer_mask_from_edges(edge_image, prior_mask)


def outer_layer_mask_for_path(input_path: Path) -> np.ndarray:
    image = load_image_for_cv(input_path)
    return outer_layer_mask_for_image(image)


def boundary_points_by_row(
    binary_edges: np.ndarray,
    allowed_mask: np.ndarray,
    side: str,
) -> list[Tuple[int, int]]:
    image_height, image_width = binary_edges.shape
    points = []
    border_margin = 8
    search_radius = 28
    min_stable_points = 8
    min_initial_span = image_width * 0.18
    last_x = None

    for y in range(border_margin, image_height - border_margin):
        row = (binary_edges[y] > 0) & allowed_mask[y]
        xs = np.flatnonzero(row)
        xs = xs[(xs >= border_margin) & (xs < image_width - border_margin)]
        if xs.size == 0:
            if points:
                points.append((last_x, y))
            continue

        if last_x is None:
            if xs[-1] - xs[0] < min_initial_span:
                continue
            x = int(xs[0] if side == "left" else xs[-1])
        else:
            nearby = xs[np.abs(xs - last_x) <= search_radius]
            if nearby.size == 0:
                if len(points) < min_stable_points:
                    points = []
                    last_x = None
                    continue
                x = last_x
            else:
                x = choose_connected_boundary_x(nearby, last_x, side)

        touches_crop_boundary = (
            x <= border_margin + 2 if side == "left" else x >= image_width - border_margin - 3
        )
        points.append((x, y))
        last_x = x

        if touches_crop_boundary and len(points) > min_stable_points:
            break

    return points


def choose_connected_boundary_x(xs: np.ndarray, last_x: int, side: str) -> int:
    distances = np.abs(xs - last_x)
    nearest_distance = distances.min()
    nearest = xs[distances == nearest_distance]
    if side == "left":
        return int(nearest.min())
    return int(nearest.max())


def smooth_boundary_points(points: list[Tuple[int, int]]) -> list[Tuple[int, int]]:
    if len(points) < 7:
        return points

    xs = np.array([point[0] for point in points], dtype=np.float32)
    ys = np.array([point[1] for point in points], dtype=np.int32)
    kernel = np.array([1, 4, 7, 10, 13, 10, 7, 4, 1], dtype=np.float32)
    kernel /= kernel.sum()
    padded_xs = np.pad(xs, (4, 4), mode="edge")
    smoothed_xs = np.convolve(padded_xs, kernel, mode="valid")
    return [(int(round(x)), int(y)) for x, y in zip(smoothed_xs, ys)]


def draw_boundary_polyline(
    output: np.ndarray,
    points: list[Tuple[int, int]],
) -> None:
    if len(points) < 2:
        return

    segment = [points[0]]
    for point in points[1:]:
        if point[1] - segment[-1][1] > 4:
            draw_segment(output, segment)
            segment = [point]
        else:
            segment.append(point)
    draw_segment(output, segment)


def draw_segment(output: np.ndarray, points: list[Tuple[int, int]]) -> None:
    if len(points) < 2:
        return

    cv2.polylines(
        output,
        [np.array(points, dtype=np.int32)],
        isClosed=False,
        color=(255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def draw_top_boundary_bridge(
    output: np.ndarray,
    left_points: list[Tuple[int, int]],
    right_points: list[Tuple[int, int]],
    binary_edges: np.ndarray,
    allowed_mask: np.ndarray,
) -> None:
    if not left_points or not right_points:
        return

    left_top = left_points[0]
    right_top = right_points[0]
    start_x = min(left_top[0], right_top[0])
    end_x = max(left_top[0], right_top[0])
    top_y = min(left_top[1], right_top[1])
    search_top = max(0, top_y - 14)
    search_bottom = min(binary_edges.shape[0], top_y + 24)
    bridge_points = []

    for x in range(start_x, end_x + 1):
        column = (binary_edges[search_top:search_bottom, x] > 0) & allowed_mask[
            search_top:search_bottom,
            x,
        ]
        ys = np.flatnonzero(column)
        if ys.size:
            bridge_points.append((x, int(search_top + ys[0])))

    if len(bridge_points) < 3:
        mid_x = (start_x + end_x) // 2
        arch_y = max(0, top_y - 5)
        bridge_points = [left_top, (mid_x, arch_y), right_top]

    bridge_points = smooth_top_bridge_points(left_top, right_top, bridge_points)
    cv2.polylines(
        output,
        [np.array(bridge_points, dtype=np.int32)],
        isClosed=False,
        color=(255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def smooth_top_bridge_points(
    left_top: Tuple[int, int],
    right_top: Tuple[int, int],
    points: list[Tuple[int, int]],
) -> list[Tuple[int, int]]:
    if len(points) < 5:
        return points

    start_x = min(left_top[0], right_top[0])
    end_x = max(left_top[0], right_top[0])
    edge_top_y = min(point[1] for point in points)
    endpoint_y = min(left_top[1], right_top[1])
    control_y = max(0, min(edge_top_y, endpoint_y) - 5)
    control_x = (start_x + end_x) / 2

    curve = []
    steps = max(12, end_x - start_x)
    for index in range(steps + 1):
        t = index / steps
        one_minus_t = 1 - t
        x = (
            one_minus_t * one_minus_t * left_top[0]
            + 2 * one_minus_t * t * control_x
            + t * t * right_top[0]
        )
        y = (
            one_minus_t * one_minus_t * left_top[1]
            + 2 * one_minus_t * t * control_y
            + t * t * right_top[1]
        )
        curve.append((int(round(x)), int(round(y))))
    return curve


def save_outermost_layer_preview(input_path: Path, output_path: Path) -> Path:
    image = load_image_for_cv(input_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = edge_detection_sliding_window(grayscale)
    face = detect_largest_face(image)
    prior_mask = None
    if face is not None:
        prior_mask = geometric_foreground_mask(image, face)
        prior_mask = cv2.dilate(
            prior_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
    outer_layer = outermost_layer_from_edges(edge_image, prior_mask)
    cv2.imwrite(str(output_path), outer_layer)
    return output_path


def square_crop_bounds(
    image_width: int,
    image_height: int,
    center_x: int,
    center_y: int,
    face_size: int,
    target_face_width_ratio: float = TARGET_FACE_WIDTH_RATIO,
    target_face_center_x: float = TARGET_FACE_CENTER_X,
    target_face_center_y: float = TARGET_FACE_CENTER_Y,
) -> Tuple[int, int, int, int]:
    half_face = face_size / 2
    face_left = center_x - half_face
    face_top = center_y - half_face
    face_right = center_x + half_face
    face_bottom = center_y + half_face

    left_margin = face_size * (target_face_center_x / target_face_width_ratio - 0.5)
    right_margin = face_size * (
        (1 - target_face_center_x) / target_face_width_ratio - 0.5
    )
    top_margin = face_size * (target_face_center_y / target_face_width_ratio - 0.5)
    bottom_margin = face_size * (
        (1 - target_face_center_y) / target_face_width_ratio - 0.5
    )
    top_margin = max(top_margin, face_size * MIN_TOP_MARGIN_FACE_RATIO)

    left = face_left - left_margin
    top = face_top - top_margin
    right = face_right + right_margin
    bottom = face_bottom + bottom_margin

    width = right - left
    height = bottom - top
    square_size = max(width, height)

    if width < square_size:
        extra = square_size - width
        left -= extra * target_face_center_x
        right += extra * (1 - target_face_center_x)

    if height < square_size:
        extra = square_size - height
        top -= extra * target_face_center_y
        bottom += extra * (1 - target_face_center_y)

    left_i = round(left)
    top_i = round(top)
    square_i = round(square_size)
    return left_i, top_i, left_i + square_i, top_i + square_i


def refine_crop_with_edges(
    image: np.ndarray,
    face: Tuple[int, int, int, int],
    initial_crop: Tuple[int, int, int, int],
    target_face_width_ratio: float,
    target_face_center_x: float,
    target_face_center_y: float,
) -> Tuple[int, int, int, int]:
    x, y, width, height = face
    face_size = max(width, height)
    face_center_x = x + width / 2
    face_center_y = y + height / 2
    initial_left, initial_top, initial_right, initial_bottom = initial_crop
    initial_size = initial_right - initial_left

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = (cv2.Canny(gray, 60, 150) > 0).astype(np.uint8)
    integral = cv2.integral(edges, sdepth=cv2.CV_32S)

    best_crop = initial_crop
    best_score = crop_score(
        integral,
        image.shape[1],
        image.shape[0],
        initial_left,
        initial_top,
        initial_size,
        face_center_x,
        face_center_y,
        face_size,
        target_face_width_ratio,
        target_face_center_x,
        target_face_center_y,
    )

    size_candidates = np.linspace(initial_size * 0.92, initial_size * 1.18, EDGE_REFINEMENT_STEPS)
    for size in size_candidates:
        size = max(face_size / 0.7, size)
        expected_left = face_center_x - size * target_face_center_x
        expected_top = face_center_y - size * target_face_center_y
        x_offsets = np.linspace(-face_size * 0.18, face_size * 0.18, EDGE_REFINEMENT_STEPS)
        y_offsets = np.linspace(-face_size * 0.42, face_size * 0.18, EDGE_REFINEMENT_STEPS)

        for x_offset in x_offsets:
            for y_offset in y_offsets:
                left = round(expected_left + x_offset)
                top = round(expected_top + y_offset)
                square_size = round(size)
                score = crop_score(
                    integral,
                    image.shape[1],
                    image.shape[0],
                    left,
                    top,
                    square_size,
                    face_center_x,
                    face_center_y,
                    face_size,
                    target_face_width_ratio,
                    target_face_center_x,
                    target_face_center_y,
                )
                if score < best_score:
                    best_score = score
                    best_crop = (left, top, left + square_size, top + square_size)

    return best_crop


def crop_score(
    integral: np.ndarray,
    image_width: int,
    image_height: int,
    left: int,
    top: int,
    size: int,
    face_center_x: float,
    face_center_y: float,
    face_size: int,
    target_face_width_ratio: float,
    target_face_center_x: float,
    target_face_center_y: float,
) -> float:
    if size <= 0:
        return float("inf")

    right = left + size
    bottom = top + size
    face_width_ratio = face_size / size
    face_center_x_ratio = (face_center_x - left) / size
    face_center_y_ratio = (face_center_y - top) / size

    geometry_penalty = (
        abs(face_width_ratio - target_face_width_ratio) * 3.0
        + abs(face_center_x_ratio - target_face_center_x) * 2.0
        + abs(face_center_y_ratio - target_face_center_y) * 2.0
    )

    top_margin_ratio = (face_center_y - face_size / 2 - top) / face_size
    headroom_penalty = max(0.0, MIN_TOP_MARGIN_FACE_RATIO - top_margin_ratio) * 2.5

    strip = max(3, round(size * 0.035))
    top_edges = edge_density(integral, left, top, right, top + strip)
    left_edges = edge_density(integral, left, top, left + strip, bottom)
    right_edges = edge_density(integral, right - strip, top, right, bottom)
    bottom_edges = edge_density(integral, left, bottom - strip, right, bottom)
    border_penalty = top_edges * 4.5 + (left_edges + right_edges) * 1.2 + bottom_edges * 0.35

    clipped_width = max(0, min(right, image_width) - max(left, 0))
    clipped_height = max(0, min(bottom, image_height) - max(top, 0))
    visible_area = clipped_width * clipped_height
    padding_ratio = 1 - (visible_area / (size * size))
    padding_penalty = padding_ratio * 0.55

    return geometry_penalty + headroom_penalty + border_penalty + padding_penalty


def edge_density(
    integral: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> float:
    image_height = integral.shape[0] - 1
    image_width = integral.shape[1] - 1
    clipped_left = min(max(left, 0), image_width)
    clipped_top = min(max(top, 0), image_height)
    clipped_right = min(max(right, 0), image_width)
    clipped_bottom = min(max(bottom, 0), image_height)

    if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
        return 0.0

    edge_count = (
        integral[clipped_bottom, clipped_right]
        - integral[clipped_top, clipped_right]
        - integral[clipped_bottom, clipped_left]
        + integral[clipped_top, clipped_left]
    )
    area = (clipped_right - clipped_left) * (clipped_bottom - clipped_top)
    return float(edge_count) / max(area, 1)


def crop_with_padding(
    image: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> np.ndarray:
    crop_width = right - left
    crop_height = bottom - top
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("Unable to create a valid face crop.")

    output = np.full((crop_height, crop_width, 3), 255, dtype=np.uint8)
    source_left = max(left, 0)
    source_top = max(top, 0)
    source_right = min(right, image.shape[1])
    source_bottom = min(bottom, image.shape[0])

    if source_left >= source_right or source_top >= source_bottom:
        raise ValueError("Detected face crop is outside the image.")

    dest_left = source_left - left
    dest_top = source_top - top
    output[
        dest_top : dest_top + (source_bottom - source_top),
        dest_left : dest_left + (source_right - source_left),
    ] = image[source_top:source_bottom, source_left:source_right]

    return output


def replace_background_with_white(
    image: np.ndarray,
    face_box: Tuple[int, int, int, int],
) -> np.ndarray:
    mask = subject_mask_for_image(image, face_box)
    mask = compositing_mask_for_subject(mask, face_box)
    return apply_white_background(image, mask)


def compositing_mask_for_subject(
    subject_mask: np.ndarray,
    face_box: Tuple[int, int, int, int],
) -> np.ndarray:
    alpha = np.clip((subject_mask.astype(np.float32) - 0.28) / 0.36, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    x, y, width, height = face_box
    image_height, image_width = alpha.shape[:2]
    face_left = max(0, int(x - width * 0.08))
    face_top = max(0, int(y - height * 0.08))
    face_right = min(image_width, int(x + width * 1.08))
    face_bottom = min(image_height, int(y + height * 1.08))
    alpha[face_top:face_bottom, face_left:face_right] = 1

    confident_subject = (subject_mask > 0.78).astype(np.uint8)
    confident_subject = cv2.dilate(
        confident_subject,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    alpha[confident_subject > 0] = 1
    return np.clip(alpha, 0, 1)


def subject_mask_for_image(
    image: np.ndarray,
    face_box: Optional[Tuple[int, int, int, int]] = None,
    method: str = "rembg",
) -> np.ndarray:
    if method == "geometry":
        mask = geometry_guided_subject_mask(image, face_box)
    elif method == "mediapipe":
        mask = mediapipe_subject_mask(image)
    elif method == "rembg":
        mask = rembg_subject_mask(image)
    elif method in {"sobel_rembg", "sobel_rmeb", "rmeb_sobel"}:
        mask = sobel_rembg_subject_mask(image, face_box)
    else:
        raise ValueError(f"Unknown subject mask method: {method}")

    if mask is None and face_box is not None:
        mask = foreground_mask_from_face(image, face_box)
    if mask is None:
        detected_face = detect_largest_face(image)
        if detected_face is not None:
            mask = foreground_mask_from_face(image, detected_face)
    if mask is None:
        return np.ones(image.shape[:2], dtype=np.uint8)

    foreground_ratio = float(mask.mean())
    if foreground_ratio < 0.05 or foreground_ratio > 0.98:
        if face_box is not None:
            return foreground_mask_from_face(image, face_box)
    return clean_subject_mask(mask)


def sobel_rembg_subject_mask(
    image: np.ndarray,
    face_box: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    rembg_mask = rembg_subject_mask(image)
    if rembg_mask is None:
        if face_box is None:
            face_box = detect_largest_face(image)
        if face_box is None:
            return None
        rembg_mask = foreground_mask_from_face(image, face_box).astype(np.float32)

    refined = refine_mask_outer_boundary_with_sobel(image, rembg_mask, face_box)
    if refined is None:
        return rembg_mask

    return np.maximum(refined, rembg_mask * 0.82)


def refine_mask_outer_boundary_with_sobel(
    image: np.ndarray,
    base_mask: np.ndarray,
    face_box: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    if face_box is None:
        face_box = detect_largest_face(image)

    image_height, image_width = base_mask.shape[:2]
    binary = (base_mask > 0.38).astype(np.uint8)
    if int(binary.sum()) < image_width * image_height * 0.04:
        return None

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_edges = sobel_gradient_edges(grayscale)
    edge_threshold = max(35, int(np.percentile(sobel_edges[binary > 0], 62)))
    strong_edges = sobel_edges >= edge_threshold

    search_radius = max(5, int(image_width * 0.045))
    min_width = max(18, int(image_width * 0.12))
    row_points = []

    for y in range(2, image_height - 2):
        xs = np.flatnonzero(binary[y] > 0)
        if xs.size < min_width:
            continue

        left = int(xs[0])
        right = int(xs[-1])
        left = snap_boundary_to_sobel(
            strong_edges,
            sobel_edges,
            y,
            left,
            -search_radius,
            search_radius,
        )
        right = snap_boundary_to_sobel(
            strong_edges,
            sobel_edges,
            y,
            right,
            -search_radius,
            search_radius,
        )
        if right - left >= min_width:
            row_points.append((y, left, right))

    if len(row_points) < image_height * 0.18:
        return None

    row_points = smooth_boundary_rows(row_points)
    left_points = [(left, y) for y, left, _ in row_points]
    right_points = [(right, y) for y, _, right in row_points]

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    polygon = np.array(left_points + list(reversed(right_points)), dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)

    if face_box is not None:
        x, y, width, height = face_box
        face_left = max(0, int(x - width * 0.06))
        face_top = max(0, int(y - height * 0.08))
        face_right = min(image_width, int(x + width * 1.06))
        face_bottom = min(image_height, int(y + height * 1.10))
        mask[face_top:face_bottom, face_left:face_right] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 0)
    return np.clip(mask, 0, 1)


def snap_boundary_to_sobel(
    strong_edges: np.ndarray,
    sobel_edges: np.ndarray,
    y: int,
    x: int,
    left_offset: int,
    right_offset: int,
) -> int:
    image_width = strong_edges.shape[1]
    start = max(1, x + left_offset)
    end = min(image_width - 1, x + right_offset + 1)
    if start >= end:
        return x

    edge_slice = strong_edges[max(0, y - 1) : y + 2, start:end]
    scores = sobel_edges[max(0, y - 1) : y + 2, start:end].max(axis=0)
    candidates = np.flatnonzero(edge_slice.any(axis=0))
    if candidates.size == 0:
        return x

    candidate_scores = scores[candidates]
    best_score = candidate_scores.max()
    best_candidates = candidates[candidate_scores == best_score]
    nearest = best_candidates[np.argmin(np.abs((start + best_candidates) - x))]
    return int(start + nearest)


def smooth_boundary_rows(
    row_points: list[Tuple[int, int, int]],
) -> list[Tuple[int, int, int]]:
    if len(row_points) < 7:
        return row_points

    ys = np.array([point[0] for point in row_points], dtype=np.int32)
    lefts = np.array([point[1] for point in row_points], dtype=np.float32)
    rights = np.array([point[2] for point in row_points], dtype=np.float32)
    kernel = np.array([1, 3, 5, 3, 1], dtype=np.float32)
    kernel /= kernel.sum()

    lefts = np.convolve(np.pad(lefts, (2, 2), mode="edge"), kernel, mode="valid")
    rights = np.convolve(np.pad(rights, (2, 2), mode="edge"), kernel, mode="valid")
    return [
        (int(y), int(round(left)), int(round(right)))
        for y, left, right in zip(ys, lefts, rights)
    ]


def geometry_guided_subject_mask(
    image: np.ndarray,
    face_box: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    if face_box is None:
        face_box = detect_largest_face(image)
    if face_box is None:
        return None

    x, y, width, height = face_box
    image_height, image_width = image.shape[:2]
    face_center_x = int(x + width / 2)
    face_center_y = int(y + height / 2)
    mask = np.zeros((image_height, image_width), dtype=np.float32)

    head_center = (face_center_x, int(face_center_y - height * 0.08))
    head_axes = (int(width * 0.72), int(height * 1.12))
    cv2.ellipse(mask, head_center, head_axes, 0, 0, 360, 1.0, -1)

    hair_center = (face_center_x, int(y + height * 0.24))
    hair_axes = (int(width * 0.86), int(height * 0.62))
    cv2.ellipse(mask, hair_center, hair_axes, 0, 180, 360, 1.0, -1)

    neck_top = int(y + height * 0.82)
    neck_bottom = int(y + height * 1.22)
    neck_half_width = int(width * 0.23)
    cv2.rectangle(
        mask,
        (max(0, face_center_x - neck_half_width), max(0, neck_top)),
        (min(image_width - 1, face_center_x + neck_half_width), min(image_height - 1, neck_bottom)),
        1.0,
        -1,
    )

    shoulder_y = int(y + height * 1.05)
    upper_body_y = int(y + height * 1.28)
    torso_bottom = image_height - 1
    shoulder_half_width = int(width * 1.16)
    bottom_half_width = int(width * 1.42)
    torso = np.array(
        [
            [face_center_x - int(width * 0.34), shoulder_y],
            [face_center_x + int(width * 0.34), shoulder_y],
            [face_center_x + shoulder_half_width, upper_body_y],
            [face_center_x + bottom_half_width, torso_bottom],
            [face_center_x - bottom_half_width, torso_bottom],
            [face_center_x - shoulder_half_width, upper_body_y],
        ],
        dtype=np.int32,
    )
    torso[:, 0] = np.clip(torso[:, 0], 0, image_width - 1)
    torso[:, 1] = np.clip(torso[:, 1], 0, image_height - 1)
    cv2.fillPoly(mask, [torso], 1.0)

    shoulder_center = (face_center_x, int(y + height * 1.34))
    shoulder_axes = (int(width * 1.34), int(height * 0.44))
    cv2.ellipse(mask, shoulder_center, shoulder_axes, 0, 0, 360, 1.0, -1)

    blur_size = max(15, int(min(image_width, image_height) * 0.08))
    if blur_size % 2 == 0:
        blur_size += 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    mask = np.clip(mask / max(float(mask.max()), 1e-6), 0, 1)
    return clean_subject_mask(mask)


def rembg_subject_mask(image: np.ndarray) -> Optional[np.ndarray]:
    try:
        from rembg import new_session, remove
    except ImportError:
        return None

    global REMBG_SESSION
    if REMBG_SESSION is None:
        REMBG_SESSION = new_session("isnet-general-use")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format="PNG")

    try:
        output = remove(
            image_bytes.getvalue(),
            session=REMBG_SESSION,
            only_mask=True,
            post_process_mask=True,
        )
    except Exception:
        return None

    mask = Image.open(io.BytesIO(output)).convert("L")
    mask_array = np.asarray(mask, dtype=np.float32) / 255.0
    return clean_subject_mask(mask_array)


def mediapipe_subject_mask(image: np.ndarray) -> Optional[np.ndarray]:
    try:
        import mediapipe as mp
    except ImportError:
        return None

    global MEDIAPIPE_IMAGE_SEGMENTER
    if MEDIAPIPE_IMAGE_SEGMENTER is None:
        ensure_mediapipe_model()
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL_PATH)),
            output_confidence_masks=True,
            output_category_mask=False,
        )
        MEDIAPIPE_IMAGE_SEGMENTER = mp.tasks.vision.ImageSegmenter.create_from_options(
            options
        )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = MEDIAPIPE_IMAGE_SEGMENTER.segment(mp_image)
    except Exception:
        return None

    if not result.confidence_masks or len(result.confidence_masks) < 2:
        return None

    mask = np.asarray(result.confidence_masks[1].numpy_view(), dtype=np.float32)
    return clean_subject_mask(mask)


def ensure_mediapipe_model() -> None:
    if MEDIAPIPE_MODEL_PATH.exists():
        return

    MEDIAPIPE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)


def clean_subject_mask(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_uint8 = extend_subject_to_bottom_edge(mask_uint8)
    mask_float = cv2.GaussianBlur(mask_uint8.astype(np.float32) / 255.0, (7, 7), 0)
    return np.clip(mask_float, 0, 1)


def extend_subject_to_bottom_edge(mask_uint8: np.ndarray) -> np.ndarray:
    confident = mask_uint8 > 128
    image_height, image_width = confident.shape
    start_y = int(image_height * 0.55)
    bottom_curve = np.full(image_width, np.nan, dtype=np.float32)

    for x in range(image_width):
        ys = np.flatnonzero(confident[start_y:, x])
        if ys.size == 0:
            continue
        bottom_curve[x] = start_y + float(ys[-1])

    known_x = np.flatnonzero(~np.isnan(bottom_curve))
    if known_x.size < 2:
        return mask_uint8

    filled_curve = np.interp(
        np.arange(image_width),
        known_x,
        bottom_curve[known_x],
    )
    kernel_width = max(15, int(image_width * 0.09))
    if kernel_width % 2 == 0:
        kernel_width += 1
    kernel = np.hanning(kernel_width).astype(np.float32)
    kernel /= kernel.sum()
    padded_curve = np.pad(filled_curve, (kernel_width // 2,), mode="edge")
    smooth_curve = np.convolve(padded_curve, kernel, mode="valid")

    extension = np.zeros_like(mask_uint8)
    for x, y in enumerate(smooth_curve):
        extension_start = int(np.clip(round(y), start_y, image_height - 1))
        extension[extension_start:, x] = 255

    bottom_kernel_width = max(7, int(image_width * 0.04))
    if bottom_kernel_width % 2 == 0:
        bottom_kernel_width += 1
    bottom_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (bottom_kernel_width, 5),
    )
    extension = cv2.morphologyEx(extension, cv2.MORPH_CLOSE, bottom_kernel, iterations=1)
    extension = cv2.GaussianBlur(extension, (9, 9), 0)
    filled = np.maximum(mask_uint8, extension)
    return filled



def apply_white_background(image: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
    if float(subject_mask.mean()) < 0.05:
        return image

    mask = np.clip(subject_mask.astype(np.float32), 0, 1)[:, :, None]
    white = np.full_like(image, 255, dtype=np.float32)
    image_float = image.astype(np.float32)
    result = image_float * mask + white * (1 - mask)
    return np.clip(result, 0, 255).astype(np.uint8)


def hard_replace_background_with_white(
    image: np.ndarray,
    face_box: Tuple[int, int, int, int],
) -> np.ndarray:
    mask = foreground_mask_from_face(image, face_box)
    if mask is None:
        return image

    foreground_ratio = float(mask.mean())
    if foreground_ratio < 0.12 or foreground_ratio > 0.97:
        return image

    mask_3d = mask[:, :, None].astype(np.uint8)
    white = np.full_like(image, 255)
    return image * mask_3d + white * (1 - mask_3d)


def neutralize_background(image: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
    background_mask = (1 - subject_mask).astype(np.float32)
    if float(background_mask.mean()) < 0.05:
        return image

    blurred = cv2.GaussianBlur(background_mask, (31, 31), 0)
    alpha = np.clip(blurred * 0.65, 0, 0.65)[:, :, None]

    white = np.full_like(image, 255, dtype=np.float32)
    image_float = image.astype(np.float32)
    neutralized = image_float * (1 - alpha) + white * alpha
    return np.clip(neutralized, 0, 255).astype(np.uint8)


def geometric_foreground_mask(
    image: np.ndarray,
    face_box: Tuple[int, int, int, int],
) -> np.ndarray:
    x, y, width, height = face_box
    image_height, image_width = image.shape[:2]
    face_center_x = int(x + width / 2)
    face_center_y = int(y + height / 2)

    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    head_center = (face_center_x, int(face_center_y - height * 0.05))
    head_axes = (int(width * 0.62), int(height * 1.05))
    cv2.ellipse(mask, head_center, head_axes, 0, 0, 360, 1, -1)

    shoulder_y = int(y + height * 0.95)
    torso_bottom = image_height
    left_shoulder = int(face_center_x - width * 1.02)
    right_shoulder = int(face_center_x + width * 1.02)
    left_waist = int(face_center_x - width * 0.9)
    right_waist = int(face_center_x + width * 0.9)
    torso = np.array(
        [
            [left_shoulder, shoulder_y],
            [right_shoulder, shoulder_y],
            [right_waist, torso_bottom],
            [left_waist, torso_bottom],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [torso], 1)

    kernel = np.ones((9, 9), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (13, 13), 0)
    return (mask > 0.12).astype(np.uint8)


def foreground_mask_from_face(
    image: np.ndarray,
    face_box: Tuple[int, int, int, int],
) -> np.ndarray:
    prior = geometric_foreground_mask(image, face_box)
    x, y, width, height = face_box
    image_height, image_width = image.shape[:2]
    mask = np.full((image_height, image_width), cv2.GC_BGD, dtype=np.uint8)
    mask[prior == 1] = cv2.GC_PR_FGD

    face_left = max(0, x)
    face_top = max(0, y)
    face_right = min(image_width, x + width)
    face_bottom = min(image_height, y + height)
    mask[face_top:face_bottom, face_left:face_right] = cv2.GC_FGD

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            image,
            mask,
            None,
            bg_model,
            fg_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return prior

    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1,
        0,
    ).astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)

    foreground[face_top:face_bottom, face_left:face_right] = 1

    return foreground
