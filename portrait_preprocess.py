from pathlib import Path
from typing import Optional, Tuple
import json

import cv2
import numpy as np


IMAGE_SIZE = (240, 240)
CALIBRATION_PATH = Path(__file__).resolve().parent / "face_crop_calibration.json"
TARGET_FACE_WIDTH_RATIO = 0.5417
TARGET_FACE_CENTER_X = 0.4938
TARGET_FACE_CENTER_Y = 0.5333
MIN_TOP_MARGIN_FACE_RATIO = 0.65
EDGE_REFINEMENT_STEPS = 7


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
    face_in_crop = (x - left, y - top, width, height)
    portrait = replace_background_with_white(portrait, face_in_crop)
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


def save_edge_detection_preview(input_path: Path, output_path: Path) -> Path:
    image = load_image_for_cv(input_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = edge_detection_sliding_window(grayscale)
    cv2.imwrite(str(output_path), edge_image)
    return output_path


def outermost_layer_from_edges(
    edge_image: np.ndarray,
    prior_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
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
    image_height, image_width = edge_image.shape
    output = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    left_points = boundary_points_by_row(binary_edges, allowed_mask, "left")
    right_points = boundary_points_by_row(binary_edges, allowed_mask, "right")
    left_points = smooth_boundary_points(left_points)
    right_points = smooth_boundary_points(right_points)
    draw_boundary_polyline(output, left_points)
    draw_boundary_polyline(output, right_points)
    draw_top_boundary_bridge(output, left_points, right_points)
    return output


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
    if len(points) < 5:
        return points

    smoothed = []
    half_window = 2
    for index, (_, y) in enumerate(points):
        start = max(0, index - half_window)
        end = min(len(points), index + half_window + 1)
        x_values = [point[0] for point in points[start:end]]
        smoothed.append((int(np.median(x_values)), y))
    return smoothed


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
    )


def draw_top_boundary_bridge(
    output: np.ndarray,
    left_points: list[Tuple[int, int]],
    right_points: list[Tuple[int, int]],
) -> None:
    if not left_points or not right_points:
        return

    left_top = left_points[0]
    right_top = right_points[0]
    top_y = min(left_top[1], right_top[1])
    cv2.line(
        output,
        (left_top[0], top_y),
        (right_top[0], top_y),
        (255, 255, 255),
        thickness=2,
    )


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
    return image


def apply_white_background(image: np.ndarray, subject_mask: np.ndarray) -> np.ndarray:
    if float(subject_mask.mean()) < 0.05:
        return image

    mask = cv2.GaussianBlur(subject_mask.astype(np.float32), (9, 9), 0)
    mask = np.clip(mask, 0, 1)[:, :, None]
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
