"""Build the crop overlay from the training portraits' alpha silhouettes.

Run: .venv/bin/python tools/build_portrait_guide.py
This is an offline dataset measurement; uploaded portraits are never segmented.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def build_guide(source, output):
    masks = []
    for path in sorted(source.iterdir()):
        if path.suffix.lower() not in {".png", ".webp"}:
            continue
        with Image.open(path) as image:
            if "A" not in image.getbands():
                continue
            alpha = np.asarray(image.convert("RGBA").resize((240, 240)))[:, :, 3]
            if alpha.min() == alpha.max():
                continue
            masks.append(alpha > 32)
    if not masks:
        raise ValueError("No training portraits with transparent backgrounds found")

    # The majority silhouette captures typical framing without following one
    # player's hairstyle, ear shape, or shirt. Average both sides for a neutral guide.
    coverage = np.mean(masks, axis=0)
    silhouette = (coverage + coverage[:, ::-1]) / 2 >= 0.5
    rows = np.flatnonzero(silhouette.any(axis=1))
    top = int(rows[0])
    y_values = np.arange(top, 240)
    widths = np.array([
        max(abs(np.flatnonzero(silhouette[y]) - 119.5))
        for y in y_values
    ])
    kernel = np.array([1, 2, 3, 2, 1]) / 9
    widths = np.convolve(np.pad(widths, (2, 2), mode="edge"), kernel, mode="valid")
    sample = list(range(0, len(y_values), 4))
    if sample[-1] != len(y_values) - 1:
        sample.append(len(y_values) - 1)
    # Travel up the left edge, over the crown, and down the right edge.
    points = [(0, 240)] + [(119.5 - widths[i], float(y_values[i])) for i in reversed(sample)]
    points += [(119.5, max(0, top - 2))]
    points += [(119.5 + widths[i], float(y_values[i])) for i in sample]
    # Close along the frame's lower edge. Quadratic midpoint curves soften the
    # measured contour without adding invented head/neck/shoulder proportions.
    points += [(240, 240)]
    mid = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    start = mid(points[-1], points[0])
    path = f"M {start[0]:.2f} {start[1]:.2f} "
    for i, point in enumerate(points):
        end = mid(point, points[(i + 1) % len(points)])
        path += f"Q {point[0]:.2f} {point[1]:.2f} {end[0]:.2f} {end[1]:.2f} "
    path += "Z"
    anchors = []
    for control_y in (192,):
        for side in (0, 1):
            anchors.append(min(points, key=lambda p: abs(p[1] - control_y) +
                               (1000 if (p[0] > 119.5) != bool(side) else 0)))
    anchors += [(0, 240), (240, 240)]
    for control_y in (64, 128):
        for side in (0, 1):
            anchors.append(min(points, key=lambda p: abs(p[1] - control_y) +
                               (1000 if (p[0] > 119.5) != bool(side) else 0)))
    anchors.append(min(points, key=lambda p: p[1]))
    output.with_suffix(".json").write_text(json.dumps({
        "points": [{"x": round(x, 3), "y": round(y, 3)} for x, y in points],
        "anchors": [{"x": round(x, 3), "y": round(y, 3)} for x, y in anchors],
        "dataset_images": len(masks),
    }, indent=2) + "\n")
    output.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240">
  <title>Training portrait head and shoulder guide</title>
  <desc>Symmetric majority silhouette from {len(masks)} transparent training portraits. Includes hair, neck and shoulders with typical headroom.</desc>
  <path d="M0 0H240V240H0Z {path}" fill="#14211b" fill-opacity="0.45" fill-rule="evenodd"/>
  <path d="{path}" fill="none" stroke="#14211b" stroke-opacity="0.65" stroke-width="2.6"/>
  <path d="{path}" fill="none" stroke="#c7f36b" stroke-width="1.2"/>
</svg>
''')
    print(f"Built {output} from {len(masks)} portraits; typical headroom: {top}/240 pixels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "static" / "portrait-guide.svg")
    args = parser.parse_args()
    build_guide(args.source, args.output)
