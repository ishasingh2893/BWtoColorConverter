/* Adjust the outline geometry only. This module never reads image pixels. */
(function (root) {
  const limit = (value, min, max) => Math.max(min, Math.min(max, value));
  function contour(points, anchors, offsets) {
    const crownY = Math.min(...points.map((p) => p.y));
    return points.map((point) => {
      const side = point.x < 119.5 ? 0 : 1;
      const controls = [
        { y: crownY, offset: offsets[8] },
        ...[side + 4, side + 6, side, side + 2].map((index) => ({
          y: anchors[index].y, offset: offsets[index],
        })),
      ];
      if (point.y <= crownY) return { x: point.x + offsets[8].x, y: point.y + offsets[8].y };
      const segment = controls.findIndex((control, i) => i < controls.length - 1 &&
        point.y >= control.y && point.y <= controls[i + 1].y);
      if (segment < 0) return { ...point };
      const from = controls[segment], to = controls[segment + 1];
      const t = limit((point.y - from.y) / (to.y - from.y), 0, 1);
      const blend = t * t * (3 - 2 * t);
      return {
        x: limit(point.x + from.offset.x * (1 - blend) + to.offset.x * blend, 0, 240),
        y: point.y + from.offset.y * (1 - t) + to.offset.y * t,
      };
    });
  }
  function controlOffset(index, dx, dy, offsets, anchors, crownY) {
    if (index === 8) {
      const left = anchors[4].x + offsets[4].x;
      const right = anchors[5].x + offsets[5].x;
      const lowest = Math.min(anchors[4].y + offsets[4].y, anchors[5].y + offsets[5].y) - 12;
      return {
        x: limit(offsets[8].x + dx, Math.max(-24, left + 12 - anchors[8].x), Math.min(24, right - 12 - anchors[8].x)),
        y: limit(offsets[8].y + dy, 4 - crownY, Math.min(24, lowest - crownY)),
      };
    }
    const side = index % 2;
    const order = [side + 4, side + 6, side, side + 2];
    const position = order.indexOf(index);
    const previous = position > 0 ? order[position - 1] : null;
    const next = order[position + 1];
    const minY = previous === null ? crownY + offsets[8].y + 12 : anchors[previous].y + offsets[previous].y + 10;
    const maxY = anchors[next].y + offsets[next].y - 12;
    const anchor = anchors[index];
    return {
      x: limit(offsets[index].x + dx,
        Math.max(-24, (side ? 130 : 4) - anchor.x),
        Math.min(24, (side ? 236 : 109) - anchor.x)),
      y: limit(offsets[index].y + dy, Math.max(-24, minY - anchor.y), Math.min(24, maxY - anchor.y)),
    };
  }
  function edgeOffset(index, dx, dy, offsets, anchors) {
    const side = index % 2;
    const anchor = anchors[index];
    const proposedX = limit(anchor.x + offsets[index].x + dx, side ? 145 : 0, side ? 240 : 95);
    const minimumY = anchors[side].y + offsets[side].y + 12;
    const proposedY = limit(anchor.y + offsets[index].y + dy, minimumY, 240);
    const distanceToSide = side ? 240 - proposedX : proposedX;
    return 240 - proposedY > distanceToSide
      ? { x: 0, y: proposedY - 240 }
      : { x: proposedX - anchor.x, y: 0 };
  }
  function path(points) {
    let result = `M ${points[0].x} ${points[0].y}`;
    for (let i = 0; i < points.length - 1; i++) {
      const before = points[Math.max(0, i - 1)];
      const current = points[i];
      const next = points[i + 1];
      const after = points[Math.min(points.length - 1, i + 2)];
      result += ` C ${limit(current.x + (next.x - before.x) / 6, 0, 240)} ${limit(current.y + (next.y - before.y) / 6, 0, 240)}`;
      result += ` ${limit(next.x - (after.x - current.x) / 6, 0, 240)} ${limit(next.y - (after.y - current.y) / 6, 0, 240)} ${next.x} ${next.y}`;
    }
    return result;
  }
  const api = { contour, path, edgeOffset, controlOffset, limit };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.PortraitBoundary = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
