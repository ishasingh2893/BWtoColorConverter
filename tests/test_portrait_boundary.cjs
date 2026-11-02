const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const boundary = require('../static/portrait-boundary.js');
const guide = require('../static/portrait-guide.json');
const zero = () => guide.anchors.map(() => ({ x: 0, y: 0 }));

test('reset restores the measured training outline exactly', () => {
  assert.deepEqual(boundary.contour(guide.points, guide.anchors, zero()), guide.points);
});

test('all eight handles independently reshape their contour and leave the crown fixed', () => {
  for (const indexOfControl of [0, 1, 2, 3, 4, 5, 6, 7]) {
    const side = indexOfControl % 2;
    const offsets = zero();
    offsets[indexOfControl] = { x: side ? -16 : 16, y: indexOfControl === 2 || indexOfControl === 3 ? 0 : 10 };
    const result = boundary.contour(guide.points, guide.anchors, offsets);
    guide.points.forEach((point, i) => {
      if (point.y <= 18 || (point.x < 119.5 ? 0 : 1) !== side) {
        assert.deepEqual(result[i], point);
      }
    });
    const anchor = guide.anchors[indexOfControl];
    const index = guide.points.findIndex((p) => p.x === anchor.x && p.y === anchor.y);
    assert.ok(index >= 0);
    assert.deepEqual(result[index], { x: anchor.x + offsets[indexOfControl].x, y: anchor.y + offsets[indexOfControl].y });
    guide.anchors.forEach((other, j) => {
      if (j === indexOfControl) return;
      const at = guide.points.findIndex((p) => p.x === other.x && p.y === other.y);
      assert.deepEqual(result[at], other, 'Other control points stay fixed');
    });
    const path = boundary.path(result);
    assert.ok(path.startsWith('M ') && !path.endsWith(' Z'));
    assert.ok(!path.includes('NaN'));
  }
});

test('changing the outline never changes the preview or submitted photo draw', () => {
  const elements = new Map();
  const cropDraws = [];
  const makeContext = () => ({
    setTransform() {}, fillRect() {}, clearRect() {},
    drawImage(...args) { cropDraws.push(args); },
  });
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      style: {}, classList: { toggle() {}, remove() {} },
      textContent: id === '#portrait-guide-data' ? JSON.stringify(guide) : '',
      addEventListener() {}, setAttribute() {}, getContext: makeContext,
    });
    return elements.get(id);
  }
  const sandbox = vm.createContext({
    document: { querySelector: element, createElement: () => ({ getContext: makeContext }) },
    PortraitBoundary: boundary,
  });
  vm.runInContext(fs.readFileSync(require.resolve('../static/app.js'), 'utf8'), sandbox);
  vm.runInContext('image = {width: 240, height: 240}; draw();', sandbox);
  const original = cropDraws.slice();
  cropDraws.length = 0;
  vm.runInContext('moveOutline(0, 18, -5); moveOutline(1, -10, 8); moveOutline(2, -12, 9); moveOutline(3, 15, -4); moveOutline(4, 10, -8); moveOutline(5, -12, 5); moveOutline(6, 8, 10); moveOutline(7, -6, -9); moveOutline(8, 10, -6); draw();', sandbox);
  assert.deepEqual(cropDraws, original);
});

test('shoulder adjustments keep each contour ordered and inside the frame', () => {
  const offsets = zero();
  offsets[2] = { x: 0, y: -30 };
  offsets[3] = { x: -40, y: 0 };
  const result = boundary.contour(guide.points, guide.anchors, offsets);
  for (const side of [0, 1]) {
    const rows = result.filter((p) => (p.x < 119.5 ? 0 : 1) === side && p.y >= 140 && p.y < 240);
    assert.ok(rows.every((p) => p.x >= 0 && p.x <= 240));
    for (let i = 1; i < rows.length; i++) {
      assert.ok(side === 0 ? rows[i].y < rows[i - 1].y : rows[i].y > rows[i - 1].y);
    }
  }
});

test('round shoulder controls start at frame corners and can meet an edge away from the corners', () => {
  assert.deepEqual(guide.anchors.slice(2, 4), [{ x: 0, y: 240 }, { x: 240, y: 240 }]);
  const offsets = zero();
  offsets[2] = boundary.edgeOffset(2, 45, 0, offsets, guide.anchors);
  offsets[3] = boundary.edgeOffset(3, 0, -30, offsets, guide.anchors);
  const result = boundary.contour(guide.points, guide.anchors, offsets);
  assert.deepEqual(result[0], { x: 45, y: 240 });
  assert.deepEqual(result.at(-1), { x: 240, y: 210 });
  assert.ok(!result.some((point) => point.x === 0 && point.y === 240));
  assert.ok(!result.some((point) => point.x === 240 && point.y === 240));
  assert.ok(boundary.path(result).startsWith('M 45 240'));
});

test('edge controls stay on the frame and below the corresponding neck', () => {
  const offsets = zero();
  for (const index of [2, 3]) {
    for (const [dx, dy] of [[-500, -500], [500, 500], [30, -50], [-30, -50]]) {
      const offset = boundary.edgeOffset(index, dx, dy, offsets, guide.anchors);
      const x = guide.anchors[index].x + offset.x;
      const y = guide.anchors[index].y + offset.y;
      assert.ok(y === 240 || x === 0 || x === 240);
      assert.ok(x >= 0 && x <= 240 && y <= 240);
      assert.ok(y >= guide.anchors[index % 2].y + 12);
    }
  }
});

test('new head and cheek controls stay in order and on their own side', () => {
  assert.equal(guide.anchors.length, 9);
  const offsets = zero();
  const crownY = Math.min(...guide.points.map((point) => point.y));
  for (const index of [4, 5, 6, 7, 0, 1]) {
    for (const [dx, dy] of [[500, 500], [-500, -500]]) {
      offsets[index] = boundary.controlOffset(index, dx, dy, offsets, guide.anchors, crownY);
      const point = { x: guide.anchors[index].x + offsets[index].x,
        y: guide.anchors[index].y + offsets[index].y };
      assert.ok(index % 2 ? point.x >= 130 : point.x <= 109);
      for (const side of [0, 1]) {
        const sequence = [side + 4, side + 6, side, side + 2].map((i) => guide.anchors[i].y + offsets[i].y);
        assert.ok(sequence.every((y, i) => i === 0 || y >= sequence[i - 1] + 10));
      }
    }
  }
});

test('top handle moves the crown and both upper curves without moving other anchors', () => {
  const offsets = zero();
  offsets[8] = boundary.controlOffset(8, 12, -8, offsets, guide.anchors, guide.anchors[8].y);
  const result = boundary.contour(guide.points, guide.anchors, offsets);
  const top = guide.points.findIndex((p) => p.y === guide.anchors[8].y);
  assert.deepEqual(result[top], { x: guide.anchors[8].x + 12, y: guide.anchors[8].y - 8 });
  guide.anchors.slice(0, 8).forEach((anchor) => {
    const at = guide.points.findIndex((p) => p.x === anchor.x && p.y === anchor.y);
    assert.deepEqual(result[at], anchor);
  });
  for (const side of [0, 1]) {
    assert.ok(guide.points.some((p, i) => p.y > guide.anchors[8].y && p.y < guide.anchors[4].y &&
      (p.x < 119.5 ? 0 : 1) === side && result[i].y !== p.y));
  }
  const extreme = boundary.controlOffset(8, 500, -500, offsets, guide.anchors, guide.anchors[8].y);
  assert.equal(guide.anchors[8].y + extreme.y, 4);
  assert.ok(Math.abs(extreme.x) <= 24);
});
