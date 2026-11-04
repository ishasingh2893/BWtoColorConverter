const test = require('node:test');
const assert = require('node:assert/strict');
const { apply, toLab, toRgb } = require('../static/color-balance.js');

test('sRGB and LAB conversion match reference colors and round trip', () => {
  const red = toLab(255, 0, 0);
  assert.ok(Math.abs(red[0] - 53.24) < 0.02);
  assert.ok(Math.abs(red[1] - 80.09) < 0.02);
  assert.ok(Math.abs(red[2] - 67.20) < 0.02);
  for (const rgb of [[0,0,0], [255,255,255], [180,140,100], [30,120,200], [128,128,128]]) {
    const result = toRgb(...toLab(...rgb));
    result.forEach((value, i) => assert.ok(Math.abs(value - rgb[i]) <= 1));
  }
});

test('zero bias restores exact output and correction never changes the cached pixels', () => {
  const original = new Uint8ClampedArray([180,140,100,255, 70,90,110,120]);
  const snapshot = original.slice();
  assert.deepEqual(apply(original, 0), snapshot);
  apply(original, 100);
  assert.deepEqual(original, snapshot);
  assert.deepEqual(apply(original, 50), apply(original, 50));
  assert.deepEqual(apply(original, 0), snapshot);
});

test('neutral correction reduces red and yellow while maintaining LAB lightness', () => {
  const original = new Uint8ClampedArray([180,140,100,255]);
  const before = toLab(...original.slice(0, 3));
  for (const strength of [25,50,100]) {
    const corrected = apply(original, strength);
    const after = toLab(...corrected.slice(0,3));
    assert.ok(Math.abs(after[0] - before[0]) < 0.3);
    assert.ok(Math.abs(after[1] - (before[1] - 3 * strength / 100)) < 0.6);
    assert.ok(Math.abs(after[2] - (before[2] - 10 * strength / 100)) < 0.6);
    assert.equal(corrected[3], 255);
  }
});

test('white padding, transparency, bounds, and alpha are preserved', () => {
  const pixels = new Uint8ClampedArray([255,255,255,255, 250,249,248,255, 120,80,40,0, 190,140,90,128]);
  const result = apply(pixels, 100);
  assert.deepEqual(result.slice(0,12), pixels.slice(0,12));
  assert.equal(result[15], 128);
  assert.deepEqual(apply(pixels, -20), pixels);
  assert.deepEqual(apply(pixels, 500), result);
});
