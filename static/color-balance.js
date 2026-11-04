/* sRGB / CIELAB D65 color balance. Amount 0 preserves the original bytes. */
(function (root) {
  const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
  const linear = (v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
  const gamma = (v) => 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055);
  const f = (v) => v > 216 / 24389 ? Math.cbrt(v) : (24389 / 27 * v + 16) / 116;
  const inverse = (v) => v > 6 / 29 ? v ** 3 : (116 * v - 16) / (24389 / 27);
  function toLab(r, g, b) {
    r = linear(r / 255); g = linear(g / 255); b = linear(b / 255);
    const x = f((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047);
    const y = f(0.2126729 * r + 0.7151522 * g + 0.0721750 * b);
    const z = f((0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883);
    return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
  }
  function toRgb(l, a, b) {
    const y = (l + 16) / 116;
    const x = inverse(y + a / 500) * 0.95047;
    const z = inverse(y - b / 200) * 1.08883;
    const yy = inverse(y);
    return [
      gamma(3.2404542 * x - 1.5371385 * yy - 0.4985314 * z),
      gamma(-0.9692660 * x + 1.8760108 * yy + 0.0415560 * z),
      gamma(0.0556434 * x - 0.2040259 * yy + 1.0572252 * z),
    ].map((v) => Math.round(clamp(v, 0, 255)));
  }
  function apply(pixels, amount) {
    const strength = clamp(Number(amount) || 0, 0, 100) / 100;
    const output = new Uint8ClampedArray(pixels);
    if (!strength) return output;
    for (let i = 0; i < pixels.length; i += 4) {
      const [r, g, b, alpha] = pixels.subarray(i, i + 4);
      // Preserve white padding and transparent pixels.
      if (!alpha || (r >= 248 && g >= 248 && b >= 248)) continue;
      const [l, a, bb] = toLab(r, g, b);
      const rgb = toRgb(l, a - 3 * strength, bb - 10 * strength);
      output.set(rgb, i);
    }
    return output;
  }
  const api = { apply, toLab, toRgb };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ColorBalance = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
