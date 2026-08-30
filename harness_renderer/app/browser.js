import crypto from "node:crypto";

import { PNG } from "pngjs";

import { RendererError } from "./schemas.js";

export function sha256(content) {
  return crypto.createHash("sha256").update(content).digest("hex");
}

export function analyzePng(content) {
  let png;
  try {
    png = PNG.sync.read(content);
  } catch {
    throw new RendererError("invalid_capture", "renderer output is not a valid PNG", 502);
  }
  let count = 0;
  let mean = 0;
  let sumSquares = 0;
  let opaque = 0;
  let gray = 0;
  const colors = new Set();
  for (let offset = 0; offset < png.data.length; offset += 4) {
    const red = png.data[offset];
    const green = png.data[offset + 1];
    const blue = png.data[offset + 2];
    const alpha = png.data[offset + 3];
    const luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
    count += 1;
    const delta = luminance - mean;
    mean += delta / count;
    sumSquares += delta * (luminance - mean);
    if (alpha > 0) opaque += 1;
    if (Math.max(red, green, blue) - Math.min(red, green, blue) <= 2) gray += 1;
    if (colors.size < 4096) colors.add(`${red},${green},${blue},${alpha}`);
  }
  return {
    gray_fraction: count ? gray / count : 1,
    height: png.height,
    luminance_mean: mean,
    luminance_variance: count > 1 ? sumSquares / (count - 1) : 0,
    opaque_fraction: count ? opaque / count : 0,
    sampled_unique_colors: colors.size,
    width: png.width
  };
}

export function assertCaptureQuality(stats, viewport) {
  if (stats.width !== viewport.width || stats.height !== viewport.height) {
    throw new RendererError(
      "capture_dimensions_mismatch",
      `capture is ${stats.width}x${stats.height}, expected ${viewport.width}x${viewport.height}`,
      502,
      { actual: { height: stats.height, width: stats.width }, expected: viewport }
    );
  }
  if (
    stats.opaque_fraction < 0.99 ||
    stats.sampled_unique_colors < 16 ||
    stats.luminance_variance < 20 ||
    stats.gray_fraction > 0.98
  ) {
    throw new RendererError(
      "blank_or_gray_capture",
      "capture failed nonblank canvas checks",
      502,
      stats
    );
  }
}
