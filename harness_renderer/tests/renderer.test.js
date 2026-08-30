import assert from "node:assert/strict";
import test from "node:test";

import { PNG } from "pngjs";

import { analyzePng, assertCaptureQuality, sha256 } from "../app/browser.js";
import { RendererError, validateMapState } from "../app/schemas.js";
import { routeRendererRequest } from "../app/sessions.js";

function pngBuffer(width, height, pixel) {
  const png = new PNG({ width, height });
  for (let offset = 0; offset < png.data.length; offset += 4) {
    const [red, green, blue, alpha] = pixel(offset / 4);
    png.data[offset] = red;
    png.data[offset + 1] = green;
    png.data[offset + 2] = blue;
    png.data[offset + 3] = alpha;
  }
  return PNG.sync.write(png);
}

function mockRoute(url) {
  const calls = [];
  return {
    calls,
    request: () => ({ url: () => url }),
    continue: async () => calls.push({ method: "continue" }),
    fulfill: async (options) => calls.push({ method: "fulfill", options }),
    abort: async (reason) => calls.push({ method: "abort", reason })
  };
}

test("PNG analysis rejects blank gray output", () => {
  const content = pngBuffer(4, 4, () => [128, 128, 128, 255]);
  const stats = analyzePng(content);
  assert.throws(
    () => assertCaptureQuality(stats, { width: 4, height: 4 }),
    (error) => error instanceof RendererError && error.code === "blank_or_gray_capture"
  );
});

test("PNG analysis accepts varied opaque output", () => {
  const content = pngBuffer(16, 16, (index) => [index % 256, (index * 7) % 256, (index * 19) % 256, 255]);
  const stats = analyzePng(content);
  assert.doesNotThrow(() => assertCaptureQuality(stats, { width: 16, height: 16 }));
  assert.equal(sha256(content).length, 64);
});

test("map state validation is strict", () => {
  const state = {
    bbox: { west: 121.45, south: 31.2, east: 121.55, north: 31.3 },
    center: { longitude: 121.5, latitude: 31.25 },
    layers: {
      "layer-asset-worldcover-n30e120": {
        asset_id: "asset-worldcover-n30e120",
        layer_id: "layer-asset-worldcover-n30e120",
        name: "WorldCover",
        opacity: 0.85,
        style_id: "worldcover-official-rgb-v1",
        time_range: null,
        visible: true
      }
    },
    active_time_range: null
  };
  assert.deepEqual(validateMapState(state), state);
  assert.throws(
    () => validateMapState({ ...state, bbox: { ...state.bbox, east: 999 } }),
    RendererError
  );
});

test("renderer request routing keeps font CSS offline and blocks other external origins", async () => {
  const allowedOrigin = "http://terriamap:3001";
  const sameOrigin = mockRoute("http://terriamap:3001/build/901.TerriaMap.css");
  await routeRendererRequest(sameOrigin, allowedOrigin);
  assert.deepEqual(sameOrigin.calls, [{ method: "continue" }]);

  const fontCss = mockRoute("https://fonts.googleapis.com/css?family=Roboto:300,400,500");
  await routeRendererRequest(fontCss, allowedOrigin);
  assert.deepEqual(fontCss.calls, [{
    method: "fulfill",
    options: {
      status: 200,
      contentType: "text/css; charset=utf-8",
      body: ""
    }
  }]);

  const external = mockRoute("https://example.com/remote.js");
  await routeRendererRequest(external, allowedOrigin);
  assert.deepEqual(external.calls, [{ method: "abort", reason: "blockedbyclient" }]);
});
