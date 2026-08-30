(() => {
  const MAX_ATTEMPTS = 100;
  const BRIDGE_VERSION = "2.0.0";
  const TERRIAMAP_VERSION = "0.4.6";
  const TERRIAJS_VERSION = "8.12.2";
  const CESIUM_VERSION = "23.0.2";
  const LAYER_MODEL_IDS = {
    "asset-worldcover-n30e120": "esa-worldcover-2021"
  };
  let lastDesiredMapState;

  function findTerria() {
    const root = document.querySelector("#ui");
    const rootKey = Object.keys(root || {}).find((key) =>
      key.startsWith("__reactContainer$")
    );
    if (!rootKey) return undefined;

    const stack = [root[rootKey]];
    const seen = new Set();
    while (stack.length > 0) {
      const fiber = stack.pop();
      if (!fiber || typeof fiber !== "object" || seen.has(fiber)) continue;
      seen.add(fiber);

      for (const props of [fiber.memoizedProps, fiber.pendingProps]) {
        const terria = props?.terria || props?.viewState?.terria;
        if (terria?.currentViewer && terria?.mapNavigationModel) return terria;
      }

      if (fiber.child) stack.push(fiber.child);
      if (fiber.sibling) stack.push(fiber.sibling);
    }
    return undefined;
  }

  function exposeZoomOnSmallScreens() {
    const terria = findTerria();
    const zoomItem = terria?.mapNavigationModel?.items?.find(
      (item) => item.id === "zoom"
    );
    if (!zoomItem) return false;

    // TerriaJS 8.12.2 registers ZoomControl as desktop-only.
    zoomItem.screenSize = "any";
    return true;
  }

  function cloneJson(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function assertFiniteNumber(value, name) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new TypeError(`${name} must be a finite number`);
    }
  }

  function validateBbox(bbox) {
    if (!bbox || typeof bbox !== "object") {
      throw new TypeError("map.bbox must be an object");
    }
    for (const key of ["west", "south", "east", "north"]) {
      assertFiniteNumber(bbox[key], `map.bbox.${key}`);
    }
    if (
      bbox.west < -180 || bbox.east > 180 ||
      bbox.south < -90 || bbox.north > 90 ||
      bbox.west >= bbox.east || bbox.south >= bbox.north
    ) {
      throw new RangeError("map.bbox is outside valid WGS84 bounds");
    }
  }

  function validateMapState(mapState) {
    if (!mapState || typeof mapState !== "object") {
      throw new TypeError("map state must be an object");
    }
    validateBbox(mapState.bbox);
    if (!mapState.layers || typeof mapState.layers !== "object") {
      throw new TypeError("map.layers must be an object");
    }
    for (const [layerId, layer] of Object.entries(mapState.layers)) {
      if (!layer || layer.layer_id !== layerId) {
        throw new TypeError(`map layer key does not match layer_id: ${layerId}`);
      }
      if (!LAYER_MODEL_IDS[layer.asset_id]) {
        throw new RangeError(`unsupported asset_id: ${layer.asset_id}`);
      }
      if (typeof layer.visible !== "boolean") {
        throw new TypeError(`layer ${layerId} visible must be boolean`);
      }
      assertFiniteNumber(layer.opacity, `layer ${layerId} opacity`);
      if (layer.opacity < 0 || layer.opacity > 1) {
        throw new RangeError(`layer ${layerId} opacity must be between 0 and 1`);
      }
    }
    return cloneJson(mapState);
  }

  function degrees(value) {
    return value * 180 / Math.PI;
  }

  function radians(value) {
    return value * Math.PI / 180;
  }

  function sleep(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  }

  async function waitUntil(predicate, timeoutMs, message) {
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      const value = predicate();
      if (value) return value;
      await sleep(50);
    }
    throw new Error(message);
  }

  async function bridgeReady(options = {}) {
    const timeoutMs = Math.min(Math.max(options.timeout_ms || 30000, 100), 60000);
    const terria = await waitUntil(() => {
      const candidate = findTerria();
      const canvas = candidate?.currentViewer?.scene?.canvas;
      return candidate && canvas?.width > 0 && canvas?.height > 0
        ? candidate
        : undefined;
    }, timeoutMs, "TerriaMap did not become bridge-ready");
    return {
      ready: true,
      canvas: {
        height: terria.currentViewer.scene.canvas.height,
        width: terria.currentViewer.scene.canvas.width
      },
      versions: bridgeVersions()
    };
  }

  function bridgeVersions() {
    return {
      bridge: BRIDGE_VERSION,
      cesium: CESIUM_VERSION,
      terriamap: TERRIAMAP_VERSION,
      terriajs: TERRIAJS_VERSION
    };
  }

  async function waitForStableFrame(options = {}) {
    const timeoutMs = Math.min(Math.max(options.timeout_ms || 30000, 100), 60000);
    const terria = findTerria();
    if (!terria) throw new Error("TerriaMap is not ready");

    await waitUntil(() => {
      const desiredLayers = Object.values(lastDesiredMapState?.layers || {});
      const layersReady = desiredLayers.every((layer) => {
        const modelId = LAYER_MODEL_IDS[layer.asset_id];
        const model = terria.workbench.items.find((item) => item.uniqueId === modelId);
        if (!model) return false;
        if (!layer.visible) return !model.isLoading;
        return !model.isLoading && !model.isLoadingMapItems && model.mapItems?.length > 0;
      });
      return layersReady && !terria.currentViewer.isMapZooming &&
        terria.currentViewer.scene?.globe?.tilesLoaded === true;
    }, timeoutMs, "TerriaMap layers did not reach a stable frame");

    for (let index = 0; index < 3; index += 1) {
      terria.currentViewer.notifyRepaintRequired?.();
      terria.currentViewer.scene?.requestRender?.();
      await new Promise((resolve) => window.requestAnimationFrame(resolve));
    }
    await sleep(100);
    return readState();
  }

  function readState() {
    const terria = findTerria();
    if (!terria) throw new Error("TerriaMap is not ready");
    if (!lastDesiredMapState) throw new Error("no harness map state has been applied");

    const camera = terria.currentViewer.getCurrentCameraView();
    const rectangle = camera?.rectangle;
    if (!rectangle) throw new Error("renderer camera rectangle is unavailable");
    const actualBbox = {
      east: degrees(rectangle.east),
      north: degrees(rectangle.north),
      south: degrees(rectangle.south),
      west: degrees(rectangle.west)
    };
    const target = lastDesiredMapState.bbox;
    const targetCenter = {
      latitude: (target.south + target.north) / 2,
      longitude: (target.west + target.east) / 2
    };
    const actualCenter = {
      latitude: (actualBbox.south + actualBbox.north) / 2,
      longitude: (actualBbox.west + actualBbox.east) / 2
    };
    const tolerance = 0.005;
    const cameraConsistent =
      actualBbox.west <= target.west + tolerance &&
      actualBbox.south <= target.south + tolerance &&
      actualBbox.east >= target.east - tolerance &&
      actualBbox.north >= target.north - tolerance &&
      Math.abs(actualCenter.longitude - targetCenter.longitude) <= tolerance &&
      Math.abs(actualCenter.latitude - targetCenter.latitude) <= tolerance;

    const layers = {};
    let layersConsistent = true;
    for (const [layerId, desired] of Object.entries(lastDesiredMapState.layers)) {
      const modelId = LAYER_MODEL_IDS[desired.asset_id];
      const model = terria.workbench.items.find((item) => item.uniqueId === modelId);
      const actual = {
        asset_id: desired.asset_id,
        layer_id: layerId,
        loaded: Boolean(model && !model.isLoading && !model.isLoadingMapItems),
        opacity: model?.opacity,
        visible: model?.show
      };
      layers[layerId] = actual;
      layersConsistent = layersConsistent && actual.loaded &&
        actual.visible === desired.visible &&
        Math.abs(actual.opacity - desired.opacity) <= 1e-6;
    }

    return {
      actual_bbox: actualBbox,
      actual_center: actualCenter,
      camera_consistent: cameraConsistent,
      consistent: cameraConsistent && layersConsistent,
      desired_bbox: cloneJson(target),
      desired_center: targetCenter,
      layers,
      layers_consistent: layersConsistent,
      stable: terria.currentViewer.scene?.globe?.tilesLoaded === true &&
        !terria.currentViewer.isMapZooming,
      versions: bridgeVersions()
    };
  }

  async function applyState(mapState, options = {}) {
    const desired = validateMapState(mapState);
    await bridgeReady(options);
    const terria = findTerria();

    for (const layer of Object.values(desired.layers)) {
      const modelId = LAYER_MODEL_IDS[layer.asset_id];
      const model = terria.workbench.items.find((item) => item.uniqueId === modelId);
      if (!model) throw new Error(`TerriaMap layer is missing: ${modelId}`);
      model.setTrait("user", "show", layer.visible);
      model.setTrait("user", "opacity", layer.opacity);
    }

    lastDesiredMapState = desired;
    const bbox = desired.bbox;
    await terria.currentViewer.zoomTo({
      rectangle: {
        east: radians(bbox.east),
        north: radians(bbox.north),
        south: radians(bbox.south),
        west: radians(bbox.west)
      }
    }, 0);
    terria.currentViewer.notifyRepaintRequired?.();
    terria.currentViewer.scene?.requestRender?.();
    const readback = await waitForStableFrame(options);
    if (!readback.consistent) {
      throw new Error("TerriaMap read-back does not match desired map state");
    }
    return readback;
  }

  window.__EO_HARNESS_V2__ = Object.freeze({
    applyState,
    readState,
    ready: bridgeReady,
    version: BRIDGE_VERSION,
    versions: bridgeVersions,
    waitForStableFrame
  });

  function labelLanguageButton() {
    const button = document.querySelector(".tjs-menu-bar__langBtn");
    if (!button) return;

    const label = "切换语言 / Change language";
    button.setAttribute("aria-label", label);
    button.setAttribute("title", label);
  }

  function installZoomRenderBridge() {
    const marker = "eoHarnessZoomRenderBridge";
    if (document.documentElement.dataset[marker]) return;

    document.addEventListener("click", (event) => {
      const target = event.target instanceof Element ? event.target : undefined;
      if (!target?.closest('[class*="ZoomControl__StyledZoomControl"]')) return;

      const terria = findTerria();
      if (!terria) return;

      const renderUntil = performance.now() + 350;
      const requestFrame = () => {
        terria.currentViewer?.notifyRepaintRequired?.();
        terria.currentViewer?.scene?.requestRender?.();
        if (performance.now() < renderUntil) {
          window.requestAnimationFrame(requestFrame);
        }
      };
      window.requestAnimationFrame(requestFrame);
    });

    document.documentElement.dataset[marker] = "true";
  }

  installZoomRenderBridge();

  let attempts = 0;
  const timer = window.setInterval(() => {
    attempts += 1;
    labelLanguageButton();
    if (exposeZoomOnSmallScreens() || attempts >= MAX_ATTEMPTS) {
      window.clearInterval(timer);
    }
  }, 200);
})();
