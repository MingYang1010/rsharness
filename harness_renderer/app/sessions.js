import { createRequire } from "node:module";

import { chromium } from "playwright";

import { analyzePng, assertCaptureQuality, sha256 } from "./browser.js";
import { RendererError, validateEpisodeId, validateMapState } from "./schemas.js";

const require = createRequire(import.meta.url);
const PLAYWRIGHT_VERSION = require("playwright/package.json").version;

export async function routeRendererRequest(route, allowedOrigin) {
  const url = new URL(route.request().url());
  if (["about:", "blob:", "data:"].includes(url.protocol) || url.origin === allowedOrigin) {
    await route.continue();
    return;
  }
  if (url.hostname === "fonts.googleapis.com") {
    await route.fulfill({
      status: 200,
      contentType: "text/css; charset=utf-8",
      body: ""
    });
    return;
  }
  await route.abort("blockedbyclient");
}

export class SessionManager {
  constructor(options = {}) {
    this.terriaUrl = options.terriaUrl || "http://terriamap:3001/?lng=en";
    this.viewport = options.viewport || { width: 1024, height: 768 };
    this.timeoutMs = options.timeoutMs || 45000;
    this.sessions = new Map();
    this.browser = undefined;
  }

  async start() {
    if (this.browser) return;
    this.browser = await chromium.launch({
      headless: true,
      chromiumSandbox: false,
      args: [
        "--disable-background-networking",
        "--disable-dev-shm-usage",
        "--disable-features=Translate,MediaRouter,OptimizationHints",
        "--force-color-profile=srgb",
        "--force-device-scale-factor=1",
        "--use-gl=swiftshader"
      ]
    });
  }

  async stop() {
    const sessions = [...this.sessions.values()];
    this.sessions.clear();
    await Promise.allSettled(sessions.map((session) => session.context.close()));
    if (this.browser) await this.browser.close();
    this.browser = undefined;
  }

  versions() {
    return {
      chromium: this.browser?.version() || null,
      playwright: PLAYWRIGHT_VERSION,
      renderer: "1.0.0"
    };
  }

  async create(episodeId) {
    validateEpisodeId(episodeId);
    await this.start();
    if (this.sessions.has(episodeId)) return this.inspect(episodeId);

    const context = await this.browser.newContext({
      colorScheme: "light",
      deviceScaleFactor: 1,
      hasTouch: false,
      locale: "en-US",
      reducedMotion: "reduce",
      serviceWorkers: "block",
      timezoneId: "UTC",
      viewport: this.viewport
    });
    const page = await context.newPage();
    const allowedOrigin = new URL(this.terriaUrl).origin;
    await page.route("**/*", (route) => routeRendererRequest(route, allowedOrigin));
    page.setDefaultTimeout(this.timeoutMs);
    await page.goto(this.terriaUrl, { waitUntil: "domcontentloaded", timeout: this.timeoutMs });
    await page.addStyleTag({
      content: "*,*::before,*::after{animation:none!important;caret-color:transparent!important;transition:none!important}"
    });
    await page.waitForFunction(
      () => window.__EO_HARNESS_V2__?.version === "2.0.0",
      undefined,
      { timeout: this.timeoutMs }
    );
    const bridge = await page.evaluate(
      (timeoutMs) => window.__EO_HARNESS_V2__.ready({ timeout_ms: timeoutMs }),
      this.timeoutMs
    );
    this.sessions.set(episodeId, {
      bridge,
      context,
      createdAt: new Date().toISOString(),
      lastReadback: null,
      page
    });
    return this.inspect(episodeId);
  }

  get(episodeId) {
    validateEpisodeId(episodeId);
    const session = this.sessions.get(episodeId);
    if (!session) {
      throw new RendererError("session_not_found", "renderer session does not exist", 404);
    }
    return session;
  }

  async apply(episodeId, mapState) {
    const session = this.get(episodeId);
    const validated = validateMapState(mapState);
    const readback = await session.page.evaluate(
      ({ state, timeoutMs }) => window.__EO_HARNESS_V2__.applyState(
        state,
        { timeout_ms: timeoutMs }
      ),
      { state: validated, timeoutMs: this.timeoutMs }
    );
    if (!readback.consistent || !readback.stable) {
      throw new RendererError(
        "renderer_state_mismatch",
        "TerriaMap read-back does not match desired state",
        409,
        readback
      );
    }
    session.lastReadback = readback;
    return readback;
  }

  async read(episodeId) {
    const session = this.get(episodeId);
    const readback = await session.page.evaluate(
      () => window.__EO_HARNESS_V2__.readState()
    );
    session.lastReadback = readback;
    return readback;
  }

  async capture(episodeId) {
    const session = this.get(episodeId);
    const stable = await session.page.evaluate(
      (timeoutMs) => window.__EO_HARNESS_V2__.waitForStableFrame({ timeout_ms: timeoutMs }),
      this.timeoutMs
    );
    if (!stable.consistent || !stable.stable) {
      throw new RendererError("renderer_unstable", "renderer is not stable", 409, stable);
    }
    const canvas = session.page.locator(".cesium-widget canvas").first();
    await canvas.waitFor({ state: "visible", timeout: this.timeoutMs });
    const first = await canvas.screenshot({ animations: "disabled", type: "png" });
    await session.page.evaluate(
      (timeoutMs) => window.__EO_HARNESS_V2__.waitForStableFrame({ timeout_ms: timeoutMs }),
      this.timeoutMs
    );
    const second = await canvas.screenshot({ animations: "disabled", type: "png" });
    const firstHash = sha256(first);
    const secondHash = sha256(second);
    if (firstHash !== secondHash) {
      throw new RendererError(
        "nondeterministic_capture",
        "consecutive stable captures produced different SHA-256 values",
        409,
        { first_sha256: firstHash, second_sha256: secondHash }
      );
    }
    const pixelStats = analyzePng(second);
    assertCaptureQuality(pixelStats, this.viewport);
    session.lastReadback = stable;
    return {
      content: second,
      pixelStats,
      readback: stable,
      sha256: secondHash,
      sizeBytes: second.length,
      versions: { ...stable.versions, ...this.versions() }
    };
  }

  inspect(episodeId) {
    const session = this.get(episodeId);
    return {
      bridge: session.bridge,
      created_at: session.createdAt,
      episode_id: episodeId,
      last_readback: session.lastReadback,
      versions: this.versions(),
      viewport: this.viewport
    };
  }

  async close(episodeId) {
    const session = this.get(episodeId);
    this.sessions.delete(episodeId);
    await session.context.close();
  }
}
