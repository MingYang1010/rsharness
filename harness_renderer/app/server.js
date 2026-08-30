import http from "node:http";

import { RendererError, readJsonBody, validateEpisodeId } from "./schemas.js";
import { SessionManager } from "./sessions.js";

const port = Number(process.env.EO_RENDERER_PORT || 8090);
const manager = new SessionManager({
  terriaUrl: process.env.EO_RENDERER_TERRIA_URL || "http://terriamap:3001/?lng=en",
  timeoutMs: Number(process.env.EO_RENDERER_TIMEOUT_MS || 45000),
  viewport: {
    height: Number(process.env.EO_RENDERER_VIEWPORT_HEIGHT || 768),
    width: Number(process.env.EO_RENDERER_VIEWPORT_WIDTH || 1024)
  }
});

function sendJson(response, statusCode, value) {
  const body = Buffer.from(JSON.stringify(value));
  response.writeHead(statusCode, {
    "Content-Length": String(body.length),
    "Content-Type": "application/json; charset=utf-8"
  });
  response.end(body);
}

function routePath(pathname) {
  const match = pathname.match(/^\/renderer\/sessions\/(ep2-[a-f0-9]{32})(?:\/(state|capture))?$/);
  if (!match) return null;
  return { episodeId: match[1], operation: match[2] || "session" };
}

const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, `http://${request.headers.host || "renderer"}`);
    if (request.method === "GET" && url.pathname === "/healthz") {
      sendJson(response, 200, {
        status: "ok",
        sessions: manager.sessions.size,
        versions: manager.versions()
      });
      return;
    }
    if (request.method === "POST" && url.pathname === "/renderer/sessions") {
      const body = await readJsonBody(request);
      const episodeId = validateEpisodeId(body.episode_id);
      const data = await manager.create(episodeId);
      sendJson(response, 201, { data });
      return;
    }

    const route = routePath(url.pathname);
    if (!route) throw new RendererError("not_found", "renderer route not found", 404);
    if (request.method === "PUT" && route.operation === "state") {
      const body = await readJsonBody(request);
      const data = await manager.apply(route.episodeId, body.map_state);
      sendJson(response, 200, { data });
      return;
    }
    if (request.method === "GET" && route.operation === "state") {
      sendJson(response, 200, { data: await manager.read(route.episodeId) });
      return;
    }
    if (request.method === "POST" && route.operation === "capture") {
      const capture = await manager.capture(route.episodeId);
      sendJson(response, 200, {
        data: {
          content_base64: capture.content.toString("base64"),
          pixel_stats: capture.pixelStats,
          readback: capture.readback,
          sha256: capture.sha256,
          size_bytes: capture.sizeBytes,
          versions: capture.versions
        }
      });
      return;
    }
    if (request.method === "GET" && route.operation === "session") {
      sendJson(response, 200, { data: manager.inspect(route.episodeId) });
      return;
    }
    if (request.method === "DELETE" && route.operation === "session") {
      await manager.close(route.episodeId);
      sendJson(response, 200, { data: { closed: true, episode_id: route.episodeId } });
      return;
    }
    throw new RendererError("method_not_allowed", "renderer method is not allowed", 405);
  } catch (error) {
    const known = error instanceof RendererError;
    const statusCode = known ? error.statusCode : 500;
    if (!known) console.error(error);
    sendJson(response, statusCode, {
      error: {
        code: known ? error.code : "renderer_internal_error",
        details: known ? error.details : {},
        message: known ? error.message : "renderer could not complete the request"
      }
    });
  }
});

async function shutdown() {
  server.close();
  await manager.stop();
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

await manager.start();
server.listen(port, "0.0.0.0", () => {
  console.log(`EO Harness renderer listening on ${port}`);
});
