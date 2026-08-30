const EPISODE_ID = /^ep2-[a-f0-9]{32}$/;

export class RendererError extends Error {
  constructor(code, message, statusCode = 422, details = {}) {
    super(message);
    this.name = "RendererError";
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
  }
}

export function validateEpisodeId(value) {
  if (typeof value !== "string" || !EPISODE_ID.test(value)) {
    throw new RendererError("invalid_episode_id", "episode_id is invalid");
  }
  return value;
}

function finite(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RendererError("invalid_map_state", `${name} must be finite`);
  }
}

export function validateMapState(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new RendererError("invalid_map_state", "map_state must be an object");
  }
  const bbox = value.bbox;
  if (!bbox || typeof bbox !== "object") {
    throw new RendererError("invalid_map_state", "map_state.bbox is required");
  }
  for (const key of ["west", "south", "east", "north"]) {
    finite(bbox[key], `map_state.bbox.${key}`);
  }
  if (
    bbox.west < -180 || bbox.east > 180 || bbox.south < -90 || bbox.north > 90 ||
    bbox.west >= bbox.east || bbox.south >= bbox.north
  ) {
    throw new RendererError("invalid_map_state", "map_state.bbox is outside WGS84 bounds");
  }
  if (!value.layers || typeof value.layers !== "object" || Array.isArray(value.layers)) {
    throw new RendererError("invalid_map_state", "map_state.layers must be an object");
  }
  for (const [layerId, layer] of Object.entries(value.layers)) {
    if (!layer || layer.layer_id !== layerId || typeof layer.asset_id !== "string") {
      throw new RendererError("invalid_map_state", `invalid layer: ${layerId}`);
    }
    if (typeof layer.visible !== "boolean") {
      throw new RendererError("invalid_map_state", `layer ${layerId} visible must be boolean`);
    }
    finite(layer.opacity, `layer ${layerId} opacity`);
    if (layer.opacity < 0 || layer.opacity > 1) {
      throw new RendererError("invalid_map_state", `layer ${layerId} opacity is outside 0-1`);
    }
  }
  return structuredClone(value);
}

export async function readJsonBody(request, maxBytes = 2 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBytes) {
      throw new RendererError("request_too_large", "request body is too large", 413);
    }
    chunks.push(chunk);
  }
  if (size === 0) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new RendererError("invalid_json", "request body is not valid JSON", 400);
  }
}
