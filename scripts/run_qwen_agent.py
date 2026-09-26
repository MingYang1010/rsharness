#!/usr/bin/env python3
"""Trusted Qwen runner for the scoped Agent gateway; no backend/task mounts."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from app.core.events import sha256_json

import httpx

MODEL_NAME = "Qwen3.5-9B"

SYSTEM_PROMPT = """You interact with an EO Harness through tools.
Use a real tool when it helps; do not fabricate tool output. For crop tasks,
first call eo_gym.crop. Its aoi is always normalized [x0,y0,x1,y1], with
0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1; never send pixel coordinates.
After receiving an image artifact, call
memory.save_evidence with its artifact_index, a selector that constrains the
artifact, and a stable claim_id. The Harness binds the index to the exact
verified artifact_id and sha256; never copy or invent those long identifiers.
A pixel_window is [x,y,width,height], not
[x0,y0,x1,y1]; for the full image use [0,0,pixel.width,pixel.height] from the
verified artifact metadata. Then call answer.submit with valid JSON matching the
task schema and cite the saved evidence_id. For tasks with multiple input
assets, inspect every required asset before submitting and save separate
evidence for each artifact. For rendered map tasks, call map.set_view with the
fixed task AOI, then save evidence from the verified rendered artifact.

Identifiers have distinct roles. An observation_id identifies a result message
and is never valid evidence. An evidence_id is returned only by
memory.save_evidence and is the sole identifier accepted by answer.submit,
answer.abstain, and answer.request_human_review. If public metadata shows that
a required input is unavailable, inspect that input first, then call
answer.abstain with an empty evidence_ids array and state the metadata reason
in rationale; do not cite observation IDs or fabricate evidence IDs."""
SYSTEM_PROMPT += """

Never finish by writing plain text. After tool results, choose one available
terminal action: answer.submit, answer.abstain, or, when exposed,
answer.request_human_review. If governed records conflict, select the best
supported record, abstain, or request review—always through that action."""

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
REPORT_SCHEMA_VERSION = "qwen-agent-runner-v2"


def openai_tools(session: dict, artifacts: list[dict] | None = None) -> list[dict]:
    artifacts = artifacts or []
    tools = []
    for name, schema in sorted(session.get("tool_schemas", {}).items()):
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError("tool schema is not an object")
        schema = json.loads(json.dumps(schema))
        if name == "eo_gym.crop":
            aoi = schema.get("properties", {}).get("aoi")
            if not isinstance(aoi, dict):
                raise ValueError("EO-Gym crop schema lacks normalized aoi")
            aoi["items"] = {"type": "number", "minimum": 0.0, "maximum": 1.0}
            aoi["minItems"] = 4
            aoi["maxItems"] = 4
            aoi["description"] = (
                "Normalized [x0,y0,x1,y1], requiring 0 <= x0 < x1 <= 1 "
                "and 0 <= y0 < y1 <= 1; these are not pixel coordinates."
            )
        if name == "memory.search":
            for field_name in ("bbox", "time_range"):
                schema["properties"][field_name]["description"] = (
                    "Pass this as a JSON object, not a JSON-encoded string."
                )
        tools.append({"type": "function", "function": {"name": name, "description": "EO Harness tool " + name, "parameters": schema}})
    task = session.get("task", {})
    allowed_actions = set(task.get("allowed_actions", []))
    action_schemas = {
        "memory.save_evidence": {
            "type": "object",
            "properties": {
                "evidence_id": {"type": "string", "pattern": "^ev-[A-Za-z0-9][A-Za-z0-9._:-]{0,123}$"},
                "claim_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._:-]*$", "maxLength": 200},
                "artifact_index": {
                    "type": "integer",
                    "enum": list(range(len(artifacts))),
                    "description": "Index of a verified artifact shown by the Harness.",
                },
                "selector": {
                    "type": "object",
                    "properties": {
                        "bbox": {
                            "type": "object",
                            "properties": {
                                "west": {"type": "number", "minimum": -180, "maximum": 180},
                                "south": {"type": "number", "minimum": -90, "maximum": 90},
                                "east": {"type": "number", "minimum": -180, "maximum": 180},
                                "north": {"type": "number", "minimum": -90, "maximum": 90},
                            },
                            "required": ["west", "south", "east", "north"],
                            "additionalProperties": False,
                        },
                        "time_range": {
                            "type": "object",
                            "properties": {"start": {"type": "string"}, "end": {"type": "string"}},
                            "required": ["start", "end"],
                            "additionalProperties": False,
                        },
                        "bands": {"type": "array", "items": {"type": "string"}},
                        "pixel_window": {
                            "type": "array", "items": {"type": "integer", "minimum": 0},
                            "minItems": 4, "maxItems": 4,
                            "description": (
                                "[x,y,width,height], not corner coordinates. Use "
                                "[0,0,pixel.width,pixel.height] for the full verified artifact."
                            ),
                        },
                    },
                    "anyOf": [
                        {"required": ["bbox"]}, {"required": ["time_range"]},
                        {"required": ["bands"]}, {"required": ["pixel_window"]},
                    ],
                    "additionalProperties": False,
                },
                "description": {"type": "string", "minLength": 1},
            },
            "required": ["evidence_id", "claim_id", "artifact_index", "selector", "description"],
            "additionalProperties": False,
        },
        "answer.submit": {
            "type": "object",
            "properties": {
                "answer": task.get("answer_schema", {"type": "object"}),
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "evidence_ids"],
            "additionalProperties": False,
        },
        "answer.abstain": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                "description": (
                    "For this memory benchmark, send an empty array. Only IDs "
                    "returned by memory.save_evidence are valid; never send "
                    "observation IDs or memory IDs."
                ),
                },
            },
            "required": ["rationale"],
            "additionalProperties": False,
        },
        "answer.request_human_review": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "For this memory benchmark, send an empty array. Only "
                        "IDs returned by memory.save_evidence are valid; never "
                        "send observation IDs or memory IDs."
                    ),
                },
            },
            "required": ["rationale"],
            "additionalProperties": False,
        },
        "map.set_view": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "object",
                    "properties": {
                        "west": {"type": "number", "minimum": -180, "maximum": 180},
                        "south": {"type": "number", "minimum": -90, "maximum": 90},
                        "east": {"type": "number", "minimum": -180, "maximum": 180},
                        "north": {"type": "number", "minimum": -90, "maximum": 90},
                    },
                    "required": ["west", "south", "east", "north"],
                    "additionalProperties": False,
                },
            },
            "required": ["bbox"],
            "additionalProperties": False,
        },
    }
    if "eo_gym.crop" in session.get("tool_schemas", {}):
        evidence_schema = action_schemas["memory.save_evidence"]
        selector = evidence_schema["properties"]["selector"]
        pixel_window = selector["properties"]["pixel_window"]
        del evidence_schema["properties"]["selector"]
        evidence_schema["properties"]["pixel_window"] = pixel_window
        evidence_schema["required"] = [
            "pixel_window" if key == "selector" else key
            for key in evidence_schema["required"]
        ]
    for name, schema in action_schemas.items():
        if name == "memory.save_evidence" and not artifacts:
            continue
        if name in allowed_actions:
            tools.append({"type": "function", "function": {
                "name": name, "description": "EO Harness action " + name, "parameters": schema,
            }})
    return tools


def action_from_tool_call(session: dict, name: str, arguments: dict,
                          artifacts: list[dict] | None = None) -> dict:
    if name in session.get("tool_schemas", {}):
        return {"type": "tool.invoke", "tool_id": name, "arguments": arguments}
    if name == "memory.save_evidence" and name in set(session.get("task", {}).get("allowed_actions", [])):
        artifacts = artifacts or []
        artifact_index = arguments.get("artifact_index")
        if isinstance(artifact_index, bool) or not isinstance(artifact_index, int):
            raise ValueError("evidence artifact_index must be an integer")
        if artifact_index < 0 or artifact_index >= len(artifacts):
            raise ValueError("evidence artifact_index is unavailable")
        artifact = artifacts[artifact_index]
        artifact_id = artifact.get("artifact_id")
        digest = artifact.get("sha256")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("verified artifact id is unavailable")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("verified artifact hash is unavailable")
        evidence = {key: value for key, value in arguments.items() if key != "artifact_index"}
        if "eo_gym.crop" in session.get("tool_schemas", {}):
            pixel_window = evidence.pop("pixel_window", None)
            if (not isinstance(pixel_window, list) or len(pixel_window) != 4
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           for value in pixel_window)):
                raise ValueError("evidence pixel_window must be four integers")
            x, y, width, height = pixel_window
            pixel = artifact.get("pixel", {})
            artifact_width = pixel.get("width")
            artifact_height = pixel.get("height")
            if (x < 0 or y < 0 or width <= 0 or height <= 0
                    or not isinstance(artifact_width, int)
                    or not isinstance(artifact_height, int)
                    or x + width > artifact_width or y + height > artifact_height):
                raise ValueError("evidence pixel_window exceeds verified artifact bounds")
            evidence["selector"] = {"pixel_window": pixel_window}
        evidence["source_ref"] = artifact_id
        evidence["frozen_sha256"] = digest
        return {"type": name, "evidence": evidence}
    if name in set(session.get("task", {}).get("allowed_actions", [])):
        return {"type": name, **arguments}
    raise ValueError("model requested an unavailable Harness action")


def content_message(text: str, artifact=None, content: bytes | None = None, media_type: str | None = None) -> dict:
    message: dict[str, Any] = {"role": "user", "content": [{"type": "text", "text": text}]}
    if artifact is not None and content is not None:
        if len(content) > MAX_IMAGE_BYTES:
            raise ValueError("artifact image exceeds model input bound")
        if len(content) != artifact.get("size_bytes"):
            raise ValueError("artifact content size mismatch")
        digest = artifact.get("sha256")
        if not digest or len(digest) != 64:
            raise ValueError("artifact hash unavailable")
        data_url = "data:" + (media_type or artifact.get("media_type", "application/octet-stream")) + ";base64," + base64.b64encode(content).decode("ascii")
        message["content"].append({"type": "image_url", "image_url": {"url": data_url}})
    return message


def artifact_message(artifact: dict, content: bytes, artifact_index: int,
                     media_type: str | None = None) -> dict:
    public = {key: artifact[key] for key in (
        "artifact_id", "sha256", "size_bytes", "media_type", "pixel", "spatial", "temporal"
    ) if key in artifact}
    public["artifact_index"] = artifact_index
    return content_message(
        "Verified artifact metadata: " + json.dumps(public, ensure_ascii=False, sort_keys=True)
        + ". Use artifact_index when saving evidence; the Harness binds exact identifiers.",
        artifact, content, media_type,
    )


def artifact_refs(value: Any) -> list[str]:
    refs = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "artifact_ref" and isinstance(item, str):
                refs.append(item)
            refs.extend(artifact_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(artifact_refs(item))
    return refs


def tool_call_message(response: Any) -> tuple[dict, list[Any]]:
    choice = response.choices[0]
    assistant = {"role": "assistant", "content": choice.message.content or ""}
    calls = list(choice.message.tool_calls or [])
    if calls:
        assistant["tool_calls"] = [
            {"id": call.id, "type": "function", "function": tool_function(call)}
            for call in calls
        ]
    return assistant, calls


def tool_function(call: Any) -> dict:
    if hasattr(call, "function"):
        function = call.function
        if hasattr(function, "name"):
            return {"name": function.name, "arguments": function.arguments}
        return {"name": function["name"], "arguments": function["arguments"]}
    return {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}


def decode_tool_arguments(call: Any) -> dict:
    raw = tool_function(call)["arguments"]
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be an object")
    for key in ("bbox", "time_range"):
        nested = value.get(key)
        if isinstance(nested, str):
            try:
                value[key] = json.loads(nested)
            except json.JSONDecodeError:
                pass
        # Qwen sometimes serializes explicitly optional fields as empty strings.
        for key in ("platform", "instrument"):
            if value.get(key) == "":
                value[key] = None
    return value


def response_metadata(response: Any) -> dict:
    return {
        "id": getattr(response, "id", None),
        "created": getattr(response, "created", None),
        "model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "usage": getattr(response, "usage", None).model_dump(mode="json")
            if getattr(response, "usage", None) is not None else None,
    }


def provider_response(response: Any) -> dict:
    """Preserve the provider JSON response before trusted adapter transforms."""
    if hasattr(response, "model_dump"):
        value = response.model_dump(mode="json")
    elif hasattr(response, "model_dump_json"):
        value = json.loads(response.model_dump_json())
    elif isinstance(response, SimpleNamespace):
        value = {
            "id": getattr(response, "id", None),
            "created": getattr(response, "created", None),
            "model": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "choices": [
                {
                    "message": {
                        "content": getattr(choice.message, "content", None),
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": getattr(call, "type", "function"),
                                "function": {
                                    "name": tool_function(call)["name"],
                                    "arguments": tool_function(call)["arguments"],
                                },
                            }
                            for call in (getattr(choice.message, "tool_calls", None) or [])
                        ],
                    },
                }
                for choice in getattr(response, "choices", [])
            ],
            "usage": response.usage.model_dump(mode="json")
            if getattr(response, "usage", None) is not None else None,
        }
    else:
        raise TypeError("provider response has no JSON projection")
    return json.loads(json.dumps(value, ensure_ascii=False))


def adapter_projection(provider: dict, assistant: dict) -> dict:
    """Fingerprint the stable adapter representation used by the runner."""
    return {
        "schema_version": "qwen-response-adapter-projection-v1",
        "provider_response_sha256": sha256_json(provider),
        "assistant": assistant,
    }


def model_request_receipt(messages: list[dict], tools: list[dict]) -> dict:
    """Project a model request without retaining base64 image contents."""
    image_hashes = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            url = item.get("image_url", {}).get("url") if isinstance(item, dict) else None
            if isinstance(url, str) and url.startswith("data:"):
                header, separator, payload = url.partition(",")
                if separator:
                    digest = hashlib.sha256(base64.b64decode(payload, validate=True)).hexdigest()
                    image_hashes.append({
                        "media_type": header[5:].rsplit(";", 1)[0],
                        "payload_sha256": digest,
                    })
    return {
        "schema_version": "qwen-model-request-receipt-v1",
        "model": MODEL_NAME,
        "temperature": 0.0,
        "max_tokens": 4096,
        "message_count": len(messages),
        "tool_count": len(tools),
        "image_hashes": image_hashes,
    }


def usage_cost(usage: Any) -> dict:
    value = usage.model_dump(mode="json") if usage is not None else None
    if not isinstance(value, dict):
        value = {}
    return {
        "model_calls": 1,
        "prompt_tokens": int(value.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(value.get("completion_tokens", 0) or 0),
        "total_tokens": int(value.get("total_tokens", 0) or 0),
    }


def add_cost(total: dict, addition: dict) -> dict:
    return {
        key: int(total.get(key, 0) or 0) + int(addition.get(key, 0) or 0)
        for key in ("model_calls", "prompt_tokens", "completion_tokens", "total_tokens")
    }


def transcript_cost(transcript: list[dict]) -> dict:
    total = {"model_calls": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "total_tokens": 0}
    for item in transcript:
        usage = item.get("model_response", {}).get("usage")
        if isinstance(usage, dict):
            total = add_cost(total, usage_cost(SimpleNamespace(model_dump=lambda mode="json": usage)))
    return total


def error_report(exc: BaseException, *, episode_id: str | None, turns: int, tool_calls: int,
                 image_hashes: list[str], elapsed_ms: float, transcript: list[dict],
                 checkpoint: dict | None = None, cumulative_cost: dict | None = None) -> dict:
    if isinstance(exc, httpx.HTTPError):
        reason = "gateway_network_error"
    elif isinstance(exc, (ValueError, json.JSONDecodeError)):
        reason = "runner_contract_error"
    else:
        reason = type(exc).__name__
    return {"schema_version": REPORT_SCHEMA_VERSION, "status": "failed", "reason": reason,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "episode_id": episode_id, "turns": turns, "model_tool_calls": tool_calls,
            "image_hashes": image_hashes, "elapsed_ms": elapsed_ms, "transcript": transcript,
            "checkpoint": checkpoint,
            "cost": cumulative_cost or transcript_cost(transcript),
            "attempt": {"phase": "failed", "resumed": bool(checkpoint)}}


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("runner checkpoint is unavailable or oversized")
    value = json.loads(path.read_text())
    if value.get("schema_version") != "qwen-agent-checkpoint-v1":
        raise ValueError("unsupported runner checkpoint schema")
    return value


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(checkpoint, ensure_ascii=False, sort_keys=True) + "\n"
    if len(content.encode()) > MAX_JSON_BYTES:
        raise ValueError("runner checkpoint exceeds bound")
    temporary.write_text(content)
    temporary.replace(path)


def compact_messages(messages: list[dict]) -> list[dict]:
    compact = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(item, dict) and item.get("type") == "image_url"
            for item in content
        ):
            continue
        compact.append(message)
    return compact


def model_client(base_url: str, openai_client_class=None):
    """Build a model client that never routes local serving through a proxy."""
    if openai_client_class is None:
        from openai import OpenAI

        openai_client_class = OpenAI

    return openai_client_class(
        base_url=base_url,
        api_key="local",
        timeout=300.0,
        http_client=httpx.Client(trust_env=False, timeout=300.0),
    )


def run(gateway_url: str, token: str, model_client, max_turns: int = 12,
        checkpoint_path: Path | None = None) -> dict:
    started = time.time()
    transcript = []
    image_hashes = set()
    tool_calls = 0
    cumulative_cost = {"model_calls": 0, "prompt_tokens": 0,
                       "completion_tokens": 0, "total_tokens": 0}
    checkpoint_value = load_checkpoint(checkpoint_path) if checkpoint_path else None
    resume = bool(checkpoint_value)
    with httpx.Client(base_url=gateway_url, timeout=45, trust_env=False,
                      headers={"Authorization": "Bearer " + token}) as client:
        def request(method: str, path: str, body=None):
            response = client.request(method, path, json=body)
            response.raise_for_status()
            return response.json()

        if checkpoint_value:
            session = checkpoint_value["session"]
        else:
            session = request("GET", "/agent/session")
        session_json = json.dumps(session, ensure_ascii=False, sort_keys=True)
        if len(session_json.encode()) > MAX_JSON_BYTES:
            raise ValueError("agent session exceeds model input bound")
        if checkpoint_value:
            messages = checkpoint_value["messages"]
            state = checkpoint_value["state"]
            image_hashes.update(checkpoint_value.get("image_hashes", []))
            artifact_ids = list(checkpoint_value.get("artifact_ids", []))
            transcript.extend(checkpoint_value.get("transcript", []))
            tool_calls = int(checkpoint_value.get("tool_calls", 0))
            cumulative_cost.update(checkpoint_value.get("cost", cumulative_cost))
            pending_followup_calls = checkpoint_value.get(
                "pending_followup_calls", []
            )
            restored_followups = []
            for item in pending_followup_calls:
                if not isinstance(item, SimpleNamespace):
                    function = item.get("function", {})
                    item = SimpleNamespace(
                        id=item.get("id"),
                        function=SimpleNamespace(
                            name=function.get("name"),
                            arguments=function.get("arguments"),
                        ),
                    )
                restored_followups.append(item)
            pending_followup_calls = restored_followups
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "task": session.get("task", {}),
                    "state": session.get("state", {}),
                    "observation": session.get("observation", {}),
                }, ensure_ascii=False, sort_keys=True)},
            ]
            state = session["state"]
            artifact_ids = []
        artifacts = []

        def attach_artifact(artifact_id: str) -> None:
            metadata = request("GET", "/agent/artifacts/" + artifact_id)["artifact"]
            content_response = client.get("/agent/artifacts/" + artifact_id + "/content")
            content_response.raise_for_status()
            content = content_response.content
            received_hash = hashlib.sha256(content).hexdigest()
            if len(content) != metadata["size_bytes"]:
                raise ValueError("gateway artifact size mismatch")
            if received_hash != metadata["sha256"]:
                raise ValueError("gateway artifact checksum mismatch")
            image_hashes.add(metadata["sha256"])
            artifact_index = len(artifacts)
            artifacts.append(metadata)
            messages.append(artifact_message(
                metadata, content, artifact_index, content_response.headers.get("content-type"),
            ))

        if checkpoint_value and state.get("status") != "terminated":
            for artifact_id in artifact_ids:
                attach_artifact(artifact_id)
        terminated = False
        pending = checkpoint_value.get("pending_action") if checkpoint_value else None
        resume_pending = pending is not None
        if not checkpoint_value:
            pending_followup_calls = []
        for turn in range(max_turns):
            if resume and state.get("status") == "terminated":
                return {"status": "passed", "reason": None, "episode_id": state["episode_id"],
                        "turns": turn, "model_tool_calls": tool_calls,
                        "image_hashes": sorted(image_hashes), "elapsed_ms": round((time.time() - started) * 1000, 3),
                        "transcript": transcript, "resumed": True,
                        "cost": cumulative_cost,
                        "attempt": {"phase": "resume_terminal", "resumed": True,
                                    "new_model_calls": 0}}
            if resume_pending:
                calls = pending_followup_calls
            else:
                request_tools = openai_tools(session, artifacts)
                receipt = model_request_receipt(messages, request_tools)
                response = model_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=request_tools,
                    tool_choice="auto",
                    temperature=0.0,
                    max_tokens=4096,
                )
                assistant, calls = tool_call_message(response)
                provider = provider_response(response)
                cumulative_cost = add_cost(cumulative_cost, usage_cost(getattr(response, "usage", None)))
                messages.append(assistant)
                transcript.append({
                    "turn": turn,
                    "assistant": assistant,
                    "provider_response": provider,
                    "adapter_projection": adapter_projection(provider, assistant),
                    "model_request": receipt,
                    "model_response": response_metadata(response),
                })
            if not calls and not resume_pending:
                # A final answer without the required evidence/submit tools is a model failure.
                return {"status": "failed", "reason": "model_returned_text_without_action", "episode_id": state["episode_id"],
                        "turns": turn + 1, "model_tool_calls": tool_calls, "image_hashes": sorted(image_hashes),
                        "elapsed_ms": round((time.time() - started) * 1000, 3), "transcript": transcript,
                        "cost": cumulative_cost,
                        "attempt": {"phase": "failed", "resumed": resume, "new_model_calls": 1}}
            if pending is not None:
                active_calls = [SimpleNamespace(id=pending["tool_call_id"])]
            else:
                active_calls = list(calls)
            resume_pending = False
            for call in active_calls:
                if pending is None:
                    name = tool_function(call)["name"]
                    arguments = decode_tool_arguments(call)
                    action_id = "qwen-" + state["episode_id"][4:12] + "-" + str(tool_calls)
                    body = {
                        "client_action_id": action_id,
                        "expected_state_version": state["state_version"],
                        "action": action_from_tool_call(
                            session, name, arguments, artifacts
                        ),
                    }
                    pending = {"body": body, "tool_call_id": call.id,
                               "name": name, "arguments": arguments}
                    pending["calls_remaining"] = len(calls) - calls.index(call) - 1
                    pending_followup_calls = list(calls[calls.index(call) + 1:])
                    if checkpoint_path is not None:
                        checkpoint_value = {
                            "schema_version": "qwen-agent-checkpoint-v1",
                            "session": session,
                            "messages": compact_messages(messages),
                            "state": state,
                            "terminated": terminated,
                            "tool_calls": tool_calls,
                            "artifact_ids": artifact_ids,
                            "image_hashes": sorted(image_hashes),
                            "transcript": transcript,
                            "cost": cumulative_cost,
                            "pending_followup_calls": [
                                {"id": item.id, **getattr(item, "__dict__", {})}
                                for item in pending_followup_calls
                            ],
                            "pending_action": pending,
                        }
                        save_checkpoint(checkpoint_path, checkpoint_value)
                else:
                    call = SimpleNamespace(id=pending["tool_call_id"])
                body = pending["body"]
                try:
                    result = request("POST", "/agent/step", body)
                except httpx.HTTPStatusError as exc:
                    try:
                        error_body = exc.response.json().get("error", {})
                        error_detail = {
                            "code": error_body.get("code"),
                            "message": error_body.get("message"),
                            "retryable": error_body.get("retryable"),
                        }
                    except (AttributeError, ValueError):
                        error_detail = None
                    transcript[-1]["gateway_error"] = error_detail
                    return error_report(
                        exc, episode_id=state["episode_id"], turns=turn + 1,
                        tool_calls=tool_calls + 1, image_hashes=sorted(image_hashes),
                        elapsed_ms=round((time.time() - started) * 1000, 3),
                        transcript=transcript, checkpoint=checkpoint_value,
                        cumulative_cost=cumulative_cost,
                    )
                state = result["state"]
                terminated = result.get("terminated", False)
                tool_calls += 1
                pending["calls_remaining"] = max(
                    0, int(pending.get("calls_remaining", 0)) - 1
                )
                completed = {
                    "tool_call_id": call.id,
                    "name": pending["name"],
                    "arguments": pending["arguments"],
                }
                if pending["calls_remaining"] == 0:
                    pending = None
                if pending is None and pending_followup_calls:
                    following = pending_followup_calls.pop(0)
                    call = following
                    following_name = tool_function(following)["name"]
                    following_arguments = decode_tool_arguments(following)
                    action_id = "qwen-" + state["episode_id"][4:12] + "-" + str(tool_calls)
                    body = {
                        "client_action_id": action_id,
                        "expected_state_version": state["state_version"],
                        "action": action_from_tool_call(
                            session, following_name, following_arguments, artifacts
                        ),
                    }
                    pending = {
                        "body": body,
                        "tool_call_id": following.id,
                        "name": following_name,
                        "arguments": following_arguments,
                        "calls_remaining": 0,
                    }
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"observation": result["observation"], "terminated": terminated}, ensure_ascii=False, sort_keys=True)})
                transcript.append({"turn": turn, **completed,
                                   "state_version": state["state_version"]})
                for artifact_id in artifact_refs(result.get("observation", {})):
                    if artifact_id not in artifact_ids:
                        artifact_ids.append(artifact_id)
                        try:
                            attach_artifact(artifact_id)
                        except ValueError as error:
                            return error_report(
                                error, episode_id=state["episode_id"],
                                turns=turn + 1, tool_calls=tool_calls,
                                image_hashes=sorted(image_hashes),
                                elapsed_ms=round((time.time() - started) * 1000, 3),
                                transcript=transcript, checkpoint=checkpoint_value,
                                cumulative_cost=cumulative_cost,
                            )
                if checkpoint_path is not None:
                    checkpoint_value = {
                        "schema_version": "qwen-agent-checkpoint-v1",
                        "session": session, "messages": compact_messages(messages), "state": state,
                        "terminated": terminated, "tool_calls": tool_calls,
                        "artifact_ids": artifact_ids, "image_hashes": sorted(image_hashes),
                        "transcript": transcript,
                        "cost": cumulative_cost,
                    }
                    save_checkpoint(checkpoint_path, checkpoint_value)
                if terminated:
                    terminal_state = {
                        "episode_id": state["episode_id"],
                        "status": state.get("status"),
                        "final_answer": state.get("final_answer"),
                        **({"evaluation": state["evaluation"]} if state.get("evaluation") is not None else {}),
                    }
                    return {"status": "passed" if state.get("status") == "terminated" else "failed",
                            "reason": None if state.get("status") == "terminated" else "nonterminal_after_answer",
                            "episode_id": state["episode_id"], "turns": turn + 1, "model_tool_calls": tool_calls,
                            "image_hashes": sorted(image_hashes), "elapsed_ms": round((time.time() - started) * 1000, 3),
                            "transcript": transcript, "cost": cumulative_cost,
                            "terminal_state": terminal_state,
                            "attempt": {"phase": "completed", "resumed": resume,
                                        "new_model_calls": cumulative_cost["model_calls"]}}
        return {"status": "failed", "reason": "max_turns_reached", "episode_id": state["episode_id"],
                "turns": max_turns, "model_tool_calls": tool_calls, "image_hashes": sorted(image_hashes),
                "elapsed_ms": round((time.time() - started) * 1000, 3), "transcript": transcript,
                "cost": cumulative_cost,
                "attempt": {"phase": "failed", "resumed": resume,
                            "new_model_calls": cumulative_cost["model_calls"]}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://agent-gateway:8083")
    parser.add_argument("--token-file", type=Path, default=Path("/run/agent-token"))
    parser.add_argument("--openai-base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    client = model_client(args.openai_base_url)
    try:
        report = run(args.gateway, args.token_file.read_text().strip(), client,
                     args.max_turns, args.checkpoint)
    except BaseException as exc:
        report = error_report(exc, episode_id=None, turns=0, tool_calls=0, image_hashes=[],
                              elapsed_ms=0, transcript=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "reason", "episode_id", "turns", "model_tool_calls", "image_hashes")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
