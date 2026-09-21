#!/usr/bin/env python3
"""Trusted Qwen runner for the scoped Agent gateway; no backend/task mounts."""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI


SYSTEM_PROMPT = """You interact with an EO Harness through tools.
Use a real tool when it helps; do not fabricate tool output. For crop tasks,
first call eo_gym.crop. Its aoi is always normalized [x0,y0,x1,y1], with
0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1; never send pixel coordinates.
After receiving an image artifact, call
memory.save_evidence with a complete evidence object: use the exact artifact_id
as source_ref, the exact sha256 as frozen_sha256, a selector that constrains the
artifact, and a stable claim_id. Then call answer.submit with valid JSON matching
the task schema and cite the saved evidence_id."""

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
REPORT_SCHEMA_VERSION = "qwen-agent-runner-v2"


def openai_tools(session: dict) -> list[dict]:
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
        tools.append({"type": "function", "function": {"name": name, "description": "EO Harness tool " + name, "parameters": schema}})
    task = session.get("task", {})
    allowed_actions = set(task.get("allowed_actions", []))
    action_schemas = {
        "memory.save_evidence": {
            "type": "object",
            "properties": {
                "evidence_id": {"type": "string", "pattern": "^ev-[A-Za-z0-9][A-Za-z0-9._:-]{0,123}$"},
                "claim_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._:-]*$", "maxLength": 200},
                "source_ref": {"type": "string", "pattern": "^art-[a-f0-9]{64}$"},
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
                        "pixel_window": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                    },
                    "anyOf": [
                        {"required": ["bbox"]}, {"required": ["time_range"]},
                        {"required": ["bands"]}, {"required": ["pixel_window"]},
                    ],
                    "additionalProperties": False,
                },
                "description": {"type": "string", "minLength": 1},
                "frozen_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            },
            "required": ["evidence_id", "claim_id", "source_ref", "selector", "description", "frozen_sha256"],
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
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["rationale"],
            "additionalProperties": False,
        },
        "answer.request_human_review": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["rationale"],
            "additionalProperties": False,
        },
    }
    for name, schema in action_schemas.items():
        if name in allowed_actions:
            tools.append({"type": "function", "function": {
                "name": name, "description": "EO Harness action " + name, "parameters": schema,
            }})
    return tools


def action_from_tool_call(session: dict, name: str, arguments: dict) -> dict:
    if name in session.get("tool_schemas", {}):
        return {"type": "tool.invoke", "tool_id": name, "arguments": arguments}
    if name == "memory.save_evidence" and name in set(session.get("task", {}).get("allowed_actions", [])):
        return {"type": name, "evidence": arguments}
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


def artifact_message(artifact: dict, content: bytes, media_type: str | None = None) -> dict:
    public = {key: artifact[key] for key in (
        "artifact_id", "sha256", "size_bytes", "media_type", "pixel", "spatial", "temporal"
    ) if key in artifact}
    return content_message(
        "Verified artifact metadata: " + json.dumps(public, ensure_ascii=False, sort_keys=True)
        + ". Use artifact_id as source_ref and sha256 as frozen_sha256.",
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


def error_report(exc: BaseException, *, episode_id: str | None, turns: int, tool_calls: int,
                 image_hashes: list[str], elapsed_ms: float, transcript: list[dict],
                 checkpoint: dict | None = None) -> dict:
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
            "checkpoint": checkpoint}


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


def model_client(base_url: str):
    """Build a model client that never routes local serving through a proxy."""
    return OpenAI(
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

        def attach_artifact(artifact_id: str) -> None:
            metadata = request("GET", "/agent/artifacts/" + artifact_id)["artifact"]
            content_response = client.get("/agent/artifacts/" + artifact_id + "/content")
            content_response.raise_for_status()
            content = content_response.content
            if len(content) != metadata["size_bytes"]:
                raise ValueError("gateway artifact size mismatch")
            image_hashes.add(metadata["sha256"])
            messages.append(artifact_message(
                metadata, content, content_response.headers.get("content-type"),
            ))

        if checkpoint_value and state.get("status") != "terminal":
            for artifact_id in artifact_ids:
                attach_artifact(artifact_id)
        terminated = False
        for turn in range(max_turns):
            if resume and state.get("status") == "terminated":
                return {"status": "passed", "reason": None, "episode_id": state["episode_id"],
                        "turns": turn, "model_tool_calls": tool_calls,
                        "image_hashes": sorted(image_hashes), "elapsed_ms": round((time.time() - started) * 1000, 3),
                        "transcript": transcript, "resumed": True}
            response = model_client.chat.completions.create(
                model="Qwen3.5-9B",
                messages=messages,
                tools=openai_tools(session),
                tool_choice="auto",
                temperature=0.0,
                max_tokens=4096,
            )
            assistant, calls = tool_call_message(response)
            messages.append(assistant)
            transcript.append({"turn": turn, "assistant": assistant, "model_response": response_metadata(response)})
            if not calls:
                # A final answer without the required evidence/submit tools is a model failure.
                return {"status": "failed", "reason": "model_returned_text_without_action", "episode_id": state["episode_id"],
                        "turns": turn + 1, "model_tool_calls": tool_calls, "image_hashes": sorted(image_hashes),
                        "elapsed_ms": round((time.time() - started) * 1000, 3), "transcript": transcript}
            for call in calls:
                name = tool_function(call)["name"]
                arguments = decode_tool_arguments(call)
                action_id = "qwen-" + state["episode_id"][4:12] + "-" + str(tool_calls)
                body = {"client_action_id": action_id, "expected_state_version": state["state_version"],
                        "action": action_from_tool_call(session, name, arguments)}
                result = request("POST", "/agent/step", body)
                state = result["state"]
                terminated = result.get("terminated", False)
                tool_calls += 1
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps({"observation": result["observation"], "terminated": terminated}, ensure_ascii=False, sort_keys=True)})
                transcript.append({"turn": turn, "tool_call_id": call.id, "name": name, "arguments": arguments, "state_version": state["state_version"]})
                for artifact_id in artifact_refs(result.get("observation", {})):
                    if artifact_id not in artifact_ids:
                        artifact_ids.append(artifact_id)
                    attach_artifact(artifact_id)
                if checkpoint_path is not None:
                    checkpoint_value = {
                        "schema_version": "qwen-agent-checkpoint-v1",
                        "session": session, "messages": compact_messages(messages), "state": state,
                        "terminated": terminated, "tool_calls": tool_calls,
                        "artifact_ids": artifact_ids, "image_hashes": sorted(image_hashes),
                        "transcript": transcript,
                    }
                    save_checkpoint(checkpoint_path, checkpoint_value)
                if terminated:
                    return {"status": "passed" if state.get("status") == "terminated" else "failed",
                            "reason": None if state.get("status") == "terminated" else "nonterminal_after_answer",
                            "episode_id": state["episode_id"], "turns": turn + 1, "model_tool_calls": tool_calls,
                            "image_hashes": sorted(image_hashes), "elapsed_ms": round((time.time() - started) * 1000, 3),
                            "transcript": transcript}
        return {"status": "failed", "reason": "max_turns_reached", "episode_id": state["episode_id"],
                "turns": max_turns, "model_tool_calls": tool_calls, "image_hashes": sorted(image_hashes),
                "elapsed_ms": round((time.time() - started) * 1000, 3), "transcript": transcript}


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
        report = run(args.gateway, args.token_file.read_text().strip(), client, args.max_turns, args.checkpoint)
    except BaseException as exc:
        report = error_report(exc, episode_id=None, turns=0, tool_calls=0, image_hashes=[],
                              elapsed_ms=0, transcript=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "reason", "episode_id", "turns", "model_tool_calls", "image_hashes")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
