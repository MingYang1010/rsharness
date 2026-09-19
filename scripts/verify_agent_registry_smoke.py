#!/usr/bin/env python3
"""Black-box two-session checks from the Agent-only network."""
import argparse
import json
import os
import re
from pathlib import Path

import httpx


def token(path: Path) -> str:
    value = path.read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RuntimeError("invalid test credential")
    return value


def request(client: httpx.Client, credential: str, method: str, path: str, body=None):
    return client.request(method, path, json=body,
                          headers={"Authorization": "Bearer " + credential})


def require(response: httpx.Response, status: int):
    if response.status_code != status:
        raise RuntimeError("unexpected gateway result: {} {}".format(
            response.status_code, response.text[:200]))
    if "application/json" in response.headers.get("content-type", ""):
        return response.json()
    return response


def artifact_ref(step: dict) -> str:
    refs = [item["artifact_ref"] for item in step["observation"]["items"]
            if item.get("artifact_ref")]
    if len(refs) != 1:
        raise RuntimeError("expected one public artifact reference")
    return refs[0]


def write_report(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["initial", "revoked-a", "rotated-a", "restarted",
                                                   "active-c", "expired-c"],
                        default="initial")
    parser.add_argument("--base-url", default="http://agent-gateway:8083")
    parser.add_argument("--token-a", type=Path, default=Path("/run/agent-token-a"))
    parser.add_argument("--token-b", type=Path, default=Path("/run/agent-token-b"))
    parser.add_argument("--token-a2", type=Path, default=Path("/run/agent-token-a2"))
    parser.add_argument("--token-c", type=Path, default=Path("/run/agent-token-c"))
    parser.add_argument("--reports", type=Path, default=Path("/reports"))
    args = parser.parse_args()
    credential_a, credential_b = token(args.token_a), token(args.token_b)
    initial_path = args.reports / "registry-initial.json"
    with httpx.Client(base_url=args.base_url, timeout=30, trust_env=False,
                      follow_redirects=False) as client:
        if args.phase in {"active-c", "expired-c"}:
            credential_c = token(args.token_c)
            response_c = request(client, credential_c, "GET", "/agent/session")
            if args.phase == "active-c":
                session_c = require(response_c, 200)
                result = {"phase": args.phase,
                          "episode_c": session_c["state"]["episode_id"]}
            else:
                if response_c.status_code != 401 or response_c.json()["error"]["code"] != "session_expired":
                    raise RuntimeError("expired token C remained active")
                result = {"phase": args.phase, "token_c_code": "session_expired"}
        elif args.phase == "initial":
            session_a = require(request(client, credential_a, "GET", "/agent/session"), 200)
            session_b = require(request(client, credential_b, "GET", "/agent/session"), 200)
            episode_a = session_a["state"]["episode_id"]
            episode_b = session_b["state"]["episode_id"]
            if episode_a == episode_b:
                raise RuntimeError("tokens resolved to the same episode")
            observation_a = session_a["observation"]["observation_id"]
            observation_b = session_b["observation"]["observation_id"]
            if request(client, credential_a, "GET", "/agent/observations/" + observation_b).status_code not in {403, 404}:
                raise RuntimeError("token A read token B observation")
            if request(client, credential_b, "GET", "/agent/observations/" + observation_a).status_code not in {403, 404}:
                raise RuntimeError("token B read token A observation")
            asset_a = session_a["task"]["input_asset_refs"][0]
            asset_b = session_b["task"]["input_asset_refs"][0]
            step_a = require(request(client, credential_a, "POST", "/agent/step", {
                "client_action_id": "registry-a-crop", "expected_state_version": 0,
                "action": {"type": "tool.invoke", "tool_id": "eo_gym.crop",
                           "arguments": {"asset_id": asset_a, "aoi": [0, 0, 0.375, 0.375]}}}), 200)
            step_b = require(request(client, credential_b, "POST", "/agent/step", {
                "client_action_id": "registry-b-crop", "expected_state_version": 0,
                "action": {"type": "tool.invoke", "tool_id": "eo_gym.crop",
                           "arguments": {"asset_id": asset_b, "aoi": [0.625, 0.625, 1, 1]}}}), 200)
            artifact_a, artifact_b = artifact_ref(step_a), artifact_ref(step_b)
            if artifact_a == artifact_b:
                raise RuntimeError("independent windows unexpectedly share artifact identity")
            own_a = request(client, credential_a, "GET", "/agent/artifacts/" + artifact_a + "/content")
            own_b = request(client, credential_b, "GET", "/agent/artifacts/" + artifact_b + "/content")
            require(own_a, 200)
            require(own_b, 200)
            if request(client, credential_a, "GET", "/agent/artifacts/" + artifact_b + "/content").status_code != 403:
                raise RuntimeError("token A read token B artifact")
            if request(client, credential_b, "GET", "/agent/artifacts/" + artifact_a + "/content").status_code != 403:
                raise RuntimeError("token B read token A artifact")
            result = {"phase": args.phase, "episode_a": episode_a, "episode_b": episode_b,
                      "state_version_a": step_a["state"]["state_version"],
                      "state_version_b": step_b["state"]["state_version"],
                      "artifact_a": artifact_a, "artifact_b": artifact_b,
                      "artifact_bytes_a": len(own_a.content), "artifact_bytes_b": len(own_b.content),
                      "cross_observation_denied": True, "cross_artifact_denied": True}
        else:
            initial = json.loads(initial_path.read_text())
            response_a = request(client, credential_a, "GET", "/agent/state")
            state_b = require(request(client, credential_b, "GET", "/agent/state"), 200)["state"]
            if state_b["episode_id"] != initial["episode_b"]:
                raise RuntimeError("token B scope changed")
            if args.phase == "revoked-a":
                if response_a.status_code != 401 or response_a.json()["error"]["code"] != "session_revoked":
                    raise RuntimeError("revoked token A remained active")
                result = {"phase": args.phase, "token_a_code": "session_revoked",
                          "episode_b": state_b["episode_id"]}
            else:
                if response_a.status_code != 401 or response_a.json()["error"]["code"] != "unauthorized":
                    raise RuntimeError("old token A remained active after rotation")
                credential_a2 = token(args.token_a2)
                state_a2 = require(request(client, credential_a2, "GET", "/agent/state"), 200)["state"]
                if state_a2["episode_id"] != initial["episode_a"]:
                    raise RuntimeError("rotated token changed episode scope")
                result = {"phase": args.phase, "old_token_a_code": "unauthorized",
                          "episode_a": state_a2["episode_id"], "episode_b": state_b["episode_id"],
                          "state_version_a": state_a2["state_version"],
                          "state_version_b": state_b["state_version"]}
    write_report(args.reports / ("registry-" + args.phase + ".json"), result)
    print(json.dumps({"status": "passed", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
