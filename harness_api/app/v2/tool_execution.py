"""Durable reserve/run/finalize path. No database write lock during tool I/O."""
from __future__ import annotations

import json
import time
import uuid

from .budgets import exhausted, update_budget
from .domain import V2DomainError, _action_allowed, elapsed_ms, utc_now
from .events import canonical_json, create_event
from .observations import semantic_state_hash, state_hash
from .schemas import EpisodeResultData, Observation, ObservationItem, V2EpisodeState

LEASE_SECONDS = 90


def error_value(error: V2DomainError) -> dict:
    return {key: getattr(error, key) for key in ("code", "message", "status_code", "retryable", "phase", "details")}


class ToolExecutionMixin:
    def _assert_no_pending_tool(self, connection, episode_id):
        pending = connection.execute("SELECT tool_run_id FROM v2_tool_runs WHERE episode_id=? AND status='running' LIMIT 1", (episode_id,)).fetchone()
        if pending:
            raise V2DomainError("tool_in_progress", "an action is already running for this episode", 409, True, "state")

    def recover_tool_runs(self, episode_id):
        with self._connect() as connection:
            pending = connection.execute("SELECT tool_run_id, run_json FROM v2_tool_runs WHERE episode_id=? AND status='running'", (episode_id,)).fetchall()
        for row in pending:
            if json.loads(row["run_json"])["lease_until"] < time.time():
                self._finalize_tool(row["tool_run_id"], None, V2DomainError(
                    "tool_interrupted", "tool lease expired; outcome is unknown and was not automatically repeated", 409, False, "tool"))

    def _tool_step(self, episode_id, expected_state_version, client_action_id, action):
        request_json = canonical_json({"expected_state_version": expected_state_version, "action": action.model_dump(mode="json")})
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._load_episode(connection, episode_id)
            prior = connection.execute("SELECT request_json,outcome,response_json FROM v2_action_results WHERE episode_id=? AND client_action_id=?", (episode_id, client_action_id)).fetchone()
            if prior:
                if prior["request_json"] != request_json:
                    raise V2DomainError("idempotency_conflict", "action ID was reused with different arguments", 409)
                if prior["outcome"] == "error":
                    raise V2DomainError(**json.loads(prior["response_json"]))
                return EpisodeResultData.model_validate_json(prior["response_json"])
            pending = connection.execute("SELECT run_json FROM v2_tool_runs WHERE episode_id=? AND status='running'", (episode_id,)).fetchone()
            if pending:
                run = json.loads(pending["run_json"])
                if run["client_action_id"] == client_action_id and run["request_json"] != request_json:
                    raise V2DomainError("idempotency_conflict", "in-flight action ID was reused", 409)
                self._assert_no_pending_tool(connection, episode_id)
            state = V2EpisodeState.model_validate_json(row["state_json"])
            if state.state_version != expected_state_version:
                raise V2DomainError("state_version_conflict", "expected state version is stale", 409, True)
            if state.status != "active":
                raise V2DomainError("episode_closed", "episode is no longer active", 409)
            manifest = self._manifest_for_row(row)
            if not _action_allowed("tool.invoke", manifest.scenario.allowed_actions):
                raise V2DomainError("policy_rejected", "task does not allow tools", 403, phase="policy")
            _, asset = self.tool_executor.prepare(action, manifest)
            if asset.asset_id not in state.accessible_asset_refs:
                raise V2DomainError("policy_rejected", "asset is not accessible in this episode", 403, phase="policy")
            budget = state.budget
            if budget.steps.remaining < 1 or budget.tool_calls.remaining < 1 or elapsed_ms(state.created_at, utc_now()) >= budget.wall_time_ms.limit:
                raise V2DomainError("tool_budget_exceeded", "step, tool or time budget exhausted", phase="policy")
            if asset.size_bytes > budget.input_bytes.remaining or budget.artifact_bytes.remaining < self.tool_executor.max_output_bytes:
                raise V2DomainError("tool_budget_exceeded", "insufficient input or output byte reservation", phase="policy")
            timestamp = utc_now()
            run_id = "tool-" + uuid.uuid4().hex
            run = {"client_action_id": client_action_id, "request_json": request_json,
                   "lease_until": time.time() + LEASE_SECONDS, "input_bytes_reserved": asset.size_bytes,
                   "output_bytes_reserved": self.tool_executor.max_output_bytes,
                   "tool_version": self.tool_executor.tool_version, "expected_state_version": expected_state_version}
            connection.execute("INSERT INTO v2_tool_runs VALUES (?,?,?,?,?,?)", (run_id, episode_id, action.tool_id, "running", canonical_json(run), timestamp))
            self._insert_event(connection, create_event(episode_id, self._next_sequence(connection, episode_id),
                "action.accepted", state.state_version, timestamp, {"client_action_id": client_action_id,
                "expected_state_version": expected_state_version, "action": action.model_dump(mode="json")}))
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        output = None
        failure = None
        try:
            output = self.tool_executor.invoke(action, manifest)
        except V2DomainError as error:
            failure = error
        except Exception:
            failure = V2DomainError("tool_failed", "tool execution failed", 502, phase="tool")
        response, stored_failure = self._finalize_tool(run_id, output, failure)
        if stored_failure:
            raise V2DomainError(**stored_failure)
        return response

    def _finalize_tool(self, run_id, output, failure):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM v2_tool_runs WHERE tool_run_id=?", (run_id,)).fetchone()
            run = json.loads(row["run_json"])
            episode_id = row["episode_id"]
            if row["status"] != "running":
                prior = connection.execute("SELECT outcome,response_json FROM v2_action_results WHERE episode_id=? AND client_action_id=?", (episode_id, run["client_action_id"])).fetchone()
                if prior["outcome"] == "error":
                    return None, json.loads(prior["response_json"])
                return EpisodeResultData.model_validate_json(prior["response_json"]), None
            state = V2EpisodeState.model_validate_json(self._load_episode(connection, episode_id)["state_json"])
            if state.state_version != run["expected_state_version"]:
                raise RuntimeError("episode changed while reserved tool was executing")
            if output is not None and output.artifact.size_bytes > run["output_bytes_reserved"]:
                output = None
                failure = V2DomainError("tool_output_too_large", "artifact exceeded reservation", phase="tool")
            if output is not None:
                existing = connection.execute("SELECT artifact_json FROM v2_artifacts WHERE artifact_id=?", (output.artifact.artifact_id,)).fetchone()
                if existing and existing["artifact_json"] != canonical_json(output.artifact.model_dump(mode="json")):
                    output = None
                    failure = V2DomainError("artifact_metadata_conflict", "identical content already has different provenance; refusing to overwrite it", 409, phase="artifact")
            timestamp = utc_now()
            state.state_version += 1
            state.step_count += 1
            state.updated_at = timestamp
            state.budget = update_budget(state.budget, elapsed_ms(state.created_at, timestamp), step_increment=1,
                tool_call_increment=1, input_byte_increment=output.input_bytes if output else 0,
                artifact_byte_increment=output.artifact.size_bytes if output else 0)
            if exhausted(state.budget):
                state.status = "truncated"
            failure_json = error_value(failure) if failure else None
            observation_id = "obs-" + uuid.uuid4().hex
            state.observation_refs.append(observation_id)
            items = [ObservationItem(type="tool_result", inline={"tool_id": row["tool_id"],
                "tool_version": run["tool_version"], "status": "failed" if failure else "completed",
                **({"error_code": failure.code} if failure else output.metadata)})]
            if output is not None:
                items.append(ObservationItem(type="raster_chip", artifact_ref=output.artifact.artifact_id))
            observation = Observation(observation_id=observation_id, sequence=state.step_count, primary_type="tool_result",
                items=items, state_hash=state_hash(state), semantic_state_hash=semantic_state_hash(state),
                provenance={"builder": "isolated-tool", "task_manifest_hash": state.task_manifest_hash}, warnings=[])
            self._insert_observation(connection, episode_id, observation, timestamp)
            if output is not None:
                self._register_artifact(connection, episode_id, observation_id, output.artifact, timestamp)
            event_values = [("action.failed" if failure else "action.completed", {
                "action_type": "tool.invoke", "tool_id": row["tool_id"], "tool_version": run["tool_version"],
                "observation_id": observation_id, "state_hash": state_hash(state),
                **({"code": failure.code} if failure else {"artifact_sha256": output.artifact.sha256})})]
            if output is not None:
                event_values.append(("artifact.created", {"artifact": output.artifact.model_dump(mode="json"), "observation_id": observation_id}))
            event_values.append(("observation.emitted", {"observation": observation.model_dump(mode="json")}))
            if state.status == "truncated":
                event_values.append(("episode.truncated", {"reason": "budget_exhausted"}))
            for event_type, payload in event_values:
                self._insert_event(connection, create_event(episode_id, self._next_sequence(connection, episode_id), event_type,
                    state.state_version, timestamp, payload))
            response = self._episode_result(state, observation)
            response_json = canonical_json(failure_json if failure else response.model_dump(mode="json"))
            connection.execute("INSERT INTO v2_action_results VALUES (?,?,?,?,?,?)", (episode_id, run["client_action_id"], run["request_json"],
                "error" if failure else "success", response_json, timestamp))
            connection.execute("UPDATE v2_episodes SET updated_at=?,status=?,state_version=?,step_count=?,state_json=? WHERE episode_id=?",
                (timestamp, state.status, state.state_version, state.step_count, canonical_json(state.model_dump(mode="json")), episode_id))
            run.update(error=failure_json, completed_at=timestamp, logical_input_bytes=output.input_bytes if output else None,
                       output_bytes=output.artifact.size_bytes if output else 0)
            connection.execute("UPDATE v2_tool_runs SET status=?,run_json=? WHERE tool_run_id=?", ("failed" if failure else "completed", canonical_json(run), run_id))
            connection.commit()
            return (None, failure_json) if failure else (response, None)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
