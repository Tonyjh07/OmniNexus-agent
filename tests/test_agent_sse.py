"""Agent Loop integration: tool SSE frames, persistence and step budget."""

from __future__ import annotations

import json

import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from tests.conftest import app_client, base_config
from tests.test_contract import parse_sse


def _tool_loop_factory(path: str = "agent.txt"):
    def factory(model_cfg, provider_cfg):
        async def stream(messages, agent_info):
            returns = sum(
                1
                for message in messages
                if message.kind == "request"
                and any(part.part_kind == "tool-return" for part in message.parts)
            )
            if returns == 0:
                yield "我先写一个文件。"
                yield {
                    0: DeltaToolCall(
                        name="file_write",
                        json_args=json.dumps({"path": path, "content": "agent wrote this"}),
                        tool_call_id="call-1",
                    )
                }
            else:
                yield "写好了。"

        return FunctionModel(stream_function=stream, model_name=model_cfg.id)

    return factory


@pytest.mark.asyncio
async def test_tool_frames_and_persisted_trace(tmp_path) -> None:
    config = base_config(tmp_path)
    async with app_client(config, model_factory=_tool_loop_factory()) as (client, app):
        response = await client.post(
            "/v1/events",
            json={
                "event_id": "61616161-6161-4161-8161-616161616161",
                "parts": [{"modality": "text", "content": "写个文件"}],
            },
            headers={"Accept": "text/event-stream"},
        )
        assert response.status_code == 200
        events = parse_sse(response.text)
        names = [name for name, _ in events]
        assert names[0] == "meta"
        assert names[-1] == "final"

        meta = next(data for name, data in events if name == "meta")
        assert meta["agent"]["enabled"] is True
        assert "file" in meta["agent"]["tools"]

        tool_frames = [data["tool"] for name, data in events if name == "tool"]
        assert [frame["status"] for frame in tool_frames] == ["started", "finished"]
        assert tool_frames[0]["name"] == "file_write"
        assert tool_frames[0]["call_id"] == tool_frames[1]["call_id"]

        final = next(data for name, data in events if name == "final")
        assert "写好了" in final["content"]
        assert final["session_id"] == meta["session_id"]

        stored = await app.state.db.list_messages(final["session_id"])
        assistant = [message for message in stored if message.role == "assistant"][-1]
        trace = assistant.metadata.get("tool_trace")
        assert trace and trace[0]["name"] == "file_write"
        assert "agent.txt" in trace[0]["arguments"]

    workspace_file = tmp_path / "data" / config.agent.workspace_dir / "agent.txt"
    assert workspace_file.read_text(encoding="utf-8") == "agent wrote this"


@pytest.mark.asyncio
async def test_step_budget_is_enforced(tmp_path) -> None:
    config = base_config(tmp_path)
    config.agent.max_steps = 2

    def factory(model_cfg, provider_cfg):
        async def stream(messages, agent_info):
            yield {
                0: DeltaToolCall(
                    name="file_list",
                    json_args=json.dumps({"path": "."}),
                    tool_call_id="call-x",
                )
            }

        return FunctionModel(stream_function=stream, model_name=model_cfg.id)

    async with app_client(config, model_factory=factory) as (client, _):
        response = await client.post(
            "/v1/events",
            json={
                "event_id": "62626262-6262-4262-8262-626262626262",
                "parts": [{"modality": "text", "content": "loop forever"}],
            },
            headers={"Accept": "text/event-stream"},
        )
        assert response.status_code == 200
        events = parse_sse(response.text)
        errors = [data for name, data in events if name == "error"]
        assert errors
        code = errors[-1].get("code") or errors[-1].get("error", {}).get("code")
        assert code == "GENERATION_INTERRUPTED"
