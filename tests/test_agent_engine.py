"""Agent engine: provider construction, error mapping and fallback semantics."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic_ai.exceptions import (
    ModelAuthenticationError,
    ModelHTTPError,
    ModelRateLimitError,
    ModelRequestTimeout,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel

from omninexus.agent.engine import (
    AgentEngine,
    DoneFrame,
    ErrorFrame,
    RouteFrame,
    TextFrame,
    map_provider_error,
)
from omninexus.config import AppConfig, GatewayConfig, ModelConfig, ProviderConfig
from omninexus.core.models import ContentPart, LLMImage, LLMMessage, TokenUsage
from omninexus.errors import ErrorCode


def _config(provider_type: str) -> AppConfig:
    provider = ProviderConfig(
        type=provider_type,
        base_url="https://example.test/v1",
        api_key_env="ALPHA_KEY",
    )
    return AppConfig(
        gateway=GatewayConfig(default_model="p:one", fallback_models=[], max_retries=0),
        providers={"p": provider},
        models=[ModelConfig(id="p:one", provider="p", model="test-model")],
    )


@pytest.mark.parametrize(
    ("provider_type", "model_type"),
    [
        ("openai", OpenAIChatModel),
        ("anthropic", AnthropicModel),
        ("gemini", GoogleModel),
        ("cohere", CohereModel),
    ],
)
def test_model_factory_builds_native_models(
    provider_type: str, model_type: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALPHA_KEY", "test-key")
    config = _config(provider_type)
    engine = AgentEngine(config)
    model = engine._build_model(config.models[0], config.providers["p"])
    assert isinstance(model, model_type)


def test_model_construction_is_cached_per_provider() -> None:
    config = _config("openai")
    config.providers["p"].connect_timeout_s = 3
    engine = AgentEngine(config)
    first = engine._http_client(config.providers["p"])
    second = engine._http_client(config.providers["p"])
    assert first is second
    third = engine._http_client(config.providers["p"], legacy=True)
    assert third is not first


def test_echo_provider_does_not_create_a_model() -> None:
    config = AppConfig(
        gateway=GatewayConfig(default_model="echo:echo"),
        providers={"echo": ProviderConfig(type="echo", requires_key=False)},
        models=[ModelConfig(id="echo:echo", provider="echo", model="echo")],
    )
    assert AgentEngine(config)._build_model(config.models[0], config.providers["echo"]) is None


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ModelAuthenticationError("m", "bad key"), ErrorCode.PROVIDER_AUTH),
        (ModelRateLimitError("m", "slow down"), ErrorCode.PROVIDER_RATE_LIMITED),
        (ModelRequestTimeout("m", 30.0), ErrorCode.PROVIDER_TIMEOUT),
        (ModelHTTPError(404, "m"), ErrorCode.MODEL_UNAVAILABLE),
        (ModelHTTPError(503, "m"), ErrorCode.PROVIDER_ERROR),
    ],
)
def test_provider_error_mapping(exc: Exception, expected: ErrorCode) -> None:
    mapped = map_provider_error(exc)
    assert mapped.code == expected
    if expected in (ErrorCode.PROVIDER_AUTH, ErrorCode.MODEL_UNAVAILABLE):
        assert mapped.retryable is False
    else:
        assert mapped.retryable is True


class _Prepared:
    """Minimal stand-in for orchestrator.PreparedTurn."""

    def __init__(self, plan, messages: list[LLMMessage]) -> None:
        self.plan = plan
        self.messages = messages
        self.system_prompt = "be brief"
        self.temperature = 0.2
        self.top_p = 1.0
        self.max_tokens = 64


def _plan(config: AppConfig):
    from omninexus.gateway.router import Gateway

    return Gateway(config).plan(
        "p:one", [ContentPart(modality="text", content="hello")]
    )


def _stream_factory(text: str):
    async def stream(messages, agent_info):
        yield text

    return stream


@pytest.mark.asyncio
async def test_fallback_records_route_and_selected_model() -> None:
    config = _config("openai")
    config.gateway.fallback_models = ["q:one"]
    config.providers["q"] = ProviderConfig(
        type="openai", base_url="https://q.test/v1", api_key_env="BETA_KEY"
    )
    config.models.append(ModelConfig(id="q:one", provider="q", model="two"))
    plan = _plan(config)

    def factory(model_cfg: ModelConfig, provider_cfg: ProviderConfig):
        if model_cfg.provider == "p":
            async def boom(messages, info):
                raise ModelHTTPError(500, model_cfg.id)
                yield ""  # pragma: no cover
            return FunctionModel(stream_function=boom, model_name=model_cfg.id)
        return FunctionModel(stream_function=_stream_factory("from q"), model_name=model_cfg.id)

    engine = AgentEngine(config, tools=[], model_factory=factory)
    prepared = _Prepared(plan, [LLMMessage(role="user", content="hello")])
    frames = [frame async for frame in engine.stream_agent(prepared)]

    routes = [frame for frame in frames if isinstance(frame, RouteFrame)]
    assert [(r.provider, r.model, r.fallback) for r in routes] == [("q", "q:one", True)]
    done = next(frame for frame in frames if isinstance(frame, DoneFrame))
    assert done.text == "from q"
    assert plan.selected_model == "q:one"
    assert plan.fallback_used is True
    assert plan.attempts[0]["code"] == ErrorCode.PROVIDER_ERROR.value


@pytest.mark.asyncio
async def test_all_candidates_failed_emits_error_frame() -> None:
    config = _config("openai")

    def factory(model_cfg: ModelConfig, provider_cfg: ProviderConfig):
        async def boom(messages, info):
            raise ModelAuthenticationError(model_cfg.id, "bad key")
            yield ""  # pragma: no cover
        return FunctionModel(stream_function=boom, model_name=model_cfg.id)

    engine = AgentEngine(config, tools=[], model_factory=factory)
    prepared = _Prepared(_plan(config), [LLMMessage(role="user", content="hello")])
    frames = [frame async for frame in engine.stream_agent(prepared)]

    error = next(frame for frame in frames if isinstance(frame, ErrorFrame))
    assert error.error.code == ErrorCode.ALL_PROVIDERS_FAILED
    assert error.error.details["attempts"][0]["code"] == ErrorCode.PROVIDER_AUTH.value


@pytest.mark.asyncio
async def test_openai_compatible_stream_usage_and_text() -> None:
    from tests.conftest import make_transport, sse_response

    config = _config("openai")
    config.gateway.default_model = "p:one"
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return sse_response(
            "streamed",
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        )

    engine = AgentEngine(config, tools=[], transport=make_transport(handler))
    prepared = _Prepared(_plan(config), [LLMMessage(role="user", content="hello")])
    try:
        frames = [frame async for frame in engine.stream_agent(prepared)]
    finally:
        await engine.close()

    text = "".join(frame.text for frame in frames if isinstance(frame, TextFrame))
    done = next(frame for frame in frames if isinstance(frame, DoneFrame))
    assert text == "streamed"
    assert done.usage == TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    body = json.loads(seen[0].read())
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert "hello" in json.dumps(body["messages"][1], ensure_ascii=False)


@pytest.mark.asyncio
async def test_tool_loop_runs_tool_and_returns_final_text() -> None:
    config = _config("openai")

    async def stream(messages, agent_info):
        returns = sum(
            1
            for message in messages
            if message.kind == "request"
            and any(part.part_kind == "tool-return" for part in message.parts)
        )
        if returns == 0:
            yield "checking"
            yield {
                0: DeltaToolCall(
                    name="file_write",
                    json_args=json.dumps({"path": "note.txt", "content": "hi"}),
                    tool_call_id="call-1",
                )
            }
        else:
            yield "done"

    def factory(model_cfg: ModelConfig, provider_cfg: ProviderConfig):
        return FunctionModel(stream_function=stream, model_name=model_cfg.id)

    import tempfile
    from pathlib import Path

    from omninexus.agent.tools import AgentTools

    tmp = Path(tempfile.mkdtemp())
    agent_tools = AgentTools(config.agent, tmp)
    engine = AgentEngine(config, tools=agent_tools.tools(), model_factory=factory)
    prepared = _Prepared(_plan(config), [LLMMessage(role="user", content="write a file")])
    frames = [frame async for frame in engine.stream_agent(prepared)]
    await engine.close()

    done = next(frame for frame in frames if isinstance(frame, DoneFrame))
    assert done.text.endswith("done")
    assert done.tool_calls == 1
    assert (tmp / config.agent.workspace_dir / "note.txt").read_text(encoding="utf-8") == "hi"


# ---------------------------------------------------------------------------
# 多模态 history 转换
# ---------------------------------------------------------------------------


def test_convert_history_inlines_data_uri_images() -> None:
    import pydantic_ai.messages as M

    from omninexus.agent.engine import convert_history

    history = convert_history(
        [
            LLMMessage(
                role="user",
                content="what is this",
                images=[LLMImage(url="data:image/png;base64,iVBORw0KGgo=", alt="sample")],
            )
        ],
        model_id="p:one",
    )
    parts = history[0].parts[0].content
    assert isinstance(parts, list)
    assert isinstance(parts[0], M.BinaryContent)
    assert parts[0].media_type == "image/png"
    assert parts[0].data.startswith(b"\x89PNG")
    assert parts[1] == "what is this"


def test_convert_history_passes_http_images_as_urls() -> None:
    import pydantic_ai.messages as M

    from omninexus.agent.engine import convert_history

    history = convert_history(
        [
            LLMMessage(
                role="user",
                content="see",
                images=[LLMImage(url="https://example.test/cat.png", mime_type="image/png")],
            )
        ],
        model_id="p:one",
    )
    parts = history[0].parts[0].content
    assert isinstance(parts[0], M.ImageUrl)
    assert parts[0].url == "https://example.test/cat.png"


def test_convert_history_rejects_unsupported_image_source() -> None:
    from omninexus.agent.engine import convert_history

    history = convert_history(
        [
            LLMMessage(
                role="user",
                images=[LLMImage(url="file:///tmp/a.png", alt="local file")],
            )
        ],
        model_id="p:one",
    )
    content = history[0].parts[0].content
    assert isinstance(content, str)
    assert "Image attachment unavailable for p:one" in content
    assert "local file" in content


def test_convert_history_degrades_images_for_text_only_model() -> None:
    from omninexus.agent.engine import convert_history

    history = convert_history(
        [
            LLMMessage(
                role="user",
                content="x",
                images=[LLMImage(url="https://example.test/cat.png", alt="cat")],
            )
        ],
        supports_images=False,
        model_id="p:text-only",
    )
    content = history[0].parts[0].content
    assert isinstance(content, list)
    assert content[0] == "x"
    assert "Image omitted for p:text-only: cat" in content[1]
