"""AgentEngine：Pydantic AI 驱动的 Agent Loop，负责模型构造、工具循环与降级。

职责边界：
- 模型构造：config 里的 providers/models → Pydantic AI 模型（openai/anthropic/gemini/cohere）；
- Agent Loop：驱动 Pydantic AI ``Agent`` 跑「推理 → 工具 → 观察 → 再推理」，直到最终答案；
- 降级：按 RoutePlan.candidates 顺序逐个候选尝试，与旧 gateway 的 fallback 语义对齐
  （记录 attempts、selected_model、fallback_used）；
- 错误映射：Pydantic AI 异常 → 现有 ErrorCode（PROVIDER_AUTH/RATE_LIMITED/TIMEOUT 等）；
- echo 提供方不走 Pydantic AI，直接本地生成诊断回复。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pydantic_ai.exceptions as PAE
import pydantic_ai.messages as M
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from omninexus.config import AppConfig, ModelConfig, ProviderConfig
from omninexus.core.models import LLMImage, LLMMessage, TokenUsage
from omninexus.core.tokenizer import estimate_tokens
from omninexus.errors import ErrorCode, OmniError, ProviderError
from omninexus.logging import get_logger

log = get_logger("agent.engine")

# ---------------------------------------------------------------------------
# 事件帧：engine 输出的内部事件，由 orchestrator 转成 SSE 帧
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RouteFrame:
    provider: str
    model: str
    fallback: bool


@dataclass(slots=True)
class TextFrame:
    text: str


@dataclass(slots=True)
class ToolStartFrame:
    name: str
    arguments: str
    call_id: str


@dataclass(slots=True)
class ToolEndFrame:
    name: str
    output: str
    call_id: str
    ok: bool


@dataclass(slots=True)
class ErrorFrame:
    error: OmniError


@dataclass(slots=True)
class DoneFrame:
    text: str
    usage: TokenUsage
    steps: int
    tool_calls: int
    interrupted: bool = False


AgentFrame = (
    RouteFrame | TextFrame | ToolStartFrame | ToolEndFrame | ErrorFrame | DoneFrame
)

# ---------------------------------------------------------------------------
# 错误映射
# ---------------------------------------------------------------------------


def map_provider_error(exc: Exception) -> OmniError:
    """把 Pydantic AI 的异常映射到现有 ErrorCode。"""
    if isinstance(exc, PAE.ModelAuthenticationError):
        return ProviderError(
            "Provider authentication failed. Check the local API key configuration.",
            code=ErrorCode.PROVIDER_AUTH,
            status_code=502,
            retryable=False,
        )
    if isinstance(exc, PAE.ModelRateLimitError):
        return ProviderError(
            "Provider rate limit reached.",
            code=ErrorCode.PROVIDER_RATE_LIMITED,
            status_code=503,
            retryable=True,
        )
    if isinstance(exc, PAE.ModelOverloadedError):
        return ProviderError(
            "Provider is overloaded, retry later.",
            code=ErrorCode.PROVIDER_RATE_LIMITED,
            status_code=503,
            retryable=True,
        )
    if isinstance(exc, PAE.ModelQuotaExceededError):
        return ProviderError(
            "Provider quota exceeded.",
            code=ErrorCode.PROVIDER_RATE_LIMITED,
            status_code=503,
            retryable=False,
        )
    if isinstance(exc, (PAE.ModelRequestTimeout, PAE.ModelTimeoutError, PAE.ModelConnectionError)):
        return ProviderError(
            "Provider timed out or is unreachable.",
            code=ErrorCode.PROVIDER_TIMEOUT,
            status_code=504,
            retryable=True,
        )
    if isinstance(exc, PAE.ModelNotFoundError):
        return ProviderError(
            "The requested model was not found on the provider.",
            code=ErrorCode.MODEL_UNAVAILABLE,
            status_code=404,
            retryable=False,
        )
    if isinstance(exc, PAE.ModelContextWindowExceededError):
        return ProviderError(
            "The conversation exceeds the model context window.",
            code=ErrorCode.PROVIDER_BAD_RESPONSE,
            status_code=502,
            retryable=False,
        )
    if isinstance(exc, PAE.ModelRequestRejectedError):
        return ProviderError(
            "The provider rejected the request.",
            code=ErrorCode.PROVIDER_AUTH,
            status_code=502,
            retryable=False,
        )
    if isinstance(exc, PAE.ModelHTTPError):
        status = exc.status_code
        if status in (401, 403):
            return ProviderError(
                "Provider authentication failed.",
                code=ErrorCode.PROVIDER_AUTH,
                status_code=502,
                retryable=False,
            )
        if status == 429:
            return ProviderError(
                "Provider rate limit reached.",
                code=ErrorCode.PROVIDER_RATE_LIMITED,
                status_code=503,
                retryable=True,
            )
        if status == 404:
            return ProviderError(
                "The requested model was not found on the provider.",
                code=ErrorCode.MODEL_UNAVAILABLE,
                status_code=404,
                retryable=False,
            )
        if status in (408, 504):
            return ProviderError(
                "Provider timed out.",
                code=ErrorCode.PROVIDER_TIMEOUT,
                status_code=504,
                retryable=True,
            )
        return ProviderError(
            "The upstream model provider rejected the request.",
            code=ErrorCode.PROVIDER_ERROR,
            status_code=502,
            retryable=status >= 500,
        )
    if isinstance(exc, (PAE.ModelServerError, PAE.ModelUnavailableError)):
        return ProviderError(
            "Provider is temporarily unavailable.",
            code=ErrorCode.PROVIDER_ERROR,
            status_code=503,
            retryable=True,
        )
    if isinstance(exc, PAE.UnexpectedModelBehavior):
        return ProviderError(
            "The model returned an unexpected response.",
            code=ErrorCode.PROVIDER_BAD_RESPONSE,
            status_code=502,
            retryable=True,
        )
    if isinstance(exc, PAE.ModelAPIError):
        return ProviderError(
            "The upstream model provider failed.",
            code=ErrorCode.PROVIDER_ERROR,
            status_code=502,
            retryable=True,
        )
    return OmniError(
        ErrorCode.INTERNAL_ERROR,
        "The agent loop failed unexpectedly.",
        status_code=500,
    )


# ---------------------------------------------------------------------------
# 消息历史转换
# ---------------------------------------------------------------------------


def _image_part(image: LLMImage, *, model_id: str) -> M.UserContent:
    """把一张图片转成 Pydantic AI 的 content part。

    data URI 转成 ``BinaryContent``（OpenAI/Anthropic/Gemini 都接受 inline bytes，
    而 Anthropic 不接受 ``data:`` URL）；http(s) URL 透传，由厂商或模型适配器处理。
    """
    url = image.url or ""
    if url.startswith("data:"):
        try:
            return M.BinaryContent.from_data_uri(url)
        except ValueError:
            return (
                f"[Image attachment unavailable for {model_id or 'this model'}: invalid data URI]"
            )
    if url.startswith(("http://", "https://")):
        return M.ImageUrl(url=url, media_type=image.mime_type or None)
    return (
        f"[Image attachment unavailable for {model_id or 'this model'}: "
        f"{image.alt or 'unsupported image source'}]"
    )


def convert_history(
    messages: list[LLMMessage],
    *,
    system_prompt: str | None = None,
    supports_images: bool = True,
    model_id: str = "",
) -> list[M.ModelMessage]:
    """把 ContextBuilder 产出的 LLMMessage 历史转成 Pydantic AI 消息。

    system prompt 以 ``SystemPromptPart`` 放在历史最前：Pydantic AI 2.x 在
    ``user_prompt=None + message_history=...`` 时会忽略 ``Agent(system_prompt=...)``，
    放进历史才能可靠地作为 system 消息发送。
    system 角色的历史消息会被跳过（system prompt 由编排层统一注入）。
    候选模型不支持图像时，把图片降级为文本占位符（与旧 gateway 的
    ``_adapt_messages`` 语义一致），保证 fallback 到纯文本模型也能回答。
    """
    history: list[M.ModelMessage] = []
    if system_prompt and system_prompt.strip():
        history.append(M.ModelRequest(parts=[M.SystemPromptPart(content=system_prompt)]))
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            parts: list[M.UserContent] = []
            if message.images and supports_images:
                for image in message.images:
                    parts.append(_image_part(image, model_id=model_id))
            if message.content:
                parts.append(message.content)
            if message.images and not supports_images:
                placeholders = [
                    f"[Image omitted for {model_id or 'this model'}: {image.alt or 'no alt text'}]"
                    for image in message.images
                ]
                parts.append("\n".join(placeholders))
            content: M.UserContent = parts[0] if len(parts) == 1 else parts
            history.append(M.ModelRequest(parts=[M.UserPromptPart(content=content)]))
        elif message.role == "assistant":
            history.append(M.ModelResponse(parts=[M.TextPart(content=message.content)]))
    return history


# ---------------------------------------------------------------------------
# 本地限流
# ---------------------------------------------------------------------------


class _TokenBucket:
    """按每分钟请求数补桶；无等待额度时直接返回 False（触发候选切换）。"""

    def __init__(self, rpm: int, capacity: int = 1) -> None:
        self.rate = rpm / 60.0
        self.capacity = float(max(1, capacity))
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, max_wait_s: float) -> bool:
        async with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            if max_wait_s <= 0:
                return False
            wait = (1.0 - self.tokens) / self.rate
            if wait > max_wait_s:
                return False
        await asyncio.sleep(wait)
        async with self._lock:
            self.tokens = max(0.0, self.tokens - 1.0)
        return True


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


class AgentEngine:
    def __init__(
        self,
        config: AppConfig,
        *,
        tools: list[Any] | None = None,
        model_factory: Any | None = None,
        transport: Any | None = None,
    ) -> None:
        self.config = config
        self._tools = tools  # list[Tool]，None 时由 orchestrator 注入默认工具
        self._model_factory = model_factory or self._build_model
        self._buckets: dict[str, _TokenBucket] = {}
        # 测试/自定义传输：注入后所有 provider 共用同一个（legacy）AsyncClient。
        self._transport = transport
        self._clients: dict[tuple[Any, ...], Any] = {}

    async def close(self) -> None:
        for client in self._clients.values():
            with contextlib.suppress(Exception):  # pragma: no cover - 关闭阶段尽力而为
                await client.aclose()
        self._clients.clear()

    # -- 模型构造 -----------------------------------------------------------

    def _timeouts(self, provider_cfg: ProviderConfig) -> tuple[float, float]:
        """(connect, read) 秒；provider 可覆盖 connect。"""
        connect = provider_cfg.connect_timeout_s or self.config.gateway.connect_timeout_s
        read = self.config.gateway.request_timeout_s
        return float(connect), float(read)

    def _http_client(self, provider_cfg: ProviderConfig, *, legacy: bool = False) -> Any:
        """按 provider 的超时配置复用 AsyncClient。

        生产路径用 ``httpx2``（pydantic-ai 的原生客户端）；``legacy=True`` 供 Cohere SDK
        与测试注入的 ``httpx.MockTransport`` 使用。
        """
        connect, read = self._timeouts(provider_cfg)
        key = (legacy, self._transport is not None, connect, read)
        if key in self._clients:
            return self._clients[key]
        if legacy:
            import httpx

            client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(read, connect=connect, write=read, pool=connect),
            )
        else:
            import httpx2

            client = httpx2.AsyncClient(
                transport=self._transport,
                timeout=httpx2.Timeout(connect=connect, read=read, write=read, pool=connect),
            )
        self._clients[key] = client
        return client

    def _build_model(self, model_cfg: ModelConfig, provider_cfg: ProviderConfig) -> Model | None:
        """config 的 provider/model → Pydantic AI Model；echo 返回 None（不走网络）。"""
        if provider_cfg.type == "echo":
            return None

        key = provider_cfg.api_key()
        use_legacy = self._transport is not None

        if provider_cfg.type == "openai":
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            if use_legacy and provider_cfg.headers:
                from openai import AsyncOpenAI

                client = AsyncOpenAI(
                    base_url=provider_cfg.base_url,
                    api_key=key or "sk-none",
                    default_headers=dict(provider_cfg.headers),
                    http_client=self._http_client(provider_cfg, legacy=True),
                )
                provider = OpenAIProvider(openai_client=client)
            else:
                provider = OpenAIProvider(
                    base_url=provider_cfg.base_url,
                    api_key=key,
                    http_client=self._http_client(provider_cfg, legacy=use_legacy),
                )
            return OpenAIChatModel(model_cfg.model, provider=provider)

        if provider_cfg.type == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicModel(
                model_cfg.model,
                provider=AnthropicProvider(
                    api_key=key,
                    base_url=provider_cfg.base_url or None,
                    http_client=self._http_client(provider_cfg, legacy=use_legacy),
                ),
            )

        if provider_cfg.type == "gemini":
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider

            return GoogleModel(
                model_cfg.model,
                provider=GoogleProvider(
                    api_key=key,
                    base_url=provider_cfg.base_url or None,
                    http_client=self._http_client(provider_cfg, legacy=use_legacy),
                ),
            )

        if provider_cfg.type == "cohere":
            from pydantic_ai.models.cohere import CohereModel
            from pydantic_ai.providers.cohere import CohereProvider

            return CohereModel(
                model_cfg.model,
                provider=CohereProvider(
                    api_key=key,
                    http_client=self._http_client(provider_cfg, legacy=True),
                ),
            )

        return None

    def _candidate_model(self, model_cfg: ModelConfig) -> Model | None:
        provider_cfg = self.config.provider_config(model_cfg.provider)
        if provider_cfg is None:
            return None
        return self._model_factory(model_cfg, provider_cfg)

    # -- 单候选 Agent 循环 ----------------------------------------------------

    async def _run_candidate(
        self,
        prepared: Any,
        model: Model,
        candidate: ModelConfig,
    ) -> AsyncIterator[AgentFrame]:
        """在一个候选模型上跑完整 Agent Loop。"""
        system_prompt = prepared.system_prompt
        history = convert_history(
            prepared.messages,
            system_prompt=system_prompt,
            supports_images="image" in candidate.input_modalities,
            model_id=candidate.id,
        )
        model_settings: dict[str, Any] = {
            "temperature": prepared.temperature,
            "top_p": prepared.top_p,
        }
        max_tokens = prepared.max_tokens
        if max_tokens and max_tokens > 0:
            model_settings["max_tokens"] = max_tokens
        # 超时/重试由本层 Agent Loop 拥有：SDK 内置重试必须关掉，否则会和候选链重试叠加。
        model_settings["max_retries"] = 0
        model_settings["timeout"] = self.config.gateway.first_token_timeout_s
        model_settings["request_timeout"] = self.config.gateway.request_timeout_s

        agent_kwargs: dict[str, Any] = {
            "model": model,
            "tools": list(self._tools or []),
        }
        agent = Agent(**agent_kwargs)
        steps = 0
        tool_calls = 0
        text_so_far: list[str] = []
        route_sent = False
        run_usage = TokenUsage()
        final_output = ""

        def _emit_route() -> bool:
            nonlocal route_sent
            if route_sent:
                return False
            route_sent = True
            return True

        try:
            async with agent.run_stream_events(
                None,
                message_history=history,
                model_settings=model_settings,
                usage_limits=UsageLimits(request_limit=self.config.agent.max_steps),
            ) as events:
                async for event in events:
                    if isinstance(event, (M.ToolCallEvent, M.FunctionToolCallEvent)):
                        steps += 1
                        tool_calls += 1
                        if _emit_route():
                            yield RouteFrame(
                                provider=candidate.provider,
                                model=candidate.id,
                                fallback=False,
                            )
                        part = event.part
                        yield ToolStartFrame(
                            name=str(part.tool_name),
                            arguments=str(part.args or ""),
                            call_id=str(part.tool_call_id or ""),
                        )
                    elif isinstance(event, (M.ToolResultEvent, M.FunctionToolResultEvent)):
                        part = event.part
                        yield ToolEndFrame(
                            name=str(part.tool_name),
                            output=str(part.content or ""),
                            call_id=str(part.tool_call_id or ""),
                            ok=part.outcome == "success",
                        )
                    elif isinstance(event, M.PartStartEvent):
                        part = event.part
                        if isinstance(part, M.TextPart) and part.content:
                            if _emit_route():
                                yield RouteFrame(
                                    provider=candidate.provider,
                                    model=candidate.id,
                                    fallback=False,
                                )
                            text_so_far.append(part.content)
                            yield TextFrame(text=part.content)
                    elif isinstance(event, M.PartDeltaEvent):
                        delta = event.delta
                        if isinstance(delta, M.TextPartDelta) and delta.content_delta:
                            if _emit_route():
                                yield RouteFrame(
                                    provider=candidate.provider,
                                    model=candidate.id,
                                    fallback=False,
                                )
                            text_so_far.append(delta.content_delta)
                            yield TextFrame(text=delta.content_delta)
                run_usage = self._usage_from_events(events)
                final_output = str(getattr(events.result, "output", "") or "")
        except PAE.UsageLimitExceeded:
            # 步数预算耗尽：保留已有输出；没有任何输出时返回明确的中断错误。
            text = "".join(text_so_far)
            if not text:
                yield ErrorFrame(
                    error=OmniError(
                        ErrorCode.GENERATION_INTERRUPTED,
                        f"The agent exceeded the step budget ({self.config.agent.max_steps}); "
                        "raise agent.max_steps or simplify the task.",
                        status_code=504,
                        details={"max_steps": self.config.agent.max_steps, "steps": steps},
                    )
                )
                return
            yield DoneFrame(
                text=text,
                usage=self._estimate_usage(prepared, text),
                steps=steps,
                tool_calls=tool_calls,
                interrupted=True,
            )
            return
        except PAE.ModelAPIError as exc:
            if text_so_far:
                text = "".join(text_so_far)
                yield DoneFrame(
                    text=text,
                    usage=self._estimate_usage(prepared, text),
                    steps=steps,
                    tool_calls=tool_calls,
                    interrupted=True,
                )
                return
            yield ErrorFrame(error=map_provider_error(exc))
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            log.exception(
                "agent_candidate_failed model=%s type=%s", candidate.id, type(exc).__name__
            )
            if text_so_far:
                text = "".join(text_so_far)
                yield DoneFrame(
                    text=text,
                    usage=self._estimate_usage(prepared, text),
                    steps=steps,
                    tool_calls=tool_calls,
                    interrupted=True,
                )
                return
            yield ErrorFrame(
                error=OmniError(
                    ErrorCode.INTERNAL_ERROR,
                    "The agent loop failed unexpectedly.",
                    status_code=500,
                    details={"type": type(exc).__name__},
                )
            )
            return

        text = "".join(text_so_far)
        if not text:
            # 模型没有产出文本（可能只有工具调用后没收尾）
            text = final_output.strip()
            if text:
                yield TextFrame(text=text)
        yield DoneFrame(
            text=text,
            usage=run_usage,
            steps=steps,
            tool_calls=tool_calls,
        )

    # -- 降级循环 ------------------------------------------------------------

    async def stream_agent(self, prepared: Any) -> AsyncIterator[AgentFrame]:
        """按 plan.candidates 依次尝试；echo 直接本地回复；全失败抛 ALL_PROVIDERS_FAILED 帧。

        每个候选按 max_retries 重试可重试错误，再切换下一个候选；本地 rate limit
        与旧 gateway 的 token bucket 语义保持一致。
        """
        plan = prepared.plan
        attempts: list[dict[str, Any]] = []

        for index, candidate in enumerate(plan.candidates):
            model = self._candidate_model(candidate)
            if model is None:
                provider_cfg = self.config.provider_config(candidate.provider)
                if provider_cfg and provider_cfg.type == "echo":
                    # 离线诊断模型：本地生成，不联网。
                    plan.selected_model = candidate.id
                    plan.selected_provider = candidate.provider
                    plan.fallback_used = index > 0
                    yield RouteFrame(
                        provider=candidate.provider, model=candidate.id, fallback=index > 0
                    )
                    async for echo_frame in self._stream_echo(prepared):
                        yield echo_frame
                    return
                attempts.append(
                    {
                        "provider": candidate.provider,
                        "model": candidate.id,
                        "code": ErrorCode.PROVIDER_ERROR.value,
                    }
                )
                continue

            retries_left = max(0, self.config.gateway.max_retries)
            while True:
                attempt = {"provider": candidate.provider, "model": candidate.id}
                if retries_left < self.config.gateway.max_retries:
                    attempt["retry"] = self.config.gateway.max_retries - retries_left

                if not await self._rate_limit(candidate.provider):
                    attempts.append(
                        {
                            **attempt,
                            "code": ErrorCode.PROVIDER_RATE_LIMITED.value,
                            "retryable": True,
                            "local": True,
                        }
                    )
                    break  # 本地限流：直接切换下一个候选

                saw_text = False
                emitted_route = False
                retryable = False
                try:
                    async for frame in self._run_candidate(prepared, model, candidate):
                        if isinstance(frame, RouteFrame):
                            if not emitted_route:
                                emitted_route = True
                                plan.selected_model = candidate.id
                                plan.selected_provider = candidate.provider
                                plan.fallback_used = index > 0
                                yield RouteFrame(
                                    provider=candidate.provider,
                                    model=candidate.id,
                                    fallback=index > 0,
                                )
                        elif isinstance(frame, TextFrame):
                            saw_text = True
                            yield frame
                        elif isinstance(frame, (ToolStartFrame, ToolEndFrame)):
                            yield frame
                        elif isinstance(frame, DoneFrame):
                            yield frame
                            attempt["ok"] = True
                            attempt["steps"] = frame.steps
                            plan.attempts = attempts + [attempt]
                            return
                        elif isinstance(frame, ErrorFrame):
                            attempt.update(
                                {"code": frame.error.code.value, "retryable": frame.error.retryable}
                            )
                            retryable = bool(frame.error.retryable)
                            if frame.error.code == ErrorCode.GENERATION_INTERRUPTED:
                                # 步数预算耗尽等终止性错误：不再切换候选。
                                attempts.append(attempt)
                                plan.attempts = attempts
                                yield frame
                                return
                            if saw_text:
                                # 已经产出部分文本，不能安全切换到其他模型重放。
                                attempts.append(attempt)
                                yield DoneFrame(
                                    text="",
                                    usage=TokenUsage(),
                                    steps=0,
                                    tool_calls=0,
                                    interrupted=True,
                                )
                                plan.attempts = attempts
                                return
                            break
                    else:
                        # 循环正常结束但未 yield DoneFrame（理论上不会发生）
                        attempts.append({**attempt, "code": "NO_OUTPUT"})
                        break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover
                    attempts.append(
                        {**attempt, "code": "INTERNAL_ERROR", "type": type(exc).__name__}
                    )
                    break

                attempts.append(attempt)
                if retryable and retries_left > 0:
                    retries_left -= 1
                    if self.config.gateway.retry_backoff_s:
                        await asyncio.sleep(self.config.gateway.retry_backoff_s)
                    continue
                break  # 尝试下一个候选

        plan.attempts = attempts
        yield ErrorFrame(
            error=OmniError(
                ErrorCode.ALL_PROVIDERS_FAILED,
                "All candidate models failed. Check provider configuration and try again.",
                status_code=503,
                details={"attempts": attempts},
            )
        )

    async def _rate_limit(self, provider: str) -> bool:
        """本地 token bucket；返回 False 表示本候选应被跳过。"""
        rpm = self.config.gateway.rate_limit_rpm
        provider_cfg = self.config.provider_config(provider)
        if provider_cfg and isinstance(provider_cfg.options.get("rate_limit_rpm"), (int, float)):
            rpm = int(provider_cfg.options["rate_limit_rpm"])
        if rpm <= 0:
            return True
        bucket = self._buckets.get(provider)
        if bucket is None:
            bucket = _TokenBucket(rpm)
            self._buckets[provider] = bucket
        return await bucket.acquire(self.config.gateway.rate_limit_max_wait_s)

    # -- echo ---------------------------------------------------------------

    async def _stream_echo(self, prepared: Any) -> AsyncIterator[AgentFrame]:
        messages: list[LLMMessage] = prepared.messages
        user_messages = [m for m in messages if m.role == "user"]
        last = (user_messages[-1].content if user_messages else "").strip()
        last = " ".join(last.split())[:500]
        text = (
            "Offline diagnostic echo (no external model was called).\n\n"
            f"Your message: {last or '(empty)'}\n\n"
            f"Conversation messages in context: {len(messages)}. "
            "Configure a provider key in .env and switch models to use a real LLM."
        )
        tokens = max(1, estimate_tokens(text) // 10)
        for index in range(0, len(text), tokens):
            yield TextFrame(text=text[index : index + tokens])
        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        completion_tokens = estimate_tokens(text)
        yield DoneFrame(
            text=text,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            steps=0,
            tool_calls=0,
        )

    # -- usage --------------------------------------------------------------

    @staticmethod
    def _usage_from_events(events: Any) -> TokenUsage:
        usage = getattr(events, "usage", None)
        if usage is None:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        )

    @staticmethod
    def _estimate_usage(prepared: Any, text: str) -> TokenUsage:
        prompt = sum(estimate_tokens(m.content) for m in prepared.messages)
        completion = estimate_tokens(text)
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )
