# Changelog

## 0.2.0 — 2026-10-30

在模型层引入 Pydantic AI 驱动的 **Agent Loop**，对上游/下游保持 drop-in 兼容。

### Added

- Agent Loop（`src/omninexus/agent/`）：推理 → 工具调用 → 观察 → 再推理，`agent.max_steps`
  映射到请求预算，超预算映射为 `GENERATION_INTERRUPTED`。
- 内置工具：`file_read` / `file_write` / `file_list`（工作区锁）、`shell`（程序与 git 子命令
  白名单）、`python_code`（独立进程，超时 + 输出上限）。
- 工作区沙箱：`{data_dir}/omninexus/agent-workspace`；路径 `resolve()` 后父子判定，拒绝 `..`
  与符号链接逃逸；Windows 子进程 `CREATE_NO_WINDOW`。
- SSE 新增 `tool` 帧（`started`/`finished`/`failed` + `call_id`），旧帧语义不变。
- `meta.agent` 段（`enabled` / `max_steps` / `tools`）；`metadata.tool_trace` 随存档持久化。
- 配置：`agent.enabled` / `max_steps` / `workspace_dir` / `tools` / `shell_allow` / `git_allow`
  / `timeout_s` / `max_output_chars`。
- 文档：[Agent Loop](docs/agent-loop.md)。

### Changed

- 模型调用改由 Pydantic AI 统一适配（`OpenAIChatModel` / `AnthropicModel` / `GoogleModel` /
  `CohereModel`），替代原 `gateway/` 下的厂商适配器网络代码。
- `gateway/router.py` 瘦身为纯路由（`plan()`），只负责候选链与能力协商。
- 多模态：`data:` URI 转内联二进制，`http(s)` 透传 URL，其他来源转文字占位；文本模型仍降级。
- 依赖：`pydantic-ai-slim[openai,anthropic,google,cohere]>=2.50,<3.0`。

### Removed

- `gateway/` 下的 `anthropic.py` / `base.py` / `cohere.py` / `echo.py` / `gemini.py` /
  `openai_compat.py`（逻辑迁入 `agent/engine.py`）。
- `core.models.ChatDelta` / `ChatResult`（不再有厂商流式对象）。
- `tests/test_provider_adapters.py`（由 `test_agent_engine.py` 等替代）。

### Notes

- **Drop-in**：`module_id` 仍为 `omninexus`，端口 8102，`/v1/providers` 字段不变，旧 SSE 帧语义
  不变；新增帧可被忽略，Dounai 与启动器无需改动。
- 工具沙箱是**受限 Shell**，不是容器级隔离；信任边界等于模型层进程权限。
- 无 human-in-the-loop 批准通道（v2 计划）。

## 0.1.0 — 2026-10-24

首个可独立运行的交付版本（任务书 v1.1）。

### Added

- 多厂商模型网关：OpenAI-compatible（OpenAI、DeepSeek、Mistral、xAI、Qwen、Doubao、GLM、
  Moonshot、Qianfan、Spark、Hunyuan、MiniMax、OpenRouter、Ollama、LM Studio、vLLM、llama.cpp）、
  Anthropic、Gemini、Cohere 原生适配器。
- 运行时模型切换、上下文延续、重试、首 Token 超时、进程内限流、模型回退链与已配置模型重路由。
- 多模态能力协商：视觉路由、文本化降级、回退时图片剥离。
- 提示词档案与不可变版本：创建/保存/激活/重置/删除/预览；Jinja2 沙箱变量注入。
- 本地对话存档（SQLite WAL）：会话、消息、引用、导出 JSON/Markdown、删除、Token 预算裁剪。
- 确定性情感/内容类型/意图标签，随 `delta`/`final` 事件下发。
- 可配置联网搜索（Tavily/Brave/SearXNG/DuckDuckGo/自定义），RAG + 引用溯源，查询最小化。
- SSE 契约：`meta`/`route`/`delta`/`final`/`citations`/`warning`/`error` 与心跳。
- 本地零构建 Web 页面：对话、模型/提示词切换、图片、搜索、存档、导出/删除、提示词管理。
- 隐私：密钥仅环境变量、日志脱敏、错误正文不回显、`.env`/data/log 忽略。
- **Reserved**：外部记忆插件接口（默认禁用、健康探测 `unavailable`、契约 v1.0、mock 验证）。
- 文档：架构、API、提示词、厂商、降级、隐私、用户、开发、测试报告、验收对照。
- CI、Dockerfile、docker-compose、MIT License。

### Notes

- 未内置正式人设或测试人格；默认系统提示词为空。
- 外部记忆管线不参与本期交付。
- 不提供密钥/账户（BYOK）。
