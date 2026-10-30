# 开发文档

## 1. 目录

```text
OmniNexus-agent/
├─ config/default.yaml          # 非敏感配置、厂商/模型注册表、agent 段
├─ src/omninexus/
│  ├─ api/                      # 路由、schema、SSE
│  ├─ agent/                    # Agent Loop、工具、工作区沙箱
│  ├─ core/                     # 领域模型、分类器、Token 估算
│  ├─ gateway/                  # 纯路由：候选链与能力协商（无网络调用）
│  ├─ memory/                   # Reserved 记忆插件与冻结契约
│  ├─ prompts/                  # 提示词档案/版本
│  ├─ search/                   # 搜索源与 RAG
│  ├─ sessions/                 # 会话、上下文、导出
│  ├─ storage/                  # SQLite
│  ├─ static/                   # 无构建步骤的页面
│  ├─ config.py                 # YAML + env + 注册表
│  ├─ orchestrator.py           # 回合编排
│  ├─ logging.py                # 脱敏日志
│  └─ app.py                    # FastAPI 工厂
├─ tests/                       # pytest + MockTransport
├─ docs/                        # 开发/用户文档
├─ .github/workflows/ci.yml
├─ Dockerfile / docker-compose.yml
└─ pyproject.toml
```

## 2. 本地开发

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m pytest --cov=omninexus --cov-report=term-missing
```

运行：

```powershell
$env:OMNI_ENABLE_ECHO = "1"
python -m omninexus --reload
```

`--reload` 会设置 `OMNI_CONFIG` 为当前配置路径，供 uvicorn 子进程使用。

## 3. 添加一个 OpenAI-compatible 厂商

通常只改 YAML，不改代码：

```yaml
providers:
  myvendor:
    type: openai
    label: My Vendor
    base_url: https://api.example.com/v1
    api_key_env: MYVENDOR_API_KEY
    headers: {}
    options: {}

models:
  - id: myvendor:chat
    provider: myvendor
    model: chat-v1
    label: My Chat
    input_modalities: [text, image]
    context_window: 128000
    max_output_tokens: 2048
    fallbacks: [openai:gpt-4o-mini]
```

约束：

- `model.id` 形如 `provider:model`，前缀必须等于 `provider`。
- `fallbacks` 必须引用已存在模型；不允许环（运行时深度限制 8）。
- `api_key_env` 只填变量名，不填值。
- 本地/无鉴权 provider 设 `requires_key: false`，并显式 `enabled: true`（默认样例是 false）。

## 4. 添加原生协议（厂商 SDK）

厂商 SDK 由 Pydantic AI 适配，代码里只有一处模型构造分支：

1. 在 `src/omninexus/agent/engine.py::_build_model` 增加 `provider type` 分支，返回对应的
   `pydantic_ai.models.*` 实例。
2. `ProviderConfig.type` 增加 Literal 值。
3. 若该 SDK 需要额外依赖，加进 `pyproject.toml` 的 `pydantic-ai-slim[...]` extra。
4. 用 `FunctionModel`（pydantic-ai 的测试模型）或 `httpx.MockTransport` 写测试：工具调用、
   流式增量、usage、401/429/超时。
5. 更新 `docs/providers.md` 与契约测试。

约定：

- 不记录请求/响应正文；错误统一经 `agent/engine.py::map_provider_error` 转成 `OmniError`。
- 只有声明支持视觉的模型才发送图片；`convert_history` 会按候选能力降级为文字占位。
- 不把 API Key 拼进 URL；pydantic-ai 的 provider 各自处理鉴权头。

> 大多数厂商只要在 YAML 里登记 `type: openai` + `base_url` 即可，无需写代码。

## 5. 添加一个 Agent 工具

1. 在 `src/omninexus/agent/tools.py` 加一个普通函数，参数用类型标注（Pydantic AI 会据此生成
   JSON Schema 并校验模型参数）。
2. 所有路径操作必须走 `Workspace`，不要直接用 `Path`；子进程必须走 `Workspace.run_argv`。
3. 在 `AgentTools.tools()` 返回列表里登记，并受 `agent.tools` 开关控制。
4. 补测试：正常路径、路径逃逸拒绝、超时/输出截断。
5. 更新 [Agent Loop](agent-loop.md) 的工具表。

工具返回值就是回灌给模型的 observation，应简短、可读、可截断。

## 6. 添加 API

- 路由集中在 `api/routes.py`，请求模型放 `api/schemas.py`。
- 对外错误使用 `OmniError`/`NotFoundError`；禁止抛裸 `HTTPException` 携带内部信息。
- 写接口必须使用 `ConfigDict(extra="forbid")`。
- 对会话/提示词 ID 先校验存在，再写。
- 变更下游事件字段 = 契约变更，必须更新 `docs/api.md` 与测试。

## 7. 数据库变更

当前 schema version = 1（`storage/database.py::SCHEMA_VERSION`）。

新增字段/表：

1. 改 `_SCHEMA` 的 `CREATE TABLE IF NOT EXISTS`。
2. 增加 `schema_meta.version` 迁移函数；不要只改 schema 假设旧库自动升级。
3. 更新 `_row_to_*`、导出与测试。
4. 严禁在迁移中删除对话数据；需要时先导出。

`Database.transaction(fn)` 提供 `BEGIN IMMEDIATE`；异步代码不要在事务函数里 `await`。

## 8. 提示词安全

- 使用 `SandboxedEnvironment`，不加载模板文件。
- 变量只接受 JSON 标量；长度受限。
- 未定义变量渲染为空，避免运行时中断；语法错误在保存时拒绝。
- 不要把用户输入拼进系统提示的“可信指令”区域；检索内容用独立块包裹。

## 9. 保留记忆接口

- 契约文件：`memory/contract.py`，版本 `1.0`。
- 默认 `memory.enabled: false`；禁止改为默认启用。
- 不得删除 `HttpMemoryPlugin`、`DisabledMemoryPlugin`、契约测试或文档。
- 修改契约需联合评审 + 版本升级；见 [memory-reserved.md](memory-reserved.md)。

## 10. 测试约定

- 所有厂商网络调用用 `httpx.MockTransport` 或 pydantic-ai 的 `FunctionModel`；CI 不调用真实 API。
- 测试不得依赖真实密钥；假密钥只用于验证脱敏。
- 故障注入优先覆盖：5xx、超时、429、空流、部分流、记忆超时、搜索失败。
- Agent 侧必须覆盖：工具循环、`tool` 帧、步数预算中断、沙箱路径逃逸拒绝、shell allowlist 拒绝。
- 每个修复应补回归测试；提交前跑：

```powershell
python -m pytest -q
python -m ruff check src tests
```

## 11. 发布清单（必须逐项）

- [ ] `pytest` 全绿，`ruff` 全绿。
- [ ] 无真实密钥：`git grep` 检查 `sk-`、`AIza`、`Bearer`、`api_key=`。
- [ ] 无 `.env`；只有 `.env.example` 且为空/占位。
- [ ] 无 `data/`、`*.db`、`*.log`、`*.jsonl`、导出文件、截图临时文件。
- [ ] 无测试人格/临时人设：搜索 `persona`、`测试人格`、`test persona`、自定义临时配置名；
      `DELETE` 测试配置并清理 `data/`。
- [ ] 无调试输出/临时脚本/`print` 调试；测试脚本保留在 `tests/`。
- [ ] 日志不输出密钥/完整对话（运行脱敏测试）。
- [ ] 文档与默认配置一致（模型数、默认禁用项、Reserved 标注）。
- [ ] `src/omninexus/resources/default.yaml` 与 `config/default.yaml` 一致。
- [ ] Docker 构建成功；容器内只发布回环端口；数据卷可写。
- [ ] 版本号、README、CHANGELOG（如有）一致。
- [ ] GitHub 仓库默认分支、CI、License、README 徽章正确。

## 12. 代码风格

- Ruff：`E,F,I,UP,B,SIM,ASYNC`，行宽 100。
- 类型标注优先；公开函数写明契约/失败行为。
- 不记录敏感内容；日志使用 `get_logger(...)`。
- 配置默认值围绕“本地、安全、默认关闭增强能力”。
- 不在仓库中放生成物（构建、缓存、日志、数据库）。

## 13. 提交规范建议

```text
feat(agent): add shell tool with allowlist
feat(agent): stream tool frames
fix(api): encode non-ASCII export filenames
docs(memory): freeze reserved contract v1.0
test(degradation): cover memory timeout zero-impact
```

一次提交聚焦一件事；契约变更单独提交并在说明中标注版本。
