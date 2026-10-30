"""Agent 内置工具：文件读写（锁工作区）、Shell、Python 代码执行。

安全设计（见 sandbox.py）：
- 文件工具路径必须落在工作区根内；
- Shell 受命令 allowlist 约束（git 再限定子命令）；
- Python 代码以工作区为 cwd 独立进程执行，带超时与输出上限。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic_ai import Tool

from omninexus.agent.sandbox import SandboxError, Workspace
from omninexus.config import AgentConfig


def _format_run(result: dict) -> str:
    parts: list[str] = []
    if result.get("stdout"):
        parts.append("STDOUT:\n" + result["stdout"].rstrip())
    if result.get("stderr"):
        parts.append("STDERR:\n" + result["stderr"].rstrip())
    body = "\n\n".join(parts) if parts else "(无输出)"
    return f"exit={result['code']} {body}"


class AgentTools:
    """持有工作区的工具集合；构造后通过 ``tools()`` 生成 Pydantic AI 工具。"""

    def __init__(self, config: AgentConfig, data_dir: Path) -> None:
        root = Path(data_dir) / config.workspace_dir
        self.workspace = Workspace(
            root,
            timeout_s=config.timeout_s,
            max_output_chars=config.max_output_chars,
        )
        self.config = config

    # -- 文件 ---------------------------------------------------------------

    async def file_read(self, path: str) -> str:
        """读取工作区内文件的内容（相对路径或绝对路径；绝对路径必须位于工作区内）。"""
        return self.workspace.read(path)

    async def file_write(self, path: str, content: str) -> str:
        """在工作区内写入/覆盖一个文本文件；必要时自动创建父目录。"""
        return self.workspace.write(path, content)

    async def file_list(self, path: str = ".") -> str:
        """列出工作区内目录的条目（目录名带 / 后缀）。"""
        return self.workspace.list(path)

    # -- Shell --------------------------------------------------------------

    async def shell(self, command: str) -> str:
        """在工作区目录内执行一条允许的命令。返回值包含退出码与输出。"""
        argv0, rest = self.workspace.check_command(
            command, allow=self.config.shell_allow, git_allow=self.config.git_allow
        )
        if argv0 == "python" or argv0 == "py":
            argv0 = sys.executable
        result = await self.workspace.run_argv([argv0, *rest])
        return _format_run(result)

    async def python_code(self, code: str) -> str:
        """在工作区目录内运行一段 Python 代码（写入临时文件后执行），返回输出与退出码。"""
        if not code.strip():
            return "exit=0 (空代码)"
        tmp = self.workspace.root / ".agent_code_tmp.py"
        tmp.write_text(code, encoding="utf-8")
        result = await self.workspace.run_argv([sys.executable, str(tmp)])
        return _format_run(result)

    # -- 组装 ---------------------------------------------------------------

    def tools(self) -> list[Tool]:
        tools: list[Tool] = []
        if self.config.tools.file:
            tools.append(Tool(name="file_read", function=self.file_read))
            tools.append(Tool(name="file_write", function=self.file_write))
            tools.append(Tool(name="file_list", function=self.file_list))
        if self.config.tools.shell:
            tools.append(Tool(name="shell", function=self.shell))
        if self.config.tools.python:
            tools.append(Tool(name="python_code", function=self.python_code))
        return tools

    async def close(self) -> None:  # 预留：无持有性资源
        return None


async def _self_check() -> None:
    """快速冒烟：验证沙箱拒绝逃逸路径。"""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        ws = Workspace(Path(tmp) / "ws")
        try:
            ws.resolve("../escape")
            raise AssertionError("should have raised")
        except SandboxError:
            pass
        ws.write("a/b.txt", "hello")
        assert ws.read("a/b.txt") == "hello"


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_self_check())
    print("sandbox self-check ok")
