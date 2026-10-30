"""Agent 工具的工作区沙箱。

安全边界（v1 现实版）：
- 文件工具只允许读写工作区根（{data_dir}/agent-workspace）内路径，拒绝 `..` 逃逸与符号链接逃逸；
- Shell/代码执行以工作区为 cwd，超时 + 输出上限 + 命令 allowlist；
- 信任边界 = 模型层进程权限（与 manifest permissions 一致），不是容器级沙箱。
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_WINDOWS = sys.platform == "win32"


class SandboxError(Exception):
    """工具安全边界被触发（路径逃逸、命令不在 allowlist 等）。"""


def _windows_command_line(argv: list[str]) -> str:
    """Windows 下用 shlex.quote 之外的规则拼一行，避免参数被二次解析。"""
    quoted: list[str] = []
    for arg in argv:
        if not arg or any(ch.isspace() for ch in arg):
            quoted.append(f'"{arg.replace(chr(34), chr(39))}"')
        else:
            quoted.append(arg)
    return " ".join(quoted)


class Workspace:
    """锁定在工作区根内的文件与子进程执行器。"""

    def __init__(
        self,
        root: Path,
        *,
        timeout_s: float = 30.0,
        max_output_chars: int = 20_000,
    ) -> None:
        self.root = root.resolve()
        self.timeout_s = timeout_s
        self.max_output_chars = max_output_chars
        self.root.mkdir(parents=True, exist_ok=True)

    # -- 路径 -------------------------------------------------------------

    def resolve(self, rel: str) -> Path:
        """把工具给的相对/绝对路径解析为工作区内的绝对路径；越界即拒绝。"""
        if not isinstance(rel, str) or not rel.strip():
            raise SandboxError("path 不能为空")
        candidate = Path(rel)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise SandboxError(f"路径越出工作区：{rel}")
        return candidate

    def read(self, rel: str, *, max_chars: int | None = None) -> str:
        path = self.resolve(rel)
        if not path.is_file():
            raise SandboxError(f"文件不存在或不是文件：{rel}")
        data = path.read_bytes()
        limit = max_chars or self.max_output_chars
        text = data[:limit].decode("utf-8", errors="replace")
        if len(data) > limit:
            text += f"\n…（输出已截断，超过 {limit} 字节）"
        return text

    def write(self, rel: str, content: str) -> str:
        path = self.resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)
        return f"已写入 {path.relative_to(self.root)}（{path.stat().st_size} 字节）"

    def list(self, rel: str = ".") -> str:
        path = self.resolve(rel)
        if not path.is_dir():
            raise SandboxError(f"不是目录：{rel}")
        lines: list[str] = []
        for child in sorted(path.iterdir()):
            if child.is_dir():
                lines.append(f"{child.name}/")
            else:
                lines.append(child.name)
        return "\n".join(lines) if lines else "(空目录)"

    # -- 子进程 -------------------------------------------------------------

    async def run_argv(self, argv: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
        """运行 argv（首个元素为程序名），cwd 默认工作区。

        返回 ``{stdout, stderr, code, duration_ms}``，输出超上限即截断。
        """
        creation_flags = subprocess.CREATE_NO_WINDOW if _WINDOWS else 0
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd or self.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=os.environ.copy(),
            creationflags=creation_flags,
        )
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        started = time.monotonic()

        async def _read(stream: asyncio.StreamReader, sink: list[bytes], budget: int) -> int:
            used = 0
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    return used
                take = min(len(chunk), budget - used)
                if take > 0:
                    sink.append(chunk[:take])
                    used += take
                if used >= budget:
                    break
            return used

        try:
            async with asyncio.timeout(self.timeout_s):
                read_stdout = asyncio.create_task(
                    _read(proc.stdout, stdout_chunks, self.max_output_chars)
                )
                read_stderr = asyncio.create_task(
                    _read(proc.stderr, stderr_chunks, self.max_output_chars)
                )
                await asyncio.gather(read_stdout, read_stderr)
                code = await proc.wait()
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "stdout": b"".join(stdout_chunks).decode("utf-8", errors="replace"),
                "stderr": f"命令超过 {self.timeout_s:g} 秒被终止",
                "code": -1,
                "duration_ms": int(self.timeout_s * 1000),
            }
        except Exception:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise

        return {
            "stdout": b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            "stderr": b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            "code": code,
            "duration_ms": int((time.monotonic() - started) * 1000),
        }

    # -- Shell allowlist -----------------------------------------------------

    def check_command(
        self, command: str, *, allow: list[str], git_allow: list[str]
    ) -> tuple[str, list[str]]:
        """校验 shell 命令；返回 (程序名, 参数列表)。不在 allowlist 直接拒绝。"""
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            raise SandboxError(f"无法解析命令：{exc}") from exc
        if not argv:
            raise SandboxError("命令为空")
        prog = argv[0].lower()
        if _WINDOWS and prog in {"python", "python3"}:
            prog = "python"
        if prog not in allow:
            raise SandboxError(f"命令 {argv[0]!r} 不在允许列表内，已拒绝")
        if prog == "git" and argv[1:2] and argv[1].lower() not in git_allow:
            raise SandboxError(f"git 子命令 {argv[1]!r} 不在允许列表内，已拒绝")
        return argv[0], argv[1:]
