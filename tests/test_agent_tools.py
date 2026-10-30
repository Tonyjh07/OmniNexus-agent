"""Agent tools: workspace confinement, shell allowlist and code execution."""

from __future__ import annotations

import pytest

from omninexus.agent.sandbox import SandboxError, Workspace
from omninexus.agent.tools import AgentTools
from omninexus.config import AgentConfig, AgentToolsConfig


def _agent_tools(tmp_path, **overrides) -> AgentTools:
    config = AgentConfig(**overrides)
    return AgentTools(config, tmp_path)


@pytest.mark.asyncio
async def test_file_roundtrip_stays_inside_workspace(tmp_path) -> None:
    tools = _agent_tools(tmp_path)
    assert "已写入" in await tools.file_write("dir/note.txt", "hello")
    assert await tools.file_read("dir/note.txt") == "hello"
    assert "dir/" in await tools.file_list(".")
    assert "note.txt" in await tools.file_list("dir")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["../escape.txt", "../../etc/passwd"])
async def test_file_tools_reject_parent_escape(tmp_path, path: str) -> None:
    tools = _agent_tools(tmp_path)
    with pytest.raises(SandboxError):
        await tools.file_write(path, "nope")
    with pytest.raises(SandboxError):
        await tools.file_read(path)


@pytest.mark.asyncio
async def test_file_tools_reject_absolute_path_outside_workspace(tmp_path) -> None:
    tools = _agent_tools(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(SandboxError):
        await tools.file_read(str(outside))


def test_workspace_rejects_symlink_escape(tmp_path) -> None:
    root = tmp_path / "ws"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    root.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    workspace = Workspace(root)
    with pytest.raises(SandboxError):
        workspace.read("link/secret.txt")


@pytest.mark.asyncio
async def test_shell_allowlist_rejects_unknown_program(tmp_path) -> None:
    tools = _agent_tools(tmp_path)
    with pytest.raises(SandboxError):
        await tools.shell("rm -rf /")


@pytest.mark.asyncio
async def test_shell_rejects_disallowed_git_subcommand(tmp_path) -> None:
    tools = _agent_tools(tmp_path)
    with pytest.raises(SandboxError):
        await tools.shell("git push origin main")


@pytest.mark.asyncio
async def test_shell_runs_allowed_command_in_workspace(tmp_path) -> None:
    tools = _agent_tools(tmp_path)
    await tools.file_write("hello.txt", "hi")
    result = await tools.shell("dir")
    assert "exit=0" in result
    assert "hello.txt" in result


@pytest.mark.asyncio
async def test_python_code_executes_with_timeout_and_output(tmp_path) -> None:
    tools = _agent_tools(tmp_path)
    result = await tools.python_code("print(6 * 7)")
    assert "exit=0" in result
    assert "42" in result


@pytest.mark.asyncio
async def test_shell_timeout_is_reported(tmp_path) -> None:
    tools = _agent_tools(tmp_path, timeout_s=0.5)
    result = await tools.shell('python -c "import time; time.sleep(5)"')
    assert "exit=-1" in result
    assert "被终止" in result


@pytest.mark.asyncio
async def test_disabled_tools_are_not_exposed(tmp_path) -> None:
    tools = _agent_tools(
        tmp_path, tools=AgentToolsConfig(file=False, shell=False, python=False)
    )
    assert tools.tools() == []


@pytest.mark.asyncio
async def test_tool_output_is_truncated(tmp_path) -> None:
    tools = _agent_tools(tmp_path, max_output_chars=16)
    await tools.file_write("big.txt", "x" * 100)
    text = await tools.file_read("big.txt")
    assert "输出已截断" in text
