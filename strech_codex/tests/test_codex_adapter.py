"""Regression test for live-mode MCP configuration."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from strech_codex import codex_adapter


def test_live_mode_registers_mcp_server(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setenv("MUJOCO_GL", "egl")

    class FakeConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    class FakeSandbox:
        workspace_write = "workspace-write"

    class FakeThread:
        async def turn(self, prompt):
            return FakeTurn()

    class FakeTurn:
        async def stream(self):
            tool = SimpleNamespace(
                type="mcpToolCall",
                id="call-1",
                server="strech-codex",
                tool="robot_get_status",
                arguments={},
                duration_ms=12,
                status=SimpleNamespace(value="completed"),
                result=None,
                error=None,
            )
            yield SimpleNamespace(
                method="item/started",
                payload=SimpleNamespace(
                    item=SimpleNamespace(root=tool), started_at_ms=1_700_000_000_000
                ),
            )
            yield SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(
                    item=SimpleNamespace(root=tool), completed_at_ms=1_700_000_000_012
                ),
            )
            yield SimpleNamespace(
                method="item/completed",
                payload=SimpleNamespace(
                    item=SimpleNamespace(root=SimpleNamespace(type="agentMessage", text="done")),
                    completed_at_ms=1_700_000_000_013,
                ),
            )

    class FakeCodex:
        def __init__(self, config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def thread_start(self, **kwargs):
            captured["thread"] = kwargs
            return FakeThread()

    monkeypatch.setattr(codex_adapter, "AsyncCodex", FakeCodex)
    monkeypatch.setattr(codex_adapter, "CodexConfig", FakeConfig)
    monkeypatch.setattr(codex_adapter, "Sandbox", FakeSandbox)
    monkeypatch.setattr(codex_adapter, "DEFAULT_LOG_DIR", tmp_path)

    result = asyncio.run(codex_adapter.run_live("test"))
    overrides = captured["config"]["config_overrides"]
    assert result.final_response == "done"
    assert result.episode_path is not None
    episode = json.loads(Path(result.episode_path).read_text(encoding="utf-8"))
    assert episode["status"] == "completed"
    assert episode["steps"][0]["tool"] == "robot_get_status"
    assert episode["steps"][0]["duration_ms"] == 12
    assert any("strech-codex" in value and ".command=" in value for value in overrides)
    assert any(".env.STRECH_CODEX_VIDEO_PATH=" in value for value in overrides)
    assert any(".env.STRECH_CODEX_EPISODE_DIR=" in value for value in overrides)
    assert any('.env.MUJOCO_GL="egl"' in value for value in overrides)
    assert captured["thread"]["cwd"].endswith("stretch_mujoco")
