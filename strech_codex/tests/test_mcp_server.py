"""MCP stdio transport regression tests."""

from strech_codex import mcp_server


def test_dispatch_routes_simulator_stdout_to_stderr(monkeypatch, capfd):
    monkeypatch.setenv("STRECH_CODEX_MCP_STDIO", "1")
    monkeypatch.setitem(
        mcp_server.ALL_TOOLS,
        "noisy_tool",
        lambda: (print("simulator output") or {"success": True}),
    )

    assert mcp_server.dispatch_tool("noisy_tool")["success"]
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "simulator output" in captured.err
