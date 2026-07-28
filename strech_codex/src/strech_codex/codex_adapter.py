"""Adapters and diagnostics for Codex SDK live-mode integration.

Adapted from ``robot_project/codex_adapter.py``.  The offline demo is the
canonical, testable path; this module enables live Codex SDK orchestration
when the SDK and API key are available.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import stretch_mujoco

from .episode_logger import DEFAULT_LOG_DIR, EpisodeLogger

# Optional Codex SDK imports
try:
    from openai_codex import AsyncCodex, CodexConfig, Sandbox  # type: ignore
except ImportError:  # pragma: no cover
    AsyncCodex = CodexConfig = Sandbox = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SdkAvailability:
    mcp: bool
    openai_codex_sdk: bool
    has_openai_api_key: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "mcp": self.mcp,
            "openai_codex_sdk": self.openai_codex_sdk,
            "has_openai_api_key": self.has_openai_api_key,
        }


@dataclass(frozen=True)
class LiveRunResult:
    task: str
    final_response: str
    raw_turn: Any | None = None
    episode_path: str | None = None
    video_path: str | None = None


def inspect_environment() -> SdkAvailability:
    return SdkAvailability(
        mcp=_has_module("mcp"),
        openai_codex_sdk=all(item is not None for item in (AsyncCodex, CodexConfig, Sandbox)),
        has_openai_api_key=bool(
            os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")
        ),
    )


def live_mode_unavailable_message() -> str:
    availability = inspect_environment()
    installed = ", ".join(f"{name}={value}" for name, value in availability.as_dict().items())
    return (
        "live mode 无法启动。当前项目会直接使用 Codex SDK；"
        "如果 SDK 或所需类型无法导入，相关对象会自动退化为 None。\n"
        f"当前环境检测结果: {installed}\n"
        "请先使用 offline mode 验证 MCP/工具编排流程；如果要接入真实 Codex SDK，"
        "请安装对应 SDK 并设置 API key，然后再运行 live mode。"
    )


async def run_live(
    task: str,
    *,
    working_directory: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    on_event: Callable[[Any], None] | None = None,
) -> LiveRunResult:
    """Run the task through the installed Codex SDK."""
    if AsyncCodex is None or CodexConfig is None or Sandbox is None:
        raise RuntimeError(live_mode_unavailable_message())

    project_root = Path(stretch_mujoco.__file__).resolve().parents[1]
    package_root = Path(__file__).resolve().parents[2]
    server_name = "mcp_servers.strech-codex"
    episode = EpisodeLogger(task, DEFAULT_LOG_DIR)
    mcp_env = {
        "STRECH_CODEX_EPISODE_DIR": str(episode.path.parent),
        "STRECH_CODEX_VIDEO_PATH": str(episode.video_path),
    }
    for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "EGL_DEVICE_ID"):
        if value := os.environ.get(name):
            mcp_env[name] = value
    overrides = (
        f"{server_name}.command={json.dumps(os.sys.executable)}",
        f"{server_name}.args={json.dumps(['-m', 'strech_codex.mcp_server'])}",
        f"{server_name}.cwd={json.dumps(str(package_root))}",
        *(f"{server_name}.env.{name}={json.dumps(value)}" for name, value in mcp_env.items()),
        f"{server_name}.required=true",
        f"{server_name}.startup_timeout_sec=30",
        f"{server_name}.tool_timeout_sec=300",
    )

    env = dict(os.environ)
    selected_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    if selected_base_url:
        env["OPENAI_BASE_URL"] = selected_base_url

    env.update(mcp_env)

    config = CodexConfig(
        codex_bin=os.environ.get("CODEX_PATH_OVERRIDE"),
        config_overrides=overrides,
        cwd=str(project_root),
        env=env,
    )
    selected_model = model or os.environ.get("CODEX_MODEL") or os.environ.get("OPENAI_MODEL")
    try:
        async with AsyncCodex(config=config) as codex:
            thread = await codex.thread_start(
                model=selected_model,
                cwd=working_directory or str(project_root),
                sandbox=Sandbox.workspace_write,
            )
            prompt = build_live_prompt(task, project_root)
            result = await _run_live_streamed(thread, prompt, task, on_event, episode)
            episode.finish(result.final_response)
            return LiveRunResult(
                task=task,
                final_response=result.final_response,
                raw_turn=None,
                episode_path=str(episode.path),
                video_path=(
                    str(episode.video_path)
                    if episode.video_path.is_file() and episode.video_path.stat().st_size > 0
                    else None
                ),
            )
    except Exception as exc:
        episode.finish(error=exc)
        raise


def run_live_sync(
    task: str,
    *,
    working_directory: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    on_event: Callable[[Any], None] | None = None,
) -> LiveRunResult:
    """Synchronous wrapper around ``run_live``."""
    return asyncio.run(
        run_live(
            task,
            working_directory=working_directory,
            base_url=base_url,
            model=model,
            on_event=on_event,
        )
    )


async def _run_live_streamed(
    thread: Any,
    prompt: str,
    task: str,
    on_event: Callable[[Any], None] | None,
    episode: EpisodeLogger,
) -> LiveRunResult:
    """Consume SDK events while retaining the final response."""
    turn = await thread.turn(prompt)
    final_parts: list[str] = []
    async for event in turn.stream():
        episode.record(event)
        if on_event is not None:
            on_event(event)
        if getattr(event, "method", None) == "item/completed":
            item = getattr(getattr(event, "payload", None), "item", None)
            root = getattr(item, "root", None)
            if getattr(root, "type", None) == "agentMessage":
                final_parts.append(str(getattr(root, "text", "")))

    return LiveRunResult(task=task, final_response="".join(final_parts).strip(), raw_turn=None)


def format_live_event(event: Any) -> str | None:
    """Format a Codex SDK stream event for terminal output."""
    method = getattr(event, "method", None)
    if method == "turn/started":
        return "Codex 开始执行任务"
    if method == "item/started":
        item = getattr(getattr(event, "payload", None), "item", None)
        root = getattr(item, "root", None)
        if getattr(root, "type", None) == "mcpToolCall":
            return (
                f"START {getattr(root, 'server', '?')}.{getattr(root, 'tool', '?')}"
                f"({_format_arguments(getattr(root, 'arguments', {}))})"
            )
        return None
    if method == "item/completed":
        item = getattr(getattr(event, "payload", None), "item", None)
        root = getattr(item, "root", None)
        if getattr(root, "type", None) == "mcpToolCall":
            status = getattr(getattr(root, "status", None), "value", "completed")
            duration = getattr(root, "duration_ms", None)
            elapsed = f" {duration / 1000:.3f}s" if duration is not None else ""
            error = getattr(getattr(root, "error", None), "message", None)
            detail = f"ERROR {error}" if error else _format_mcp_result(root).strip()
            return (
                f"{status.upper()} {getattr(root, 'server', '?')}.{getattr(root, 'tool', '?')}"
                f"{elapsed}: {detail}"
            )
        return None
    if method == "turn/completed":
        turn = getattr(getattr(event, "payload", None), "turn", None)
        status = getattr(getattr(turn, "status", None), "value", "completed")
        return f"Codex 执行结束: {status}"

    event_type = getattr(event, "type", "unknown")
    if event_type == "thread.started":
        return f"🧵 任务线程: {getattr(event, 'thread_id', 'unknown')}"
    if event_type == "turn.started":
        return "🚀 Codex 开始执行任务"
    if event_type == "turn.completed":
        usage = getattr(event, "usage", None)
        if usage is None:
            return "✅ Codex 执行完成"
        return (
            "📊 Token 用量: "
            f"(input={getattr(usage, 'input_tokens', '?')}, "
            f"cached={getattr(usage, 'cached_input_tokens', '?')}, "
            f"output={getattr(usage, 'output_tokens', '?')})"
        )
    if event_type == "turn.failed":
        error = getattr(event, "error", None)
        return f"❌ 执行失败: {getattr(error, 'message', error)}"
    if event_type == "error":
        return f"❌ 线程错误: {getattr(event, 'message', 'unknown error')}"

    item = getattr(event, "item", None)
    if item is None:
        return None
    item_type = getattr(item, "type", "unknown")

    if event_type == "item.started":
        if item_type == "mcp_tool_call":
            return (
                f"▶ {getattr(item, 'tool', '?')}"
                f"({_format_arguments(getattr(item, 'arguments', {}))})"
            )
        if item_type == "command_execution":
            return f"💻 命令: {getattr(item, 'command', '')}"
        return None

    if event_type != "item.completed":
        return None

    if item_type == "reasoning":
        return None
    if item_type == "mcp_tool_call":
        error = getattr(item, "error", None)
        if error is not None:
            return f"  ❌ {getattr(error, 'message', error)}"
        return _format_mcp_result(item)
    if item_type == "agent_message":
        return None
    if item_type == "error":
        return f"❌ 项目错误: {getattr(item, 'message', 'unknown error')}"
    return f"• {item_type}: {_json_text(item)}"


def _json_text(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif isinstance(value, list):
        value = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in value
        ]
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _format_arguments(value: Any) -> str:
    if not isinstance(value, dict):
        return _json_text(value)
    return ", ".join(f"{key}={_compact_value(val)}" for key, val in value.items())


def _compact_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    return _json_text(value)


def _format_mcp_result(item: Any) -> str:
    payload = _extract_mcp_payload(getattr(item, "result", None))
    if isinstance(payload, dict):
        message = payload.get("message")
        success = payload.get("success")
        if message:
            icon = "✅" if success is not False else "❌"
            out = f"  {icon} {message}"
            wps = payload.get("waypoints")
            if wps:
                out += f" ({len(wps)} waypoints)"
            return out
    compact = _json_text(payload)
    if len(compact) > 240:
        compact = compact[:237] + "..."
    return f"  ✅ {compact}"


def _extract_mcp_payload(result: Any) -> Any:
    if result is None:
        return None
    if hasattr(result, "model_dump"):
        result = result.model_dump(mode="json")
    if not isinstance(result, dict):
        return result
    structured = result.get("structured_content")
    if isinstance(structured, dict) and "result" in structured:
        return _parse_json_value(structured["result"])
    content = result.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                return _parse_json_value(block["text"])
    return result


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def build_live_prompt(task: str, project_root: Path) -> str:
    """Build the system prompt that tells Codex about available tools."""
    return (
        "你现在在一个 MuJoCo 机器人仿真项目中，需要通过 MCP 工具完成任务。\n"
        f"项目根目录: {project_root}\n"
        "可用工具由 strech-codex MCP server (strech-codex-server) 提供，工具语义如下：\n\n"
        "=== 导航工具 ===\n"
        "- nav_build_grid(scene_xml?, bounds?, resolution?, agent_radius?) — 构建占据栅格\n"
        "- nav_get_grid_info() — 获取栅格信息\n"
        "- nav_is_free(x, y) — 检查点是否可通行\n"
        "- nav_astar(start_x, start_y, goal_x, goal_y, smoothing?) — A* 路径规划\n"
        "- nav_fmm(start_x, start_y, goal_x, goal_y, smoothing?) — FMM 路径规划\n"
        "- nav_execute_path(waypoints, timeout?, position_tolerance?, ...) — 让当前机器人闭环跟踪路径\n"
        "- nav_fbe_init(...) — 初始化自主探索 (FBE)\n"
        "- nav_fbe_step(robot_x, robot_y, robot_yaw) — FBE 探索一步\n"
        "- nav_fbe_path() — 获取当前 FBE 路径\n"
        "- nav_vlfm_init(instruction, vlm_type?, ...) — 初始化 VLFM 视觉语言探索\n"
        "- nav_vlfm_step(robot_x, robot_y, robot_yaw) — VLFM 探索一步\n"
        "- nav_vlfm_inject_observation(rgb_image_path, depth_image_path, ...) — 注入观测\n\n"
        "=== 机器人工具 ===\n"
        "- robot_init(robot_type, scene_xml?, headless?) — 启动 robot (stretch3/google_robot/tidybot)\n"
        "- robot_stop() — 停止\n"
        "- robot_move_to(actuator_name, position) — 关节绝对移动\n"
        "- robot_move_by(actuator_name, delta) — 关节相对移动\n"
        "- robot_set_base_velocity(v_linear, omega, v_lateral?) — 底盘速度控制\n"
        "- robot_home() / robot_stow() — 回到预设位姿\n"
        "- robot_wait_until_at_setpoint(actuator_name, timeout?, position_tolerance?) — 等待到达\n"
        "- robot_get_status() — 拉取关节状态\n"
        "- robot_get_base_pose() — 获取 (x,y,theta) 底盘位姿\n"
        "- robot_get_ee_pose() — 获取末端 4×4 位姿\n"
        "- robot_get_camera_data(camera_name?) — 获取相机数据或列出相机\n"
        "- robot_attach_object(object_id) — 抓取物体\n"
        "- robot_release_object() — 释放物体\n"
        "- robot_list_actuators() — 列出执行器\n"
        "- robot_list_cameras() — 列出相机\n\n"
        "=== 场景工具 ===\n"
        "- scene_list_objects() — 列出可移动物体\n"
        "- scene_get_object_pose(object_name) — 获取物体位姿\n"
        "- scene_add_world_frame(x, y, z) — 标记坐标系\n\n"
        "执行规则：\n"
        "1. 只使用 strech-codex-server MCP 工具完成任务；不要读取、搜索或修改项目源码。\n"
        "2. 导航类任务必须先调用 nav_build_grid 再调用 nav_astar 或 nav_fmm。\n"
        "3. 机器人操作任务必须先调用 robot_init 再调用其他 robot_* 工具。\n"
        "4. 用户要求机器人实际导航/移动到目标时，将规划器返回的完整 waypoints 传给 nav_execute_path；仅要求规划、比较路径时不要执行。\n"
        "5. 用户给出明确导航起点时，robot_init 必须用 start_translation=[x,y,0] 使真实底盘与规划起点一致。\n"
        "6. 工具调用失败后不要声称任务已经完成，应报告具体错误。\n"
        "7. 完成后调用 robot_stop 清理资源。\n"
        "任务目标：\n"
        f"{task}\n\n"
        "请按顺序规划和执行这些工具调用，最后返回一句简短中文总结。"
    )


def extract_final_response(turn: Any) -> str:
    """Pull a readable final response from the Codex turn object."""
    for attr in ("final_response", "response", "output_text"):
        value = getattr(turn, attr, None)
        if value:
            return str(value)

    items = getattr(turn, "items", None)
    if items:
        parts: list[str] = []
        for item in items:
            if getattr(item, "tool_name", None):
                parts.append(f"{item.tool_name}({getattr(item, 'arguments', {})})")
            elif getattr(item, "command", None):
                parts.append(str(getattr(item, "command")))
        if parts:
            return "\n".join(parts)
    return str(turn)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None
