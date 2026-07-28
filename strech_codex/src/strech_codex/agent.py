"""Offline agent loop for robot tool orchestration.

Parses natural-language commands (Chinese + English) into tool-call
sequences and executes them through the MCP dispatch layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import mcp_server
from .world import state as _world


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult:
    call: ToolCall
    result: dict[str, Any]


@dataclass
class AgentRunResult:
    task: str
    mode: str
    planned_calls: list[ToolCall]
    tool_results: list[ToolCallResult]
    final_response: str


# ---------------------------------------------------------------------------
# Offline task parser
# ---------------------------------------------------------------------------


def plan_offline(task: str) -> list[ToolCall]:
    """Plan a deterministic tool sequence from a natural-language task.

    Recognises these command patterns (Chinese + English):
    - 导航/去/到 + room → nav_astar on pre-built grid
    - 构建地图/加载场景 → nav_build_grid
    - 探索 → nav_fbe_init + nav_fbe_step
    - 视觉探索/VLFM → nav_vlfm_init
    - 启动 + robot → robot_init
    - 移动 + joint → robot_move_to
    - 抓取/释放 → robot_attach_object / robot_release_object
    - 查询状态 → robot_get_status
    """
    calls: list[ToolCall] = []

    _has_explicit_grid = any(kw in task for kw in ["构建地图", "加载场景", "build grid", "load scene"])
    if _has_explicit_grid:
        calls.append(ToolCall("nav_build_grid", {"resolution": 0.08, "agent_radius": 0.25}))

    # Auto-inject grid build if a scene path is available and navigation/exporation is requested
    _needs_nav = any(
        kw in task
        for kw in ["导航", "navigate", "路径", "path", "探索", "explore", "扫描", "scan"]
    )
    _has_grid = _world.has_nav()
    if _needs_nav and not _has_grid and not _has_explicit_grid and _world.get_scene_path():
        calls.insert(0, ToolCall("nav_build_grid", {"resolution": 0.08, "agent_radius": 0.25}))

    # --- Robot init ---
    robot_type = _world.get_robot_type()
    if "stretch" in task.lower() or "拉伸" in task:
        robot_type = "stretch3"
    elif "google" in task.lower() or "谷歌" in task:
        robot_type = "google_robot"
    elif "tidybot" in task.lower() or "整理" in task:
        robot_type = "tidybot"

    if any(kw in task for kw in ["启动", "初始化", "start robot", "init robot"]):
        if robot_type is None:
            robot_type = "stretch3"  # default
        calls.append(ToolCall("robot_init", {"robot_type": robot_type, "headless": True}))

    # --- Navigation ---
    nav_match = _parse_navigation(task)
    if nav_match:
        calls.append(nav_match)

    # --- Exploration ---
    if any(kw in task for kw in ["探索", "扫描", "explore", "scan"]):
        if "vlfm" in task.lower() or "视觉" in task or "vision" in task.lower():
            instruction = _extract_instruction(task)
            calls.append(ToolCall("nav_vlfm_init", {"instruction": instruction, "vlm_type": "clip"}))
        else:
            calls.append(ToolCall("nav_fbe_init", {}))

    # --- Robot movement ---
    move_calls = _parse_movement(task)
    calls.extend(move_calls)

    # --- Grasping ---
    if any(kw in task for kw in ["抓", "拿", "捡", "grasp", "grab", "pick"]):
        obj = _extract_object(task)
        if obj:
            calls.append(ToolCall("robot_attach_object", {"object_id": obj}))

    if any(kw in task for kw in ["放", "释放", "松", "release", "drop"]):
        calls.append(ToolCall("robot_release_object", {}))

    # --- Status query ---
    if any(kw in task for kw in ["状态", "位置", "在哪", "status", "pose", "where"]):
        calls.append(ToolCall("robot_get_status", {}))
        calls.append(ToolCall("robot_get_base_pose", {}))

    # --- Fallback: if no calls generated, try to interpret as a simple command ---
    if not calls:
        calls = _fallback_plan(task, robot_type)

    return calls


def _parse_navigation(task: str) -> ToolCall | None:
    """Extract a navigation command from the task text."""
    # Pattern: 导航到 (x, y) / 去 (x, y) / 从A到B
    import re

    # "从 (sx, sy) 到 (gx, gy)" or "from (sx, sy) to (gx, gy)"
    # Also handles "从 (sx, sy) 导航到 (gx, gy)"
    m = re.search(
        rf"(?:从|from)\s*[(（]?\s*({_NUMBER})\s*[,，]\s*({_NUMBER})\s*[)）]?\s*"
        rf"(?:.*?(?:到|to))\s*[(（]?\s*({_NUMBER})\s*[,，]\s*({_NUMBER})\s*[)）]?",
        task,
    )
    if m:
        return ToolCall(
            "nav_astar",
            {
                "start_x": float(m.group(1)),
                "start_y": float(m.group(2)),
                "goal_x": float(m.group(3)),
                "goal_y": float(m.group(4)),
            },
        )

    # "导航到 (gx, gy)" or "navigate to (gx, gy)"
    m = re.search(
        rf"(?:导航到|navigate to|go to)\s*[(（]?\s*({_NUMBER})\s*[,，]\s*({_NUMBER})\s*[)）]?",
        task,
    )
    if m:
        return ToolCall(
            "nav_astar",
            {
                "start_x": 0.0,
                "start_y": 0.0,
                "goal_x": float(m.group(1)),
                "goal_y": float(m.group(2)),
            },
        )

    return None


def _parse_movement(task: str) -> list[ToolCall]:
    """Extract robot movement commands."""
    import re

    calls: list[ToolCall] = []

    # "移动 X 到 Y" / "move X to Y"
    for m in re.finditer(
        rf"(?:移动|move)\s*(\S+)\s*(?:到|to)\s*({_NUMBER})",
        task,
    ):
        calls.append(
            ToolCall("robot_move_to", {"actuator_name": m.group(1), "position": float(m.group(2))})
        )

    # "前进" / "后退" / "左转" / "右转"
    if "前进" in task or "forward" in task.lower():
        calls.append(ToolCall("robot_set_base_velocity", {"v_linear": 0.3, "omega": 0.0}))
    if "后退" in task or "backward" in task.lower():
        calls.append(ToolCall("robot_set_base_velocity", {"v_linear": -0.3, "omega": 0.0}))
    if "左转" in task or "turn left" in task.lower():
        calls.append(ToolCall("robot_set_base_velocity", {"v_linear": 0.0, "omega": 0.5}))
    if "右转" in task or "turn right" in task.lower():
        calls.append(ToolCall("robot_set_base_velocity", {"v_linear": 0.0, "omega": -0.5}))
    if "停" in task or "stop" in task.lower():
        calls.append(ToolCall("robot_set_base_velocity", {"v_linear": 0.0, "omega": 0.0}))

    # "抬起手臂" / "放下手臂"
    if "抬起" in task or re.search(r"\blift (?:the )?arm\b", task, re.IGNORECASE):
        calls.append(ToolCall("robot_move_to", {"actuator_name": "lift", "position": 0.8}))
    if "放下" in task or "lower" in task.lower():
        calls.append(ToolCall("robot_move_to", {"actuator_name": "lift", "position": 0.2}))

    return calls


def _extract_object(task: str) -> str | None:
    """Extract an object name from a pickup/grasp command."""
    import re

    for pattern in [r"(?:抓|拿|捡|grasp|grab|pick)\s*(?:起)?(\S+)", r"(\S+?)(?:抓|拿|捡)"]:
        m = re.search(pattern, task)
        if m:
            return m.group(1)
    return None


def _extract_instruction(task: str) -> str:
    """Extract or construct a VLFM instruction from the task."""
    import re

    # Try to find a quoted string or description after "找/寻找/查找"
    m = re.search(r'["「](.+?)["」]', task)
    if m:
        return m.group(1)
    m = re.search(r"(?:找|寻找|查找|find|search for|look for)\s*(.+)", task)
    if m:
        return m.group(1).strip().rstrip("。.!！")
    return "Seems like there is a target object ahead."


def _fallback_plan(task: str, robot_type: str | None) -> list[ToolCall]:
    """Generate a minimal plan when no specific patterns match."""
    calls: list[ToolCall] = []
    if robot_type and not _world.has_sim():
        calls.append(ToolCall("robot_init", {"robot_type": robot_type, "headless": True}))
    calls.append(ToolCall("robot_get_status", {}))
    return calls


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_offline(task: str, *, reset: bool = False) -> AgentRunResult:
    """Run the offline planner and execute tools through the MCP dispatch layer."""
    if reset:
        _world.reset_world()

    calls = plan_offline(task)
    results: list[ToolCallResult] = []
    for call in calls:
        result = mcp_server.dispatch_tool(call.name, call.arguments)
        results.append(ToolCallResult(call=call, result=result))
        if not result.get("success", False):
            break

    final_response = _summarize(task, results)
    return AgentRunResult(
        task=task,
        mode="offline",
        planned_calls=calls,
        tool_results=results,
        final_response=final_response,
    )


def _summarize(task: str, results: list[ToolCallResult]) -> str:
    if not results:
        return "没有生成任何工具调用。"
    failed = next((item for item in results if not item.result.get("success", False)), None)
    if failed:
        return f"任务未完成: {failed.call.name} → {failed.result.get('message', '未知错误')}"
    return "已按计划完成任务，所有工具调用均成功。"


def format_run(result: AgentRunResult) -> str:
    """Format a run as terminal-friendly text."""
    lines = [f"📝 任务: {result.task}", "", "📋 调用的工具序列:"]
    for item in result.tool_results:
        lines.append(f"  ▶ {item.call.name}({_compact_args(item.call.arguments)})")
        msg = item.result.get("message", "")
        icon = "✅" if item.result.get("success") else "❌"
        lines.append(f"    {icon} {msg}")
        # Show waypoints if present
        if item.result.get("waypoints"):
            wps = item.result["waypoints"]
            if len(wps) <= 5:
                lines.append(f"    路径: {wps}")
            else:
                lines.append(f"    路径: [{wps[0]} ... {wps[-1]}] ({len(wps)} waypoints)")
    lines.extend(["", f"🏁 {result.final_response}"])
    return "\n".join(lines)


def _compact_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    items = []
    for k, v in args.items():
        if isinstance(v, float):
            items.append(f"{k}={v:.2f}")
        else:
            items.append(f"{k}={v}")
    return ", ".join(items)
