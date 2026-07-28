"""MCP server exposing navigation, robot, and scene tools.

Follows the ``robot_project/mcp_server.py`` pattern.  Tools are registered
on a ``FastMCP`` server with JSON-string return values.

Usage (stdio server)::

    python -m strech_codex.mcp_server

Usage (import for agent dispatch)::

    from strech_codex.mcp_server import dispatch_tool, tool_manifest
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from typing import Any

from strech_codex.evidence import STATE_CHANGING_TOOLS, capture, next_invocation
from strech_codex.tools.navigation import NAV_TOOL_FUNCTIONS
from strech_codex.tools.robot import ROBOT_TOOL_FUNCTIONS
from strech_codex.tools.scene import SCENE_TOOL_FUNCTIONS

# ---------------------------------------------------------------------------
# Aggregate tool registry
# ---------------------------------------------------------------------------

ALL_TOOLS = {
    **NAV_TOOL_FUNCTIONS,
    **ROBOT_TOOL_FUNCTIONS,
    **SCENE_TOOL_FUNCTIONS,
}


def dispatch_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch a tool call by name to the correct implementation.

    Returns a ``{"success": bool, ...}`` dict.
    """
    arguments = arguments or {}
    tool = ALL_TOOLS.get(name)
    if tool is None:
        return {"success": False, "message": f"Unknown tool: {name}"}
    with _tool_output_to_stderr():
        invocation = next_invocation() if name in STATE_CHANGING_TOOLS else None
        before = _capture_evidence(name, "start", invocation)
        try:
            result = tool(**arguments)
        except Exception as exc:
            result = {"success": False, "message": f"Tool '{name}' error: {exc}"}
        if invocation is not None:
            result["evidence"] = {
                "start": before,
                "end": _capture_evidence(name, "end", invocation),
            }
    return result


def _capture_evidence(name: str, phase: str, invocation: int | None) -> list[dict[str, Any]]:
    if invocation is None:
        return []
    try:
        return capture(name, phase, invocation)
    except Exception as exc:
        return [{"error": str(exc)}]


@contextmanager
def _tool_output_to_stderr():
    """Keep simulator prints away from the MCP stdio JSON-RPC channel."""
    if os.environ.get("STRECH_CODEX_MCP_STDIO") != "1":
        yield
        return
    sys.stdout.flush()
    saved_stdout = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_stdout, sys.stdout.fileno())
        os.close(saved_stdout)


# ---------------------------------------------------------------------------
# Tool manifest (for MCP SDK registration)
# ---------------------------------------------------------------------------


def tool_manifest() -> list[dict[str, Any]]:
    """Return the complete tool metadata exposed by the MCP server."""
    return [
        # -- Grid --
        {
            "name": "nav_build_grid",
            "description": "从 MuJoCo 场景构建导航占据栅格 / Build occupancy grid from MuJoCo scene for navigation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scene_xml": {"type": "string", "description": "Path to MuJoCo scene XML"},
                    "bounds": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "[x_min, x_max, y_min, y_max] walkable bounds",
                    },
                    "resolution": {
                        "type": "number",
                        "default": 0.08,
                        "description": "Metres per grid cell",
                    },
                    "agent_radius": {
                        "type": "number",
                        "default": 0.25,
                        "description": "Inflation radius around obstacles (m)",
                    },
                    "floor_geom_name": {"type": "string", "default": "office_floor"},
                    "require_collision": {"type": "boolean", "default": True},
                },
            },
        },
        {
            "name": "nav_get_grid_info",
            "description": "返回当前占据栅格信息 / Return current occupancy grid dimensions and free ratio.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "nav_is_free",
            "description": "检查世界坐标点是否可通行 / Check if a world point (x, y) is navigable.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["x", "y"],
            },
        },
        # -- Scene discovery --
        {
            "name": "nav_list_scenes",
            "description": "列出所有可发现的 MuJoCo 场景 XML / List all discoverable scene XML files. Use the returned 'name' directly as scene_xml in nav_build_grid.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        # -- A* --
        {
            "name": "nav_astar",
            "description": "A* 路径规划 / Plan a collision-free path using A* search from (start_x, start_y) to (goal_x, goal_y). Returns waypoints and path distance.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_x": {"type": "number"},
                    "start_y": {"type": "number"},
                    "goal_x": {"type": "number"},
                    "goal_y": {"type": "number"},
                    "smoothing": {"type": "boolean", "default": True},
                },
                "required": ["start_x", "start_y", "goal_x", "goal_y"],
            },
        },
        # -- FMM --
        {
            "name": "nav_fmm",
            "description": "FMM 路径规划 / Plan a path using Fast Marching Method from (start_x, start_y) to (goal_x, goal_y). Returns waypoints and path distance.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_x": {"type": "number"},
                    "start_y": {"type": "number"},
                    "goal_x": {"type": "number"},
                    "goal_y": {"type": "number"},
                    "smoothing": {"type": "boolean", "default": True},
                },
                "required": ["start_x", "start_y", "goal_x", "goal_y"],
            },
        },
        {
            "name": "nav_execute_path",
            "description": "执行导航路径 / Drive the active robot along world-frame waypoints with closed-loop base control.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "waypoints": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "timeout": {"type": "number", "default": 120.0},
                    "position_tolerance": {"type": "number", "default": 0.12},
                    "start_tolerance": {"type": "number", "default": 0.5},
                    "max_linear_speed": {"type": "number", "default": 0.4},
                    "max_angular_speed": {"type": "number", "default": 0.8},
                    "control_hz": {"type": "number", "default": 20.0},
                },
                "required": ["waypoints"],
            },
        },
        # -- FBE --
        {
            "name": "nav_fbe_init",
            "description": "初始化 FBE 自主探索 / Initialize Frontier-Based Exploration. Builds a god grid and creates an FBEPlanner.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scene_xml": {"type": "string"},
                    "bounds": {"type": "array", "items": {"type": "number"}},
                    "resolution": {"type": "number", "default": 0.08},
                    "agent_radius": {"type": "number", "default": 0.25},
                    "num_rays": {"type": "integer", "default": 180},
                    "max_range_m": {"type": "number", "default": 5.0},
                    "fov_degrees": {"type": "number", "default": 270.0},
                    "explore_threshold": {"type": "number", "default": 0.85},
                    "start_x": {"type": "number", "default": 0.0},
                    "start_y": {"type": "number", "default": 0.0},
                    "start_yaw": {"type": "number", "default": 0.0},
                },
            },
        },
        {
            "name": "nav_fbe_step",
            "description": "FBE 探索一步 / Advance FBE exploration by one step with robot pose.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "robot_x": {"type": "number"},
                    "robot_y": {"type": "number"},
                    "robot_yaw": {"type": "number"},
                },
                "required": ["robot_x", "robot_y", "robot_yaw"],
            },
        },
        {
            "name": "nav_fbe_path",
            "description": "获取当前 FBE 路径 / Get current FBE path waypoints.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        # -- VLFM --
        {
            "name": "nav_vlfm_init",
            "description": "初始化 VLFM 视觉语言探索 / Initialize Vision-Language Frontier Maps exploration with a natural-language instruction.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Natural-language description of what to find.",
                    },
                    "scene_xml": {"type": "string"},
                    "vlm_type": {
                        "type": "string",
                        "default": "clip",
                        "enum": ["clip", "siglip", "openai", "blip2"],
                    },
                    "device": {"type": "string", "default": "cpu"},
                    "resolution": {"type": "number", "default": 0.08},
                    "agent_radius": {"type": "number", "default": 0.25},
                    "max_range_m": {"type": "number", "default": 5.0},
                    "explore_threshold": {"type": "number", "default": 0.85},
                    "allow_geometric_fallback": {"type": "boolean", "default": True},
                    "start_x": {"type": "number", "default": 0.0},
                    "start_y": {"type": "number", "default": 0.0},
                    "start_yaw": {"type": "number", "default": 0.0},
                },
                "required": ["instruction"],
            },
        },
        {
            "name": "nav_vlfm_step",
            "description": "VLFM 探索一步 / Advance VLFM exploration by one step.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "robot_x": {"type": "number"},
                    "robot_y": {"type": "number"},
                    "robot_yaw": {"type": "number"},
                },
                "required": ["robot_x", "robot_y", "robot_yaw"],
            },
        },
        {
            "name": "nav_vlfm_inject_observation",
            "description": "向 VLFM 注入 RGB-D 观测 / Inject an RGB-D observation for VLM scoring and value map update.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rgb_image_path": {"type": "string"},
                    "depth_image_path": {"type": "string"},
                    "camera_x": {"type": "number", "default": 0.0},
                    "camera_y": {"type": "number", "default": 0.0},
                    "camera_yaw": {"type": "number", "default": 0.0},
                    "hfov_rad": {"type": "number", "default": 1.2},
                    "min_depth": {"type": "number", "default": 0.1},
                    "max_depth": {"type": "number", "default": 5.0},
                },
                "required": ["rgb_image_path", "depth_image_path"],
            },
        },
        # -- Robot lifecycle --
        {
            "name": "robot_init",
            "description": "启动机器人仿真 / Start a robot simulator. Choose from stretch3, google_robot, or tidybot.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "robot_type": {
                        "type": "string",
                        "enum": ["stretch3", "google_robot", "tidybot"],
                        "description": "Robot model to simulate.",
                    },
                    "scene_xml": {"type": "string", "description": "Path to scene XML file."},
                    "headless": {"type": "boolean", "default": True},
                    "cameras_to_use": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Camera names or all/all_rgb/all_depth",
                    },
                    "camera_hz": {"type": "number", "default": 30.0},
                    "start_translation": {"type": "array", "items": {"type": "number"}},
                    "start_rotation_quat": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["robot_type"],
            },
        },
        {
            "name": "robot_stop",
            "description": "停止机器人仿真 / Stop the active robot simulator.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        # -- Joint control --
        {
            "name": "robot_move_to",
            "description": "移动关节到绝对位置 / Move a joint to an absolute target position.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "actuator_name": {
                        "type": "string",
                        "description": "Actuator name (e.g. 'lift', 'joint_arm_l0').",
                    },
                    "position": {"type": "number"},
                },
                "required": ["actuator_name", "position"],
            },
        },
        {
            "name": "robot_move_by",
            "description": "移动关节相对偏移 / Move a joint by a relative delta.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "actuator_name": {"type": "string"},
                    "delta": {"type": "number"},
                },
                "required": ["actuator_name", "delta"],
            },
        },
        {
            "name": "robot_set_base_velocity",
            "description": "设置底盘速度 / Set mobile base velocity (m/s forward, rad/s angular).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "v_linear": {"type": "number", "default": 0.0},
                    "omega": {"type": "number", "default": 0.0},
                    "v_lateral": {"type": "number", "default": 0.0},
                },
            },
        },
        {
            "name": "robot_home",
            "description": "回到 home 位姿 / Move to home keyframe pose.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "robot_stow",
            "description": "回到 stow 位姿 / Move to stow/retract keyframe pose.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "robot_wait_until_at_setpoint",
            "description": "等待关节到达目标 / Block until a joint reaches its commanded setpoint.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "actuator_name": {"type": "string"},
                    "timeout": {"type": "number", "default": 5.0},
                    "position_tolerance": {"type": "number", "default": 0.05},
                },
                "required": ["actuator_name"],
            },
        },
        # -- Status --
        {
            "name": "robot_get_status",
            "description": "获取机器人状态 / Pull current joint status snapshot.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "robot_get_base_pose",
            "description": "获取底盘位姿 (x, y, theta) / Get base pose in world coordinates.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "robot_get_ee_pose",
            "description": "获取末端位姿 (4×4) / Get end-effector pose matrix.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "robot_get_camera_data",
            "description": "获取相机数据 / Get camera image metadata or list available cameras.",
            "inputSchema": {
                "type": "object",
                "properties": {"camera_name": {"type": "string", "default": ""}},
            },
        },
        # -- Grasping --
        {
            "name": "robot_attach_object",
            "description": "抓取物体 / Attach a named object to the gripper.",
            "inputSchema": {
                "type": "object",
                "properties": {"object_id": {"type": "string"}},
                "required": ["object_id"],
            },
        },
        {
            "name": "robot_release_object",
            "description": "释放物体 / Release the currently grasped object.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        # -- Introspection --
        {
            "name": "robot_list_actuators",
            "description": "列出所有执行器 / List all available actuators for the current robot.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "robot_list_cameras",
            "description": "列出所有相机 / List all available cameras for the current robot.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        # -- Scene --
        {
            "name": "scene_list_objects",
            "description": "列出场景中可移动物体 / List free-joint bodies in the scene.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "scene_get_object_pose",
            "description": "获取物体位姿 / Get the world pose of a named body.",
            "inputSchema": {
                "type": "object",
                "properties": {"object_name": {"type": "string"}},
                "required": ["object_name"],
            },
        },
        {
            "name": "scene_add_world_frame",
            "description": "添加世界坐标系标记 / Visualize a coordinate frame in the scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "default": 0.0},
                    "y": {"type": "number", "default": 0.0},
                    "z": {"type": "number", "default": 0.0},
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# FastMCP server (optional — requires ``mcp`` SDK)
# ---------------------------------------------------------------------------


def build_mcp_server():
    """Build an official MCP stdio server if the ``mcp`` SDK is installed.

    Returns:
        A ``FastMCP`` server object when the SDK is available, otherwise ``None``.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        return None

    server = FastMCP("strech-codex-server")
    descriptions = {item["name"]: item["description"] for item in tool_manifest()}

    def mcp_tool(function):
        return server.tool(description=descriptions[function.__name__])(function)

    # -- Grid --
    @mcp_tool
    def nav_build_grid(
        scene_xml: str | None = None,
        bounds: list[float] | None = None,
        resolution: float = 0.08,
        agent_radius: float = 0.25,
        floor_geom_name: str = "office_floor",
        minimum_obstacle_height: float = 0.08,
        maximum_obstacle_height: float = 1.80,
        require_collision: bool = True,
        exclude_prefixes: list[str] | None = None,
    ) -> str:
        return json.dumps(
            dispatch_tool(
                "nav_build_grid",
                {k: v for k, v in locals().items() if v is not None},
            ),
            ensure_ascii=False,
        )

    @mcp_tool
    def nav_get_grid_info() -> str:
        return json.dumps(dispatch_tool("nav_get_grid_info"), ensure_ascii=False)

    @mcp_tool
    def nav_is_free(x: float, y: float) -> str:
        return json.dumps(dispatch_tool("nav_is_free", {"x": x, "y": y}), ensure_ascii=False)

    # -- Scene discovery --
    @mcp_tool
    def nav_list_scenes() -> str:
        return json.dumps(dispatch_tool("nav_list_scenes"), ensure_ascii=False)

    # -- A* --
    @mcp_tool
    def nav_astar(
        start_x: float, start_y: float, goal_x: float, goal_y: float, smoothing: bool = True
    ) -> str:
        return json.dumps(
            dispatch_tool("nav_astar", {k: v for k, v in locals().items()}),
            ensure_ascii=False,
        )

    # -- FMM --
    @mcp_tool
    def nav_fmm(
        start_x: float, start_y: float, goal_x: float, goal_y: float, smoothing: bool = True
    ) -> str:
        return json.dumps(
            dispatch_tool("nav_fmm", {k: v for k, v in locals().items()}),
            ensure_ascii=False,
        )

    @mcp_tool
    def nav_execute_path(
        waypoints: list[list[float]],
        timeout: float = 120.0,
        position_tolerance: float = 0.12,
        start_tolerance: float = 0.5,
        max_linear_speed: float = 0.4,
        max_angular_speed: float = 0.8,
        control_hz: float = 20.0,
    ) -> str:
        return json.dumps(
            dispatch_tool("nav_execute_path", {k: v for k, v in locals().items()}),
            ensure_ascii=False,
        )

    # -- FBE --
    @mcp_tool
    def nav_fbe_init(
        scene_xml: str | None = None,
        bounds: list[float] | None = None,
        resolution: float = 0.08,
        agent_radius: float = 0.25,
        num_rays: int = 180,
        max_range_m: float = 5.0,
        fov_degrees: float = 270.0,
        min_cluster_size: int = 3,
        explore_threshold: float = 0.85,
        start_x: float = 0.0,
        start_y: float = 0.0,
        start_yaw: float = 0.0,
    ) -> str:
        return json.dumps(
            dispatch_tool("nav_fbe_init", {k: v for k, v in locals().items() if v is not None}),
            ensure_ascii=False,
        )

    @mcp_tool
    def nav_fbe_step(robot_x: float, robot_y: float, robot_yaw: float) -> str:
        return json.dumps(
            dispatch_tool(
                "nav_fbe_step", {"robot_x": robot_x, "robot_y": robot_y, "robot_yaw": robot_yaw}
            ),
            ensure_ascii=False,
        )

    @mcp_tool
    def nav_fbe_path() -> str:
        return json.dumps(dispatch_tool("nav_fbe_path"), ensure_ascii=False)

    # -- VLFM --
    @mcp_tool
    def nav_vlfm_init(
        instruction: str,
        scene_xml: str | None = None,
        bounds: list[float] | None = None,
        resolution: float = 0.08,
        agent_radius: float = 0.25,
        num_rays: int = 180,
        max_range_m: float = 5.0,
        fov_degrees: float = 270.0,
        min_cluster_size: int = 3,
        explore_threshold: float = 0.85,
        vlm_type: str = "clip",
        device: str = "cpu",
        start_x: float = 0.0,
        start_y: float = 0.0,
        start_yaw: float = 0.0,
        allow_geometric_fallback: bool = True,
    ) -> str:
        return json.dumps(
            dispatch_tool("nav_vlfm_init", {k: v for k, v in locals().items() if v is not None}),
            ensure_ascii=False,
        )

    @mcp_tool
    def nav_vlfm_step(robot_x: float, robot_y: float, robot_yaw: float) -> str:
        return json.dumps(
            dispatch_tool(
                "nav_vlfm_step",
                {"robot_x": robot_x, "robot_y": robot_y, "robot_yaw": robot_yaw},
            ),
            ensure_ascii=False,
        )

    @mcp_tool
    def nav_vlfm_inject_observation(
        rgb_image_path: str = "",
        depth_image_path: str = "",
        camera_x: float = 0.0,
        camera_y: float = 0.0,
        camera_yaw: float = 0.0,
        hfov_rad: float = 1.2,
        min_depth: float = 0.1,
        max_depth: float = 5.0,
    ) -> str:
        return json.dumps(
            dispatch_tool(
                "nav_vlfm_inject_observation",
                {k: v for k, v in locals().items()},
            ),
            ensure_ascii=False,
        )

    # -- Robot --
    @mcp_tool
    def robot_init(
        robot_type: str,
        scene_xml: str | None = None,
        headless: bool = True,
        cameras_to_use: list[str] | None = None,
        camera_hz: float = 30.0,
        start_translation: list[float] | None = None,
        start_rotation_quat: list[float] | None = None,
    ) -> str:
        return json.dumps(
            dispatch_tool("robot_init", {k: v for k, v in locals().items() if v is not None}),
            ensure_ascii=False,
        )

    @mcp_tool
    def robot_stop() -> str:
        return json.dumps(dispatch_tool("robot_stop"), ensure_ascii=False)

    @mcp_tool
    def robot_move_to(actuator_name: str, position: float) -> str:
        return json.dumps(
            dispatch_tool("robot_move_to", {"actuator_name": actuator_name, "position": position}),
            ensure_ascii=False,
        )

    @mcp_tool
    def robot_move_by(actuator_name: str, delta: float) -> str:
        return json.dumps(
            dispatch_tool("robot_move_by", {"actuator_name": actuator_name, "delta": delta}),
            ensure_ascii=False,
        )

    @mcp_tool
    def robot_set_base_velocity(
        v_linear: float = 0.0, omega: float = 0.0, v_lateral: float = 0.0
    ) -> str:
        return json.dumps(
            dispatch_tool(
                "robot_set_base_velocity",
                {"v_linear": v_linear, "omega": omega, "v_lateral": v_lateral},
            ),
            ensure_ascii=False,
        )

    @mcp_tool
    def robot_home() -> str:
        return json.dumps(dispatch_tool("robot_home"), ensure_ascii=False)

    @mcp_tool
    def robot_stow() -> str:
        return json.dumps(dispatch_tool("robot_stow"), ensure_ascii=False)

    @mcp_tool
    def robot_wait_until_at_setpoint(
        actuator_name: str,
        timeout: float = 5.0,
        position_tolerance: float = 0.05,
    ) -> str:
        return json.dumps(
            dispatch_tool(
                "robot_wait_until_at_setpoint",
                {
                    "actuator_name": actuator_name,
                    "timeout": timeout,
                    "position_tolerance": position_tolerance,
                },
            ),
            ensure_ascii=False,
        )

    @mcp_tool
    def robot_get_status() -> str:
        return json.dumps(dispatch_tool("robot_get_status"), ensure_ascii=False)

    @mcp_tool
    def robot_get_base_pose() -> str:
        return json.dumps(dispatch_tool("robot_get_base_pose"), ensure_ascii=False)

    @mcp_tool
    def robot_get_ee_pose() -> str:
        return json.dumps(dispatch_tool("robot_get_ee_pose"), ensure_ascii=False)

    @mcp_tool
    def robot_get_camera_data(camera_name: str = "") -> str:
        return json.dumps(
            dispatch_tool("robot_get_camera_data", {"camera_name": camera_name}),
            ensure_ascii=False,
        )

    @mcp_tool
    def robot_attach_object(object_id: str) -> str:
        return json.dumps(
            dispatch_tool("robot_attach_object", {"object_id": object_id}),
            ensure_ascii=False,
        )

    @mcp_tool
    def robot_release_object() -> str:
        return json.dumps(dispatch_tool("robot_release_object"), ensure_ascii=False)

    @mcp_tool
    def robot_list_actuators() -> str:
        return json.dumps(dispatch_tool("robot_list_actuators"), ensure_ascii=False)

    @mcp_tool
    def robot_list_cameras() -> str:
        return json.dumps(dispatch_tool("robot_list_cameras"), ensure_ascii=False)

    # -- Scene --
    @mcp_tool
    def scene_list_objects() -> str:
        return json.dumps(dispatch_tool("scene_list_objects"), ensure_ascii=False)

    @mcp_tool
    def scene_get_object_pose(object_name: str) -> str:
        return json.dumps(
            dispatch_tool("scene_get_object_pose", {"object_name": object_name}),
            ensure_ascii=False,
        )

    @mcp_tool
    def scene_add_world_frame(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> str:
        return json.dumps(
            dispatch_tool("scene_add_world_frame", {"x": x, "y": y, "z": z}),
            ensure_ascii=False,
        )

    return server


# ---------------------------------------------------------------------------
# Stdio entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the MCP server on stdio (requires ``mcp`` SDK)."""
    os.environ["STRECH_CODEX_MCP_STDIO"] = "1"
    server = build_mcp_server()
    if server is None:
        raise SystemExit("MCP SDK is not installed. Install with: pip install 'strech-codex[mcp]'")
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
