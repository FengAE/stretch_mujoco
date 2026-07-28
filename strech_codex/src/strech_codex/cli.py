"""Command-line interface for the robot Codex agent.

Usage::

    python -m strech_codex.cli --mode offline "去厨房拿苹果"
    python -m strech_codex.cli --mode inspect
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from pathlib import Path as _Path

import stretch_mujoco

from .agent import format_run, run_offline
from .codex_adapter import format_live_event, inspect_environment, run_live_sync
from .world import state

DEFAULT_TASK = "启动 stretch3 机器人并查询状态"

# Default scene path relative to the installed stretch_mujoco package.
_DEFAULT_OFFICE_SCENE = str(
    _Path(stretch_mujoco.__file__).resolve().parent / "models" / "office_scene.xml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robot Codex — MCP-based orchestration for MuJoCo robots"
    )
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="自然语言机器人任务")
    parser.add_argument(
        "--mode",
        choices=["offline", "inspect", "live"],
        default="offline",
        help="offline: deterministic local planner; inspect: SDK diagnostics; live: Codex SDK integration",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 结果")
    parser.add_argument("--robot", default=None, help="Robot type: stretch3, google_robot, tidybot")
    parser.add_argument("--scene", default=None, help="Path to MuJoCo scene XML")
    parser.add_argument(
        "--base-url",
        help="live mode API 地址；未指定时读取 OPENAI_BASE_URL",
    )
    parser.add_argument(
        "--model",
        help="live mode 模型名；未指定时读取 CODEX_MODEL",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="保留兼容；live mode 默认输出工具执行日志",
    )
    return parser


def _print_live_event(event: object) -> None:
    message = format_live_event(event)
    if message:
        print(f"[{datetime.now().astimezone():%H:%M:%S}] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "inspect":
        data = inspect_environment().as_dict()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.mode == "live":
        try:
            result = run_live_sync(
                args.task,
                base_url=args.base_url,
                model=args.model,
                on_event=_print_live_event,
            )
        except Exception as exc:
            print(f"live mode failed: {exc}", file=sys.stderr)
            return 2
        print(result.final_response)
        print(f"Episode log: {result.episode_path}")
        if result.video_path:
            print(f"Episode video: {result.video_path}")
        return 0

    # Offline mode — wire up scene / robot config before running
    if args.scene:
        state.set_scene_path(args.scene)
    elif state.get_scene_path() is None:
        # Use default office scene if available
        if _Path(_DEFAULT_OFFICE_SCENE).exists():
            state.set_scene_path(_DEFAULT_OFFICE_SCENE)

    if args.robot:
        state.set_robot_type(args.robot)

    result = run_offline(args.task)
    if args.json:
        print(
            json.dumps(
                {
                    "task": result.task,
                    "mode": result.mode,
                    "tool_results": [
                        {
                            "tool": item.call.name,
                            "arguments": item.call.arguments,
                            "result": item.result,
                        }
                        for item in result.tool_results
                    ],
                    "final_response": result.final_response,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_run(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
