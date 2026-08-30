"""Run a multi-NPC office day in Scene 2 with event-driven LLM guidance.

Each employee agent follows their own schedule, makes independent utility
decisions, and receives LLM assistance for planning and unexpected events.
The runtime already supports multiple agents natively — this example simply
loads a multi-employee configuration and processes all LLM requests in the
main loop.

Pass --mp4 to generate a 2-D top-down animation of the working day.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any

import click

from stretch_mujoco.agents import LLMProviderError, OfficeAgentRuntime
from stretch_mujoco.agents.mp4_recorder import OfficeMp4Recorder
from stretch_mujoco.semantics import SemanticWorld


MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
DEFAULT_SEMANTICS = MODELS_PATH / "office_scene2_semantics.json"
DEFAULT_AGENTS = MODELS_PATH / "office_scene2_agents.json"

# World-coordinate positions for semantic locations (Scene 2 layout).
# These drive the 2-D agent markers in the MP4 recording.
LOCATION_POSITIONS: dict[str, tuple[float, float]] = {
    "workstation_left": (-3.0, -3.0),
    "workstation_right": (3.0, -3.0),
    "chair_left": (-3.0, -3.6),
    "chair_right": (3.0, -3.6),
    "meeting_table": (-5.5, 3.0),
    "snack_counter": (6.8, 3.0),
    "storage_cabinet": (5.2, 5.5),
}


def format_clock(minute: float) -> str:
    total = int(minute) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def process_llm_requests(runtime, provider, records: list[dict[str, Any]]):
    """Drain pending LLM requests for all agents and apply responses.

    This is the same pattern as the single-NPC example — the runtime drains
    requests for *all* agents, so no per-agent loop is needed.
    """
    events = []
    for request in runtime.drain_llm_requests():
        record = {
            "time": format_clock(request.minute_of_day),
            "agent": request.agent_id,
            "trigger": request.trigger.value,
        }
        try:
            response = runtime.llm.process(request, provider)
            validation = runtime.apply_llm_response(request, response)
        except (LLMProviderError, TimeoutError, OSError, RuntimeError, ValueError) as error:
            record.update(status="provider_error", error=str(error))
        else:
            record.update(
                status="applied" if validation.valid else "rejected",
                response_fields=sorted(response) if isinstance(response, dict) else [],
                errors=list(validation.errors),
            )
            if isinstance(response, dict):
                action = response.get("action")
                if isinstance(action, dict):
                    record["action"] = {
                        "action": action.get("action"),
                        "target": action.get("target"),
                        "parameters": action.get("parameters", {}),
                    }
                dialogue = response.get("dialogue")
                if isinstance(dialogue, str):
                    record["dialogue"] = dialogue
        records.append(record)
        events.extend(runtime.drain_events())
    return events


def _per_agent_report(runtime, agent_id: str, events: list) -> dict[str, Any]:
    """Build a per-agent statistics block from the completed simulation."""
    agent = runtime.agents[agent_id]
    agent_events = [e for e in events if e.agent_id == agent_id]

    action_counts = Counter(
        e.details["action"]
        for e in agent_events
        if e.event == "action_succeeded"
    )
    goal_counts = Counter(
        e.details["goal"] for e in agent_events if e.event == "plan_selected"
    )
    rejection_counts: Counter = Counter()
    for e in agent_events:
        if e.event == "action_rejected":
            for err in e.details.get("errors", []):
                rejection_counts[err] += 1

    return {
        "agent": agent_id,
        "profile": {
            "role": agent.profile.role,
            "department": agent.profile.department,
            "personality": agent.profile.personality,
        },
        "llm_calls": runtime.llm.calls_for(runtime.day, agent_id),
        "schedule_items": len(agent.schedule.items),
        "actions": {
            "successful": dict(sorted(action_counts.items())),
            "goals_selected": dict(sorted(goal_counts.items())),
            "rejections": dict(sorted(rejection_counts.items())),
        },
        "final_state": {
            "location": agent.state.location,
            "held_object": agent.state.held_object,
            "hunger": round(agent.state.hunger, 3),
            "thirst": round(agent.state.thirst, 3),
            "fatigue": round(agent.state.fatigue, 3),
            "mood": round(agent.state.mood, 3),
            "last_action": agent.state.current_action,
        },
        "memory_entries": len(agent.memory.entries),
    }


@click.command()
@click.option("--end", "end_hour", type=int, default=18, show_default=True,
              help="Simulation end hour (10-23)")
@click.option("--step", type=float, default=0.25, show_default=True,
              help="Tick step in minutes")
@click.option("--output", type=click.Path(path_type=Path), default=None,
              help="Write JSON report to this file")
@click.option("--mp4", "mp4_path", type=click.Path(path_type=Path), default=None,
              help="Generate a 2-D top-down MP4 animation of the workday")
@click.option("--mp4-fps", type=int, default=10, show_default=True,
              help="Frame rate for MP4 output")
@click.option("--semantics", type=click.Path(exists=True, path_type=Path),
              default=DEFAULT_SEMANTICS, show_default=True)
@click.option("--agents", "agents_path", type=click.Path(exists=True, path_type=Path),
              default=DEFAULT_AGENTS, show_default=True)
@click.option("--chat-log", is_flag=True, help="Print a live chat-style log of agent activity")
def main(
    end_hour: int,
    step: float,
    output: Path | None,
    mp4_path: Path | None,
    mp4_fps: int,
    semantics: Path,
    agents_path: Path,
    chat_log: bool,
) -> None:
    """Simulate multiple office employees from 09:00 until END."""
    if not 10 <= end_hour <= 23:
        raise click.BadParameter("end must be between 10 and 23")
    if step <= 0:
        raise click.BadParameter("step must be positive")

    world = SemanticWorld.from_json(semantics)
    runtime = OfficeAgentRuntime.from_json(world, agents_path, auto_plan=True)
    provider = runtime.create_llm_provider()

    agent_ids = sorted(runtime.agents.keys())
    click.echo(f"Loaded {len(agent_ids)} agent(s): {', '.join(agent_ids)}")
    for aid in agent_ids:
        agent = runtime.agents[aid]
        click.echo(f"  {aid}: {agent.profile.role} at {agent.state.location}")

    all_events = list(runtime.drain_events())
    llm_records: list[dict[str, Any]] = []

    # Kick off day-start LLM events for all agents
    all_events.extend(process_llm_requests(runtime, provider, llm_records))

    # --- MP4 recorder ---
    recorder: OfficeMp4Recorder | None = None
    mp4_manifest = MODELS_PATH / "assets" / "office_scenes" / "office_02_cross_axis.json"
    if mp4_path is not None:
        recorder = OfficeMp4Recorder(
            mp4_path,
            mp4_manifest if mp4_manifest.exists() else None,
            fps=mp4_fps,
        )
        recorder.start()
        click.echo(f"Recording MP4 -> {mp4_path}")

    end_minute = end_hour * 60
    mp4_frame_interval = max(1.0, 1.0 / mp4_fps)  # real seconds between MP4 frames
    mp4_accumulator = 0.0

    while runtime.minute_of_day < end_minute:
        remaining_seconds = (end_minute - runtime.minute_of_day) / runtime.minutes_per_second
        tick_seconds = min(step, remaining_seconds)
        all_events.extend(runtime.tick(tick_seconds))

        # Auto-complete all pending robot tasks so agents can continue
        for task in runtime.pending_robot_tasks():
            runtime.complete_robot_task(task.task_id, success=True)
        all_events.extend(runtime.drain_events())

        # Process new LLM requests (planning, dialogue, events)
        new_events = process_llm_requests(runtime, provider, llm_records)

        # --- MP4 frame ---
        if recorder is not None:
            mp4_accumulator += tick_seconds
            if mp4_accumulator >= mp4_frame_interval:
                mp4_accumulator -= mp4_frame_interval
                agents_viz = {}
                for aid, agent in runtime.agents.items():
                    pos = LOCATION_POSITIONS.get(
                        agent.state.location, (0.0, 0.0)
                    )
                    agents_viz[aid] = {
                        "position": pos,
                        "action": agent.state.current_action,
                        "label": agent.profile.role.split()[0][:8],
                    }
                event_lines = []
                for e in all_events[-6:]:
                    detail = e.details.get("action", e.details.get("goal", ""))
                    event_lines.append(
                        f"[{format_clock(e.time)}] {e.agent_id}: {e.event} {detail}"
                    )
                recorder.record_frame(
                    runtime.minute_of_day, agents_viz, events=event_lines
                )

        # Live chat log
        if chat_log:
            for record in llm_records:
                if "dialogue" in record:
                    click.echo(
                        f"[{record['time']}] {record['agent']}: "
                        f"{record['dialogue']}"
                    )
                elif record.get("action"):
                    act = record["action"]
                    click.echo(
                        f"[{record['time']}] {record['agent']} -> "
                        f"{act.get('action','?')} "
                        f"target={act.get('target','?')} "
                        f"({record.get('status','?')})"
                    )

        all_events.extend(new_events)

    # Let in-progress actions finish
    scheduled_end = runtime.minute_of_day
    runtime.auto_plan = False
    for agent in runtime.agents.values():
        agent.planner.action_queue.clear()
    while any(agent.executor.is_busy for agent in runtime.agents.values()):
        all_events.extend(runtime.tick(step))

    # --- Finalise MP4 ---
    if recorder is not None:
        recorder.close()
        click.echo(f"MP4 saved -> {mp4_path}  ({recorder._frame_count} frames)")

    # --- Build report ---
    event_counts = Counter(e.event for e in all_events)
    per_agent = {
        aid: _per_agent_report(runtime, aid, all_events) for aid in agent_ids
    }

    # Dialogue summary
    dialogues = [
        r for r in llm_records
        if "dialogue" in r and r.get("status") == "applied"
    ]

    report = {
        "scene": world.scene,
        "workday": {
            "start": "09:00",
            "scheduled_end": format_clock(scheduled_end),
            "completed_at": format_clock(runtime.minute_of_day),
        },
        "llm_config": runtime.llm_config_summary(),
        "agent_count": len(agent_ids),
        "agents": per_agent,
        "llm": {
            "total_calls": sum(runtime.llm.calls_for(runtime.day, aid) for aid in agent_ids),
            "requests": llm_records,
        },
        "dialogues": dialogues,
        "global_event_counts": dict(sorted(event_counts.items())),
        "robot_tasks": [
            {
                "task": task.task,
                "object": task.object_id,
                "destination": task.destination,
                "requester": task.requester,
                "status": task.status.value,
            }
            for task in runtime.robot_tasks.values()
        ],
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        click.echo(f"\nReport written to {output}")
    else:
        click.echo(rendered)


if __name__ == "__main__":
    main()
