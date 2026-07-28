"""Run one accelerated office day with event-driven LLM guidance."""

from collections import Counter
import json
from pathlib import Path
from typing import Any

import click

from stretch_mujoco.agents import LLMProviderError, OfficeAgentRuntime
from stretch_mujoco.semantics import SemanticWorld


MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


def format_clock(minute: float) -> str:
    total = int(minute) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def process_llm_requests(runtime, provider, records: list[dict[str, Any]]):
    events = []
    for request in runtime.drain_llm_requests():
        record = {"time": format_clock(request.minute_of_day), "trigger": request.trigger.value}
        try:
            response = runtime.llm.process(request, provider)
            validation = runtime.apply_llm_response(request, response)
        except (LLMProviderError, TimeoutError, OSError, RuntimeError, ValueError) as error:
            record.update(status="provider_error", error=str(error))
        else:
            record.update(
                status="applied" if validation.valid else "rejected",
                response_fields=sorted(response),
                errors=list(validation.errors),
            )
            if isinstance(response.get("action"), dict):
                action = response["action"]
                record["action"] = {
                    "action": action.get("action"),
                    "target": action.get("target"),
                    "parameters": action.get("parameters", {}),
                }
        records.append(record)
        events.extend(runtime.drain_events())
    return events


@click.command()
@click.option("--end", "end_hour", type=int, default=18, show_default=True)
@click.option("--step", type=float, default=0.25, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def main(end_hour: int, step: float, output: Path | None) -> None:
    """Simulate employee_01 from 09:00 until END using a configured LLM."""
    if not 10 <= end_hour <= 23:
        raise click.BadParameter("end must be between 10 and 23")
    if step <= 0:
        raise click.BadParameter("step must be positive")

    world = SemanticWorld.from_json(MODELS_PATH / "office_semantics.json")
    runtime = OfficeAgentRuntime.from_json(
        world,
        MODELS_PATH / "office_agents.json",
        auto_plan=True,
    )
    provider = runtime.create_llm_provider()
    employee = runtime.agents["employee_01"]
    events = list(runtime.drain_events())
    llm_records: list[dict[str, Any]] = []
    events.extend(process_llm_requests(runtime, provider, llm_records))

    end_minute = end_hour * 60
    while runtime.minute_of_day < end_minute:
        remaining_seconds = (end_minute - runtime.minute_of_day) / runtime.minutes_per_second
        events.extend(runtime.tick(min(step, remaining_seconds)))

        # This runner validates agent behavior, not robot navigation. Treat each
        # semantic delivery as successful so the employee can continue the day.
        for task in runtime.pending_robot_tasks():
            runtime.complete_robot_task(task.task_id, success=True)
        events.extend(runtime.drain_events())
        events.extend(process_llm_requests(runtime, provider, llm_records))

    scheduled_end = runtime.minute_of_day
    runtime.auto_plan = False
    employee.planner.action_queue.clear()
    while employee.executor.is_busy:
        events.extend(runtime.tick(step))

    event_counts = Counter(event.event for event in events)
    action_counts = Counter(
        event.details["action"] for event in events if event.event == "action_succeeded"
    )
    plan_counts = Counter(
        event.details["goal"] for event in events if event.event == "plan_selected"
    )
    rejection_counts = Counter(
        error
        for event in events
        if event.event == "action_rejected"
        for error in event.details.get("errors", [])
    )
    report = {
        "workday": {
            "start": "09:00",
            "scheduled_end": format_clock(scheduled_end),
            "completed_at": format_clock(runtime.minute_of_day),
        },
        "agent": employee.agent_id,
        "profile": {
            "role": employee.profile.role,
            "department": employee.profile.department,
        },
        "llm": {
            "config": runtime.llm_config_summary(),
            "calls": runtime.llm.calls_for(runtime.day, employee.agent_id),
            "requests": llm_records,
        },
        "schedule": [
            {
                "id": item.item_id,
                "start": format_clock(item.start_minute),
                "end": format_clock(item.end_minute),
                "activity": item.activity,
                "location": item.location,
                "variation_minutes": item.variation_minutes,
            }
            for item in employee.schedule.items
        ],
        "activity": {
            "successful_actions": dict(sorted(action_counts.items())),
            "selected_goals": dict(sorted(plan_counts.items())),
            "action_rejections": dict(sorted(rejection_counts.items())),
            "event_counts": dict(sorted(event_counts.items())),
        },
        "robot_tasks": [
            {
                "task": task.task,
                "object": task.object_id,
                "destination": task.destination,
                "status": task.status.value,
            }
            for task in runtime.robot_tasks.values()
        ],
        "final_state": vars(employee.state),
        "memory_entries": len(employee.memory.entries),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    click.echo(rendered)


if __name__ == "__main__":
    main()
