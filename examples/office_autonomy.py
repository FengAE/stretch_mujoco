"""Run a low-compute autonomous office behavior trace."""

import json
from pathlib import Path

import click

from stretch_mujoco.agents import OfficeAgentRuntime
from stretch_mujoco.semantics import SemanticWorld


MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


@click.command()
@click.option("--seconds", type=float, default=60.0, show_default=True)
@click.option("--step", type=float, default=0.25, show_default=True)
def main(seconds: float, step: float) -> None:
    """Print seeded plans, actions, office events, and queued LLM triggers."""
    if seconds <= 0 or step <= 0:
        raise click.BadParameter("seconds and step must be positive")
    world = SemanticWorld.from_json(MODELS_PATH / "office_semantics.json")
    runtime = OfficeAgentRuntime.from_json(
        world,
        MODELS_PATH / "office_agents.json",
        auto_plan=True,
    )
    events = list(runtime.drain_events())
    elapsed = 0.0
    while elapsed < seconds:
        events.extend(runtime.tick(min(step, seconds - elapsed)))
        elapsed += step

    employee = runtime.agents["employee_01"]
    output = {
        "clock": {
            "day": runtime.day,
            "minute_of_day": runtime.minute_of_day,
        },
        "state": vars(employee.state),
        "last_utility_scores": [
            {"goal": score.goal.value, "score": round(score.score, 4)}
            for score in employee.planner.last_scores
        ],
        "events": [
            {
                "time": round(event.time, 2),
                "event": event.event,
                "details": event.details,
            }
            for event in events
        ],
        "queued_llm_triggers": [request.trigger.value for request in runtime.drain_llm_requests()],
        "llm_calls_during_simulation": runtime.llm.calls_for(runtime.day, "employee_01"),
        "llm_config": runtime.llm_config_summary(),
    }
    click.echo(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
