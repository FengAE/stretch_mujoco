from pathlib import Path
import time

import click

from stretch_mujoco import StretchMujocoSimulator


OFFICE_SCENE_PATH = (
    Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "office_scene.xml"
)


@click.command()
@click.option("--headless", is_flag=True, help="Run without opening the MuJoCo viewer.")
@click.option("--show-viewer-ui", is_flag=True, help="Show the viewer's full control panel.")
@click.option(
    "--npc-animation",
    type=click.Choice(("idle", "walk", "sit")),
    default="idle",
    show_default=True,
    help="NPC animation to play.",
)
@click.option(
    "--animation-demo",
    is_flag=True,
    help="Cycle through Idle, Walk, and Sit for visual inspection.",
)
@click.option(
    "--npc-chair",
    type=click.Choice(("chair_left", "chair_right")),
    default="chair_right",
    show_default=True,
    help="Office chair used by the NPC's Sit animation.",
)
def main(
    headless: bool,
    show_viewer_ui: bool,
    npc_animation: str,
    animation_demo: bool,
    npc_chair: str,
) -> None:
    """Launch Stretch 3 in the interactive office and snack scene."""
    sim = StretchMujocoSimulator(scene_xml_path=str(OFFICE_SCENE_PATH))

    try:
        sim.start(headless=headless, show_viewer_ui=show_viewer_ui)
        sim.set_humanoid_sit_target(npc_chair)
        sim.set_humanoid_animation(npc_animation)
        demo_schedule = (("idle", 3.0), ("walk", 3.0), ("sit", 10.0))
        demo_cycle_duration = sum(duration for _, duration in demo_schedule)
        last_demo_clip = None
        while sim.is_running():
            if animation_demo:
                cycle_time = sim.pull_status().time % demo_cycle_duration
                demo_clip = demo_schedule[-1][0]
                elapsed = 0.0
                for scheduled_clip, duration in demo_schedule:
                    elapsed += duration
                    if cycle_time < elapsed:
                        demo_clip = scheduled_clip
                        break
                if demo_clip != last_demo_clip:
                    sim.set_humanoid_animation(demo_clip)
                    last_demo_clip = demo_clip
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
