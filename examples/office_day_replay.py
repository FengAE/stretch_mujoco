"""Visually replay a compressed LLM-generated NPC workday in MuJoCo."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import click
import cv2

from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.stretch_cameras import StretchCameras


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = ROOT / "stretch_mujoco" / "models" / "office_day_replay.json"
OFFICE_SCENE = ROOT / "stretch_mujoco" / "models" / "office_scene.xml"
LUNCH_OBJECT = "bread_snack"


@dataclass(frozen=True)
class ReplayItem:
    item_id: str
    start_minute: int
    end_minute: int
    activity: str
    location: str


def parse_clock(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def format_clock(minute: float) -> str:
    value = int(minute) % (24 * 60)
    return f"{value // 60:02d}:{value % 60:02d}"


def load_replay(path: Path) -> tuple[int, int, tuple[ReplayItem, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    start = parse_clock(payload["workday"]["start"])
    end = parse_clock(payload["workday"]["end"])
    items = tuple(
        ReplayItem(
            item_id=item["id"],
            start_minute=parse_clock(item["start"]),
            end_minute=parse_clock(item["end"]),
            activity=item["activity"],
            location=item["location"],
        )
        for item in payload["schedule"]
    )
    if end <= start:
        raise ValueError("Replay workday end must be after its start")
    return start, end, items


def active_item(items: tuple[ReplayItem, ...], minute: float) -> ReplayItem | None:
    current = next(
        (item for item in items if item.start_minute <= minute < item.end_minute),
        None,
    )
    if current is not None:
        return current
    previous = [item for item in items if item.end_minute <= minute]
    future = [item for item in items if item.start_minute > minute]
    if previous and future:
        return previous[-1]
    return None


def item_progress(item: ReplayItem, minute: float) -> float:
    return min(
        max((minute - item.start_minute) / (item.end_minute - item.start_minute), 0.0),
        1.0,
    )


def visual_action(item: ReplayItem | None, minute: float) -> str:
    if item is None:
        return "idle"
    if item.activity == "work":
        return "work"
    if item.location == "snack_counter":
        progress = item_progress(item, minute)
        if progress < 0.35:
            return "walk"
        if progress < 0.78:
            return "eat"
        return "idle"
    if item.location in {"chair_left", "chair_right"}:
        return "sit"
    return "walk"


def apply_item(
    sim: StretchMujocoSimulator,
    item: ReplayItem | None,
    action: str | None = None,
) -> None:
    action = action or visual_action(item, item.start_minute if item else 0.0)
    if item is None:
        sim.set_humanoid_animation("idle")
        return
    if item.location in {"workstation_left", "workstation_right"}:
        chair = "chair_left" if item.location.endswith("left") else "chair_right"
        sim.set_humanoid_sit_target(chair)
        sim.set_humanoid_animation("work")
        return
    if item.location in {"chair_left", "chair_right"}:
        sim.set_humanoid_sit_target(item.location)
        sim.set_humanoid_animation("sit")
        return
    if action in {"eat", "idle"}:
        sim.set_humanoid_animation(action)
        return
    sim.set_humanoid_navigation_target(item.location)
    sim.set_humanoid_animation("walk")


def annotate_frame(
    frame,
    office_minute: float,
    item: ReplayItem | None,
    action: str | None = None,
):
    activity = (action or ("idle" if item is None else item.activity)).upper()
    location = "office" if item is None else item.location
    label = "Schedule gap" if item is None else item.item_id
    annotated = frame.copy()
    cv2.rectangle(annotated, (18, 16), (550, 96), (20, 24, 28), thickness=-1)
    cv2.putText(
        annotated,
        f"OFFICE DAY  {format_clock(office_minute)}",
        (34, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"{activity}  |  {location}  |  {label}",
        (34, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 210, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


@click.command()
@click.option(
    "--replay",
    "replay_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_REPLAY,
    show_default=True,
)
@click.option(
    "--duration",
    type=float,
    default=120.0,
    show_default=True,
    help="Wall-clock seconds used to replay the nine-hour workday.",
)
@click.option("--loop", "loop_replay", is_flag=True, help="Restart after 18:00.")
@click.option("--headless", is_flag=True, help="Run without opening the MuJoCo viewer.")
@click.option("--show-viewer-ui", is_flag=True, help="Show MuJoCo control panels.")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write an MP4 using the fixed office overview camera.",
)
@click.option("--fps", type=click.IntRange(1, 60), default=24, show_default=True)
def main(
    replay_path: Path,
    duration: float,
    loop_replay: bool,
    headless: bool,
    show_viewer_ui: bool,
    output: Path | None,
    fps: int,
) -> None:
    """Replay one employee's LLM-generated office schedule."""
    if duration <= 0:
        raise click.BadParameter("duration must be positive")
    if output is not None and loop_replay:
        raise click.BadParameter("--loop cannot be combined with --output")
    start_minute, end_minute, items = load_replay(replay_path)
    recording = output is not None
    cameras = [StretchCameras.office_overview_rgb] if recording else []
    sim = StretchMujocoSimulator(
        scene_xml_path=str(OFFICE_SCENE),
        camera_hz=fps,
        cameras_to_use=cameras,
    )
    writer = None
    playback_speed = max(1.0, (end_minute - start_minute) / duration)
    sim.set_humanoid_playback_speed(playback_speed)
    sim.restore_office_object(LUNCH_OBJECT)

    try:
        sim.start(headless=headless or recording, show_viewer_ui=show_viewer_ui)
        while sim.is_running():
            sim.restore_office_object(LUNCH_OBJECT)
            replay_started = time.monotonic()
            previous_visual_state = object()
            lunch_consumed = False
            written_frames = 0
            target_frames = round(duration * fps)
            while sim.is_running():
                elapsed = time.monotonic() - replay_started
                phase = min(elapsed / duration, 1.0)
                office_minute = start_minute + phase * (end_minute - start_minute)
                item = active_item(items, office_minute)
                action = visual_action(item, office_minute)
                visual_state = (item, action)
                if visual_state != previous_visual_state:
                    apply_item(sim, item, action)
                    activity = action
                    location = "office" if item is None else item.location
                    label = "schedule gap" if item is None else item.item_id
                    click.echo(
                        f"[{format_clock(office_minute)}] {activity:<7} " f"{location:<20} {label}"
                    )
                    previous_visual_state = visual_state
                if (
                    item is not None
                    and item.location == "snack_counter"
                    and item_progress(item, office_minute) >= 0.78
                    and not lunch_consumed
                ):
                    sim.consume_office_object(LUNCH_OBJECT)
                    lunch_consumed = True
                    click.echo(f"[{format_clock(office_minute)}] consumed {LUNCH_OBJECT}")
                if recording:
                    expected_frames = min(int(elapsed * fps) + 1, target_frames)
                    if written_frames < expected_frames:
                        camera_data = sim.pull_camera_data()
                        frame = camera_data.get_camera_data(StretchCameras.office_overview_rgb)
                        annotated = annotate_frame(frame, office_minute, item, action)
                        if writer is None:
                            output.parent.mkdir(parents=True, exist_ok=True)
                            height, width = annotated.shape[:2]
                            writer = cv2.VideoWriter(
                                str(output),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                fps,
                                (width, height),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(f"Cannot open MP4 writer for {output}")
                        while written_frames < expected_frames:
                            writer.write(annotated)
                            written_frames += 1
                if phase >= 1.0 and (not recording or written_frames >= target_frames):
                    sim.set_humanoid_animation("idle")
                    click.echo("[18:00] replay complete")
                    break
                time.sleep(min(0.02, 0.5 / fps))
            if not loop_replay:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        sim.stop()
    if output is not None:
        click.echo(f"MP4 saved to {output.resolve()}")


if __name__ == "__main__":
    main()
