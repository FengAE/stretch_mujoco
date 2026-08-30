import random

import numpy as np

from examples.collect_random_object_grasp_episodes import (
    EXCLUDED_OBJECTS,
    GRASP_YAW_ALIGNMENT,
    HORIZONTAL_SEED,
    TARGET_X_RANGE,
    TARGET_Y_RANGE,
    _world_grasp_point,
    _grasp_roll,
    eligible_objects,
    sample_episode_scene,
    task_prompt,
)
from stretch_mujoco.graspgen.ik import StretchGraspIK


def test_eligible_objects_exclude_keyboard_and_calculator():
    objects = eligible_objects()

    assert len(objects) == 11
    assert not EXCLUDED_OBJECTS.intersection(objects)
    assert task_prompt("038_milk-box") == "pick up the milk box"
    assert task_prompt("green_apple") == "pick up the green apple"
    assert _grasp_roll("043_book") == 0.0
    assert _grasp_roll("071_can") == 0.0


def test_sampled_scene_has_reachable_target_and_nonoverlapping_distractors():
    ik = StretchGraspIK()
    scene = sample_episode_scene("071_can", random.Random(4), ik=ik)

    assert len(scene.objects) in (3, 4)
    assert scene.objects[0] == "071_can"
    target = scene.placements[scene.target]
    assert TARGET_X_RANGE[0] <= target.x <= TARGET_X_RANGE[1]
    assert TARGET_Y_RANGE[0] <= target.y <= TARGET_Y_RANGE[1]
    np.testing.assert_allclose(_world_grasp_point(scene.target, target), scene.grasp_point)
    assert len(scene.target_joints) == 6
    assert all(name not in EXCLUDED_OBJECTS for name in scene.objects)


def test_every_target_can_sample_an_ik_reachable_scene():
    rng = random.Random(11)
    ik = StretchGraspIK()

    for target in eligible_objects():
        scene = sample_episode_scene(target, rng, ik=ik)
        assert scene.target == target
        assert np.isfinite(scene.target_joints).all()


def test_gripper_closing_axis_tracks_object_yaw():
    ik = StretchGraspIK()
    target_yaw = 0.73
    seed_pose = ik.forward(HORIZONTAL_SEED)
    cosine, sine = np.cos(target_yaw + GRASP_YAW_ALIGNMENT), np.sin(
        target_yaw + GRASP_YAW_ALIGNMENT
    )
    yaw_rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    closing_axis = (yaw_rotation @ seed_pose[:3, :3])[:2, 1]

    np.testing.assert_allclose(closing_axis, [-np.sin(target_yaw), np.cos(target_yaw)], atol=2e-5)
