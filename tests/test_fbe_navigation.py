from __future__ import annotations

import numpy as np
import pytest

from stretch_mujoco.navigations import (
    Algorithm,
    NavigationController,
    NavigationPathError,
    OccupancyGrid,
)
from stretch_mujoco.navigations.FBE import (
    FBEPlanner,
    FBEState,
    FrontierCluster,
    LocalOccupancyGrid,
    SimulatedLaserObservationSource,
    best_frontier,
)


def make_grid(occupancy: np.ndarray) -> OccupancyGrid:
    rows, cols = occupancy.shape
    return OccupancyGrid(
        (0.0, float(cols - 1), 0.0, float(rows - 1)),
        occupancy,
        resolution=1.0,
        agent_radius=0.0,
        obstacles=(),
    )


def test_raycast_positive_y_updates_positive_rows() -> None:
    occupancy = np.zeros((7, 7), dtype=bool)
    occupancy[5, 3] = True
    grid = make_grid(occupancy)
    local_map = LocalOccupancyGrid(grid)
    sensor = SimulatedLaserObservationSource(
        grid,
        num_rays=1,
        max_range_cells=3,
        fov_degrees=1.0,
    )

    sensor.update(
        local_map,
        np.array([3.0, 3.0]),
        np.pi / 2,
    )

    assert local_map.data[5, 3] == 1
    assert local_map.data[1, 3] == -1


def test_local_map_accepts_sensor_independent_updates() -> None:
    local_map = LocalOccupancyGrid(make_grid(np.zeros((3, 3), dtype=bool)))

    local_map.update_cells(free_cells=[(1, 1), (1, 2)], obstacle_cells=[(1, 2)])
    local_map.update_from_array(
        np.array(
            [
                [-1, 0, -1],
                [-1, -1, -1],
                [-1, -1, 1],
            ],
            dtype=np.int8,
        )
    )

    assert local_map.data[1, 1] == 0
    assert local_map.data[1, 2] == 1
    assert local_map.data[0, 1] == 0
    assert local_map.data[2, 2] == 1


def test_fbe_accepts_replaceable_observation_source() -> None:
    class FakeObservationSource:
        def __init__(self):
            self.calls = 0

        def update(self, local_map, robot_xy, robot_yaw):
            self.calls += 1
            local_map.update_cells(free_cells=[local_map.world_to_cell(robot_xy)])

    source = FakeObservationSource()
    planner = FBEPlanner(
        make_grid(np.zeros((3, 3), dtype=bool)),
        robot_xy=(1.0, 1.0),
        observation_source=source,
    )

    planner._do_scan()

    assert source.calls == 1
    assert planner.local_map.data[1, 1] == 0


def test_unknown_cells_are_not_traversable() -> None:
    planner = FBEPlanner(make_grid(np.zeros((5, 7), dtype=bool)), robot_xy=(1.0, 2.0))
    planner.local_map.data[:, :] = -1
    planner.local_map.data[2, 1:3] = 0
    planner.local_map.data[2, 4:6] = 0

    planning_grid = planner._local_to_occupancy()

    with pytest.raises(NavigationPathError):
        planner._astar.plan(
            planning_grid,
            np.array([1.0, 2.0]),
            np.array([5.0, 2.0]),
        )


def test_unreachable_frontier_is_not_selected_repeatedly() -> None:
    planner = FBEPlanner(
        make_grid(np.zeros((5, 7), dtype=bool)),
        robot_xy=(1.0, 2.0),
        min_cluster_size=2,
    )
    planner.local_map.data[:, :] = -1
    planner.local_map.data[:, 0:3] = 0
    planner.local_map.data[:, 3] = 1
    planner.local_map.data[:, 4] = 0

    planner._do_detect()
    assert planner.state == FBEState.PLANNING
    failed_cells = set(planner._target_cluster.cells)

    planner._do_plan()
    assert planner.state == FBEState.SCANNING
    assert failed_cells <= planner._blocked_frontier_cells

    planner._stagnant_scans = planner.max_stagnant_scans
    planner._do_detect()
    assert planner.state == FBEState.FINISHED
    assert planner._target_cluster is None


def test_command_publication_does_not_skip_waypoints() -> None:
    planner = FBEPlanner(make_grid(np.zeros((3, 5), dtype=bool)), robot_xy=(0.0, 1.0))
    planner._path = [
        np.array([0.0, 1.0]),
        np.array([2.0, 1.0]),
        np.array([4.0, 1.0]),
    ]
    planner._path_idx = 1

    first = planner.pop_command()
    second = planner.pop_command()

    assert np.array_equal(first, np.array([2.0, 1.0]))
    assert np.array_equal(second, first)
    assert planner._path_idx == 1


def test_new_obstacle_invalidates_active_path_before_motion() -> None:
    occupancy = np.zeros((1, 5), dtype=bool)
    occupancy[0, 2] = True
    planner = FBEPlanner(
        make_grid(occupancy),
        robot_xy=(0.0, 0.0),
        num_rays=1,
        max_range_m=4.0,
        fov_degrees=1.0,
    )
    planner.local_map.data[0, :] = 0
    planner.local_map.data[0, 2] = -1
    planner._path = [np.array([0.0, 0.0]), np.array([4.0, 0.0])]
    planner._path_idx = 1
    planner.state = FBEState.MOVING

    planner._do_scan()

    assert planner.local_map.data[0, 2] == 1
    assert planner.state == FBEState.SCANNING
    assert not planner.has_path()


def test_waypoint_tolerance_cannot_cut_the_next_corner() -> None:
    occupancy = np.zeros((3, 3), dtype=bool)
    occupancy[1, 1] = True
    planner = FBEPlanner(
        make_grid(occupancy),
        robot_xy=(0.9, 0.0),
        num_rays=1,
        max_range_m=1.0,
        fov_degrees=1.0,
    )
    planner.local_map.data[:, :] = occupancy.astype(np.int8)
    planner._path = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 2.0]),
    ]
    planner._path_idx = 1
    planner.state = FBEState.MOVING

    planner.step((0.9, 0.0), 0.0)

    assert planner.state == FBEState.SCANNING
    assert planner.current_target() is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_rays": 0}, "num_rays"),
        ({"max_range_m": 0.0}, "max_range_m"),
        ({"fov_degrees": 361.0}, "fov_degrees"),
        ({"min_cluster_size": 0}, "min_cluster_size"),
        ({"min_frontier_dist": -0.1}, "min_frontier_dist"),
        ({"explore_threshold": 1.1}, "explore_threshold"),
        ({"max_stagnant_scans": 0}, "max_stagnant_scans"),
        ({"robot_xy": (np.nan, 0.0)}, "robot_xy"),
        ({"robot_yaw": np.inf}, "robot_yaw"),
    ],
)
def test_fbe_parameters_are_validated(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        FBEPlanner(make_grid(np.zeros((3, 3), dtype=bool)), **kwargs)


def test_frontier_score_balances_information_gain_against_distance() -> None:
    near = FrontierCluster([(0, col) for col in range(5)], np.array([1.0, 0.0]), 5)
    far = FrontierCluster([(1, col) for col in range(10)], np.array([10.0, 0.0]), 10)

    selected = best_frontier([far, near], np.array([0.0, 0.0]), min_dist=0.0)

    assert selected is near


def test_stagnant_map_without_frontiers_finishes_with_reason() -> None:
    planner = FBEPlanner(
        make_grid(np.zeros((3, 3), dtype=bool)),
        max_stagnant_scans=2,
    )
    planner.local_map.data[:, :] = -1
    planner._stagnant_scans = 2

    planner._do_detect()

    assert planner.state == FBEState.FINISHED
    assert "map stopped changing" in planner.diag.finish_reason


def test_fbe_is_available_through_navigation_controller(monkeypatch) -> None:
    grid = make_grid(np.zeros((3, 3), dtype=bool))
    monkeypatch.setattr(OccupancyGrid, "from_model", lambda *args, **kwargs: grid)

    controller = NavigationController(
        object(),
        object(),
        algorithm=Algorithm.FBE,
        planner_kwargs={"robot_xy": (1.0, 1.0)},
    )

    assert isinstance(controller.explorer, FBEPlanner)
    diagnostics = controller.step_exploration((1.0, 1.0), 0.0)
    assert diagnostics is controller.explorer.diag
    with pytest.raises(RuntimeError, match="without a fixed goal"):
        controller.plan((1.0, 1.0), (2.0, 2.0))
