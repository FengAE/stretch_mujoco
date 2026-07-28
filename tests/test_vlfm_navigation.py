from __future__ import annotations

import io
import math
import types

import mujoco
import numpy as np
import pytest
from PIL import Image

import stretch_mujoco.navigations.VLFM.vlm_client as vlm_client_module
from stretch_mujoco.navigations import Algorithm, NavigationController, OccupancyGrid
from stretch_mujoco.navigations.FBE import FrontierCluster
from stretch_mujoco.navigations.VLFM import (
    AcyclicEnforcer,
    BLIP2ITMVLMClient,
    CLIPVLMClient,
    ImageCapture,
    SigLIPVLMClient,
    VLFMPlanner,
    VLMClient,
    ValueMap,
)


class ConstantVLM(VLMClient):
    def __init__(self, score: float = 0.75) -> None:
        self.score = score

    def score_image(self, image: bytes, text: str) -> float:
        return self.score


class FailingVLM(VLMClient):
    def score_image(self, image: bytes, text: str) -> float:
        raise RuntimeError("backend unavailable")


def make_grid(
    bounds: tuple[float, float, float, float] = (0.0, 4.0, 0.0, 4.0),
) -> OccupancyGrid:
    return OccupancyGrid(
        bounds,
        np.zeros((5, 5), dtype=bool),
        resolution=1.0,
        agent_radius=0.0,
        obstacles=(),
    )


def test_value_map_uses_navigation_map_world_origin() -> None:
    value_map = ValueMap((5, 5), 1.0, origin_xy=(10.0, 20.0))

    assert value_map._world_to_cell(np.array([10.0, 20.0])) == (0, 0)
    assert value_map._world_to_cell(np.array([14.0, 24.0])) == (4, 4)

    value_map.update(
        np.array([0.8], dtype=np.float32),
        np.full((8, 16), 2.0, dtype=np.float32),
        camera_xy=(10.0, 20.0),
        camera_yaw=0.0,
        hfov_rad=math.radians(70.0),
        min_depth=0.1,
        max_depth=3.0,
    )

    assert value_map.get_value_at(np.array([11.0, 20.0])) == pytest.approx(0.8)


def test_depth_boundary_does_not_mark_cells_behind_obstacle() -> None:
    value_map = ValueMap((101, 101), 0.1)
    local_mask = value_map._process_local_data(
        np.full((20, 40), 2.0, dtype=np.float32),
        hfov_rad=math.radians(70.0),
        min_depth=0.1,
        max_depth=5.0,
    )
    centre = local_mask.shape[0] // 2

    assert local_mask[centre, centre + 10] > 0
    assert local_mask[centre, centre + 30] == 0
    assert np.where(local_mask[centre] > 0)[0].max() <= centre + 21


def test_value_projection_stops_at_unknown_and_obstacle_cells() -> None:
    value_map = ValueMap((11, 11), 1.0, origin_xy=(0.0, 0.0))
    value_map._values[5, 7, 0] = 0.4
    value_map._conf_map[5, 7] = 1.0
    free = np.zeros((11, 11), dtype=bool)
    obstacles = np.zeros_like(free)
    # Camera and near side are known free. Cells behind the wall were explored
    # previously, so the ray check (not just the target mask) must reject them.
    free[5, 1:5] = True
    obstacles[5, 5] = True
    free[5, 6:10] = True

    value_map.update(
        np.array([0.8], dtype=np.float32),
        np.full((8, 16), 9.0, dtype=np.float32),
        camera_xy=(1.0, 5.0),
        camera_yaw=0.0,
        hfov_rad=math.radians(60.0),
        min_depth=0.1,
        max_depth=9.0,
        free_mask=free,
        obstacle_mask=obstacles,
    )

    assert value_map._values[5, 3, 0] == pytest.approx(0.8)
    assert value_map._values[5, 5, 0] == 0.0
    assert value_map._values[5, 7, 0] == 0.0
    assert np.all(value_map._values[~free] == 0.0)


def test_unknown_gap_blocks_previously_known_free_target() -> None:
    value_map = ValueMap((9, 9), 1.0, origin_xy=(0.0, 0.0))
    free = np.zeros((9, 9), dtype=bool)
    free[4, 1:4] = True
    free[4, 5:8] = True

    value_map.update(
        np.array([0.6], dtype=np.float32),
        np.full((6, 12), 7.0, dtype=np.float32),
        camera_xy=(1.0, 4.0),
        camera_yaw=0.0,
        hfov_rad=math.radians(50.0),
        min_depth=0.1,
        max_depth=7.0,
        free_mask=free,
        obstacle_mask=np.zeros_like(free),
    )

    assert value_map._values[4, 2, 0] == pytest.approx(0.6)
    assert value_map._values[4, 6, 0] == 0.0


def test_waypoint_sampling_ignores_blocked_or_unknown_cells() -> None:
    value_map = ValueMap((5, 5), 1.0, origin_xy=(0.0, 0.0))
    value_map._values[2, 3, 0] = 1.0
    value_map._values[2, 2, 0] = 0.2
    known_free = np.zeros((5, 5), dtype=bool)
    known_free[2, 2] = True

    _, values = value_map.sort_waypoints(np.array([[2.0, 2.0]]), radius=1.5, valid_mask=known_free)

    assert values == pytest.approx([0.2])


def test_planner_value_map_uses_local_map_bounds() -> None:
    planner = VLFMPlanner(make_grid((10.0, 14.0, 20.0, 24.0)), ConstantVLM())

    assert planner._value_map._world_to_cell(np.array([10.0, 20.0])) == (0, 0)


def test_projection_failure_is_reported_and_raised() -> None:
    planner = VLFMPlanner(make_grid(), ConstantVLM())

    with pytest.raises(RuntimeError, match="Value Map projection failed"):
        planner.inject_observation(
            b"rgb",
            np.empty((0, 0), dtype=np.float32),
            (0.0, 0.0),
            0.0,
            math.radians(80.0),
        )

    assert "depth_img" in planner.diag.last_projection_error


def test_successful_projection_reports_updates_and_observed_cells() -> None:
    planner = VLFMPlanner(make_grid(), ConstantVLM())
    planner.local_map.data.fill(0)
    planner.inject_observation(
        b"rgb",
        np.full((4, 8), 2.0),
        (1.0, 1.0),
        0.0,
        math.radians(70),
    )

    assert planner.diag.value_map_updates == 1
    assert planner.diag.value_map_observed_cells > 0


def test_precomputed_score_can_be_injected_without_vlm_call() -> None:
    planner = VLFMPlanner(make_grid(), FailingVLM())

    score = planner.inject_scored_observation(
        0.8,
        np.full((4, 8), 2.0),
        (1.0, 1.0),
        0.0,
        math.radians(70),
    )

    assert score == pytest.approx(0.8)
    assert planner.diag.vlm_query_count == 1


def test_vlm_failure_is_explicit_unless_fallback_is_enabled() -> None:
    strict = VLFMPlanner(make_grid(), FailingVLM())
    with pytest.raises(RuntimeError, match="backend unavailable"):
        strict.inject_observation(b"rgb", np.ones((2, 2)), (0.0, 0.0), 0.0, math.radians(70))
    assert "backend unavailable" in strict.diag.last_vlm_error

    fallback = VLFMPlanner(make_grid(), FailingVLM(), allow_geometric_fallback=True)
    with pytest.warns(RuntimeWarning, match="geometric frontier"):
        assert (
            fallback.inject_observation(b"rgb", np.ones((2, 2)), (0.0, 0.0), 0.0, math.radians(70))
            == 0.0
        )


@pytest.mark.parametrize("score", [-0.1, 1.1, np.nan])
def test_planner_rejects_vlm_scores_outside_unit_interval(score: float) -> None:
    planner = VLFMPlanner(make_grid(), ConstantVLM(score))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        planner.inject_observation(b"rgb", np.ones((2, 2)), (0.0, 0.0), 0.0, math.radians(70))


def test_vertical_fov_is_converted_to_horizontal_fov(monkeypatch) -> None:
    class FakeRenderer:
        def __init__(self, model, height, width):
            pass

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    capture = ImageCapture(model, mujoco.MjData(model), width=320, height=240)

    expected = 2.0 * math.atan((320 / 240) * math.tan(math.radians(70) / 2.0))
    assert capture.horizontal_fov_rad(70.0) == pytest.approx(expected)
    assert capture.horizontal_fov_rad(70.0) > math.radians(70.0)


def test_rgbd_capture_preserves_rgb_and_restores_render_mode(monkeypatch) -> None:
    expected_rgb = np.array(
        [[[255, 0, 0], [0, 128, 255]], [[12, 34, 56], [200, 210, 220]]],
        dtype=np.uint8,
    )
    expected_depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    class FakeRenderer:
        def __init__(self, model, height, width):
            self.depth = False

        def update_scene(self, data, camera, scene_option):
            self.camera = camera

        def enable_depth_rendering(self):
            self.depth = True

        def disable_depth_rendering(self):
            self.depth = False

        def render(self):
            return expected_depth.copy() if self.depth else expected_rgb.copy()

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    capture = ImageCapture(model, mujoco.MjData(model), width=2, height=2)

    encoded, depth = capture.capture_at_robot((0.0, 0.0), 0.0)
    decoded = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))

    assert np.array_equal(decoded, expected_rgb)
    assert np.array_equal(depth, expected_depth)
    assert capture._renderer.depth is False


def test_capture_rgb_array_skips_png_round_trip(monkeypatch) -> None:
    expected = np.full((2, 3, 3), 73, dtype=np.uint8)

    class FakeRenderer:
        def __init__(self, model, height, width):
            pass

        def update_scene(self, data, camera, scene_option):
            pass

        def render(self):
            return expected.copy()

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    capture = ImageCapture(model, mujoco.MjData(model), width=3, height=2)

    rgb = capture.capture_rgb_array((0.0, 0.0, 1.0), 0.0)

    assert np.array_equal(rgb, expected)


def test_capture_at_pose_is_an_instance_method(monkeypatch) -> None:
    class FakeRenderer:
        def __init__(self, model, height, width):
            pass

        def update_scene(self, data, camera, scene_option):
            pass

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    capture = ImageCapture(model, mujoco.MjData(model), width=2, height=2)

    assert isinstance(capture.capture_at_pose((0.0, 0.0, 1.0), 0.0), bytes)


def test_named_camera_observation_uses_actual_camera_pose(monkeypatch) -> None:
    class FakeRenderer:
        def __init__(self, model, height, width):
            pass

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="nav" pos="2 3 1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    capture = ImageCapture(model, data, width=4, height=3, camera_name="nav")

    camera_xy, camera_yaw = capture.camera_pose_xy_yaw((99.0, 99.0), 1.2)

    assert camera_xy == pytest.approx((2.0, 3.0))
    assert np.isfinite(camera_yaw)


class _Batch(dict):
    def to(self, device):
        return self


class _HTTPResponse:
    def __init__(self, payload: dict) -> None:
        import json

        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def test_blip2_http_client_health_and_score(monkeypatch) -> None:
    import base64
    import json

    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/health"):
            return _HTTPResponse({"status": "ready"})
        payload = json.loads(request.data.decode("utf-8"))
        assert base64.b64decode(payload["image_b64"]) == b"encoded-image"
        assert payload["text"] == "a chair"
        return _HTTPResponse({"score": 0.72, "raw_cosine": 0.44})

    client = BLIP2ITMVLMClient(base_url="http://127.0.0.1:12182", timeout=12.0)
    monkeypatch.setattr(client._opener, "open", fake_urlopen)

    client.prepare()
    assert client.score_image(b"encoded-image", "a chair") == pytest.approx(0.72)
    assert requests[0][1] == 5.0
    assert requests[1][1] == 12.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": ""},
        {"base_url": "127.0.0.1:12182"},
        {"timeout": 0.0},
    ],
)
def test_blip2_http_client_validates_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        BLIP2ITMVLMClient(**kwargs)


def test_blip2_http_client_rejects_invalid_service_score(monkeypatch) -> None:
    client = BLIP2ITMVLMClient()
    monkeypatch.setattr(
        client._opener,
        "open",
        lambda request, timeout: _HTTPResponse({"score": -0.1}),
    )

    with pytest.raises(RuntimeError, match="invalid score"):
        client.score_image(b"image", "chair")


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_clip_client_returns_unit_interval_cosine_score() -> None:
    client = CLIPVLMClient()
    client._processor = lambda **kwargs: _Batch()
    client._model = lambda **kwargs: types.SimpleNamespace(
        image_embeds=__import__("torch").tensor([[1.0, 0.0]]),
        text_embeds=__import__("torch").tensor([[0.0, 1.0]]),
    )

    assert client.score_image(_png_bytes(), "chair") == pytest.approx(0.5)


def test_siglip_client_returns_sigmoid_probability() -> None:
    torch = __import__("torch")
    client = SigLIPVLMClient()
    client._processor = lambda **kwargs: _Batch()
    client._model = lambda **kwargs: types.SimpleNamespace(logits_per_text=torch.tensor([[2.0]]))

    assert client.score_image(_png_bytes(), "chair") == pytest.approx(
        float(torch.sigmoid(torch.tensor(2.0)))
    )


def test_siglip_prepare_lazy_loads_model_and_processor(monkeypatch) -> None:
    class FakeModel:
        def eval(self):
            self.eval_called = True

    fake_model = FakeModel()
    fake_processor = object()
    monkeypatch.setattr(
        vlm_client_module,
        "_load_model_with_fallback",
        lambda model_cls, model_name, device: fake_model,
    )
    monkeypatch.setattr(
        vlm_client_module,
        "_load_processor_with_fallback",
        lambda processor_cls, model_name: fake_processor,
    )

    client = SigLIPVLMClient()
    client.prepare()

    assert client._model is fake_model
    assert client._processor is fake_processor
    assert fake_model.eval_called is True


def test_vlfm_is_registered_with_navigation_controller(monkeypatch) -> None:
    grid = make_grid()
    monkeypatch.setattr(OccupancyGrid, "from_model", lambda *args, **kwargs: grid)
    controller = NavigationController(
        object(),
        object(),
        algorithm=Algorithm.VLFM,
        planner_kwargs={"vlm_client": ConstantVLM(), "robot_xy": (1.0, 1.0)},
    )

    assert isinstance(controller.explorer, VLFMPlanner)
    assert controller.step_exploration((1.0, 1.0), 0.0) is controller.explorer.diag


def test_all_cyclic_frontiers_fall_back_to_closest_in_sorted_order(monkeypatch) -> None:
    planner = VLFMPlanner(make_grid(), ConstantVLM(), sticky_last_frontier=False)
    planner._robot_xy = np.array([0.0, 0.0])
    far = FrontierCluster([(0, 4)], np.array([4.0, 0.0]), 1)
    close = FrontierCluster([(0, 1)], np.array([1.0, 0.0]), 1)
    middle = FrontierCluster([(0, 2)], np.array([2.0, 0.0]), 1)
    planner._value_map._update_count = 1
    monkeypatch.setattr(
        planner._value_map,
        "sort_waypoints",
        lambda *args, **kwargs: (
            np.array([close.centroid_xy, far.centroid_xy, middle.centroid_xy]),
            [0.9, 0.8, 0.7],
        ),
    )
    monkeypatch.setattr(planner._acyclic, "check_cyclic", lambda *args: True)

    assert planner._select_frontier([far, close, middle]) is close


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance_threshold": 0.0},
        {"value_threshold": -0.1},
        {"max_history": 0},
    ],
)
def test_acyclic_enforcer_validates_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        AcyclicEnforcer(**kwargs)


def test_demo_constructor_options_are_supported() -> None:
    planner = VLFMPlanner(
        make_grid(),
        ConstantVLM(),
        instruction="Find a chair",
        robot_xy=(1.0, 1.0),
        num_rays=120,
        max_range_m=5.0,
        fov_degrees=270.0,
        min_cluster_size=3,
        explore_threshold=0.85,
        allow_geometric_fallback=False,
    )

    assert planner.instruction == "Find a chair"
