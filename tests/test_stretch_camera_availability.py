from pathlib import Path

import mujoco

from stretch_mujoco.enums.stretch_cameras import StretchCameras


MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


def test_default_scene_filters_office_only_camera() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODELS / "scene.xml"))

    available = StretchCameras.from_mjmodel(model)

    assert StretchCameras.office_overview_rgb not in available
    assert set(available) == set(StretchCameras.all()) - {StretchCameras.office_overview_rgb}


def test_office_scene_exposes_overview_camera() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODELS / "office_scene.xml"))

    assert StretchCameras.office_overview_rgb in StretchCameras.from_mjmodel(model)
