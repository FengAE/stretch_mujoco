"""Google Robot body sites for forward-kinematics and grasping queries."""

from stretch_mujoco.robots.base import BodySiteRef, RobotBodySites


class GoogleRobotSites(RobotBodySites):
    """Named reference frames on the Google Robot."""

    WRIST = "wrist"
    GRIPPER = "gripper"
    HEAD_CAMERA = "head_camera_site"
    BASE_IMU = "base_imu_site"
    NECK_IMU = "neck_imu_site"

    @property
    def site_ref(self) -> BodySiteRef:
        _map: dict[GoogleRobotSites, BodySiteRef] = {
            GoogleRobotSites.WRIST: BodySiteRef("wrist", "site", "generic"),
            GoogleRobotSites.GRIPPER: BodySiteRef("gripper", "site", "ee_grasp"),
            GoogleRobotSites.HEAD_CAMERA: BodySiteRef("head_camera_site", "site", "camera"),
            GoogleRobotSites.BASE_IMU: BodySiteRef("base_imu_site", "site", "imu"),
            GoogleRobotSites.NECK_IMU: BodySiteRef("neck_imu_site", "site", "imu"),
        }
        return _map[self]

    @staticmethod
    def grasp_sites() -> list["GoogleRobotSites"]:
        return [GoogleRobotSites.GRIPPER]

    @staticmethod
    def camera_sites() -> list["GoogleRobotSites"]:
        return [GoogleRobotSites.HEAD_CAMERA]

    @staticmethod
    def all() -> list["GoogleRobotSites"]:
        return [s for s in GoogleRobotSites]
