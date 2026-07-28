"""Stretch 3 body sites for forward-kinematics and grasping queries."""

from stretch_mujoco.robots.base import BodySiteRef, RobotBodySites


class Stretch3Sites(RobotBodySites):
    """Named reference frames on the Stretch 3 robot."""

    LINK_GRASP_CENTER = "link_grasp_center"
    BASE_IMU = "base_imu"
    LIDAR = "lidar"
    ARUCO_LEFT_BASE = "link_aruco_left_base"
    ARUCO_RIGHT_BASE = "link_aruco_right_base"
    ARUCO_TOP_WRIST = "link_aruco_top_wrist"
    ARUCO_INNER_WRIST = "link_aruco_inner_wrist"
    DOCKING_STATION = "docking_station"

    # -- _SiteRef properties -------------------------------------------

    @property
    def site_ref(self) -> BodySiteRef:
        _map: dict[Stretch3Sites, BodySiteRef] = {
            Stretch3Sites.LINK_GRASP_CENTER: BodySiteRef(
                "link_grasp_center", "body", "ee_grasp"
            ),
            Stretch3Sites.BASE_IMU: BodySiteRef("base_imu", "site", "imu"),
            Stretch3Sites.LIDAR: BodySiteRef("lidar", "site", "lidar"),
            Stretch3Sites.ARUCO_LEFT_BASE: BodySiteRef(
                "link_aruco_left_base", "body", "marker"
            ),
            Stretch3Sites.ARUCO_RIGHT_BASE: BodySiteRef(
                "link_aruco_right_base", "body", "marker"
            ),
            Stretch3Sites.ARUCO_TOP_WRIST: BodySiteRef(
                "link_aruco_top_wrist", "body", "marker"
            ),
            Stretch3Sites.ARUCO_INNER_WRIST: BodySiteRef(
                "link_aruco_inner_wrist", "body", "marker"
            ),
            Stretch3Sites.DOCKING_STATION: BodySiteRef(
                "docking_station", "body", "docking"
            ),
        }
        return _map[self]

    @staticmethod
    def grasp_sites() -> list["Stretch3Sites"]:
        return [Stretch3Sites.LINK_GRASP_CENTER]

    @staticmethod
    def camera_sites() -> list["Stretch3Sites"]:
        return []  # Stretch camera mounts are tied to camera enum, not sites

    @staticmethod
    def all() -> list["Stretch3Sites"]:
        return [s for s in Stretch3Sites]
