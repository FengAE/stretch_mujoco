"""Stanford TidyBot body sites for forward-kinematics and grasping queries."""

from stretch_mujoco.robots.base import BodySiteRef, RobotBodySites


class TidyBotSites(RobotBodySites):
    """Named reference frames on the Stanford TidyBot."""

    PINCH_SITE = "pinch_site"

    @property
    def site_ref(self) -> BodySiteRef:
        _map: dict[TidyBotSites, BodySiteRef] = {
            TidyBotSites.PINCH_SITE: BodySiteRef("pinch_site", "site", "ee_grasp"),
        }
        return _map[self]

    @staticmethod
    def grasp_sites() -> list["TidyBotSites"]:
        return [TidyBotSites.PINCH_SITE]

    @staticmethod
    def camera_sites() -> list["TidyBotSites"]:
        return []

    @staticmethod
    def all() -> list["TidyBotSites"]:
        return [s for s in TidyBotSites]
