"""VLFM (Vision-Language Frontier Maps) planner.

Aligned with the upstream `rai-opensource/vlfm <https://github.com/rai-opensource/vlfm>`_
ITM (Image-Text Matching) policy pattern (ICRA 2024).  The core idea is:

1. At each decision step, capture the current camera image.
2. Score it against the language instruction via a VLM (e.g. CLIP / SigLIP).
3. Project that scalar score through the camera FOV onto a spatial
   :class:`ValueMap` using the depth image.
4. Detect frontiers and sort them by their accumulated value-map scores.
5. Use an :class:`AcyclicEnforcer` to prevent frontier oscillation.
6. Navigate to the best frontier with A*.

This replaces the original "panorama + direction-selection" approach with
the upstream per-frame value-map accumulation pattern.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from stretch_mujoco.navigations.FBE.frontier import (
    FrontierCluster,
    best_frontier,
    cluster_frontiers,
    detect_frontier_cells,
)
from stretch_mujoco.navigations.FBE.planner import (
    FBEPlanner,
    FBEDiagnostics,
    FBEState,
)
from stretch_mujoco.navigations.VLFM.acyclic_enforcer import AcyclicEnforcer
from stretch_mujoco.navigations.VLFM.value_map import ValueMap
from stretch_mujoco.navigations.VLFM.vlm_client import VLMClient


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class VLFMDiagnostics(FBEDiagnostics):
    """Extra diagnostics for VLFM debugging and visualisation."""

    vlm_query_count: int = 0
    last_vlm_score: float = 0.0
    last_vlm_text: str = ""
    last_vlm_error: str = ""
    last_projection_error: str = ""
    value_map_updates: int = 0
    value_map_observed_cells: int = 0
    selected_value: float = 0.0
    suppressed_cyclic: int = 0
    mode: str = "explore"  # "explore" | "seek"


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class VLFMPlanner(FBEPlanner):
    """Vision-Language Frontier Maps exploration planner.

    Inherits the FBE state machine (SCANNING → PLANNING → MOVING) and
    overrides frontier selection to incorporate VLM-guided value-map
    scoring, closely following the upstream ITMPolicyV2 / ITMPolicyV3
    pattern.

    Parameters
    ----------
    god_grid:
        Ground-truth occupancy grid for simulated ray-casting.
    vlm_client:
        A :class:`VLMClient` that provides ``score_image(image, text) -> float``.
    instruction:
        Natural-language target description (e.g. "Seems like there is a chair ahead.").
    use_value_map:
        When True (default), VLM scores are projected through the FOV onto
        a spatial value map.  When False, every frontier inherits the most
        recent VLM score uniformly.
    exploration_thresh:
        When set, the planner uses two value channels: *target* (ch 0) and
        *exploration* (ch 1).  If the best target value is below this
        threshold the planner explores; once a strong target signal appears
        it switches to seeking.  Set to 0.0 for pure target-driven behaviour.
    value_map_diffusion_radius:
        Radius in grid cells for diffusing value scores around frontier cells
        (for backward-compatible :class:`LanguageValueMap` behaviour when
        *use_value_map* is False).
    **kwargs:
        Forwarded to :class:`FBEPlanner`.
    """

    def __init__(
        self,
        god_grid,
        vlm_client: VLMClient,
        *,
        instruction: str = "Seems like there is a target_object ahead.",
        use_value_map: bool = True,
        exploration_thresh: float = 0.0,
        value_map_diffusion_radius: int = 2,
        value_map_fusion: str = "max_confidence",
        sticky_last_frontier: bool = True,
        allow_geometric_fallback: bool = False,
        **kwargs,
    ) -> None:
        if not isinstance(vlm_client, VLMClient):
            raise TypeError("vlm_client must implement VLMClient")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if not isinstance(use_value_map, bool):
            raise TypeError("use_value_map must be a bool")
        if not np.isfinite(exploration_thresh) or not 0 <= exploration_thresh <= 1:
            raise ValueError("exploration_thresh must be in [0, 1]")
        if not isinstance(sticky_last_frontier, bool):
            raise TypeError("sticky_last_frontier must be a bool")
        if value_map_fusion not in {"max_confidence", "weighted_average"}:
            raise ValueError("value_map_fusion must be 'max_confidence' or 'weighted_average'")
        if not isinstance(allow_geometric_fallback, bool):
            raise TypeError("allow_geometric_fallback must be a bool")

        super().__init__(god_grid, **kwargs)
        self.vlm = vlm_client
        self.instruction = instruction.strip()
        self._use_value_map = use_value_map
        self._exploration_thresh = float(exploration_thresh)
        self._sticky_last_frontier = sticky_last_frontier
        self.allow_geometric_fallback = allow_geometric_fallback

        # Value channels: 2 if using exploration threshold, else 1
        value_channels = 2 if exploration_thresh > 0 else 1
        self._value_map = ValueMap(
            self.local_map.shape,
            self.local_map.resolution,
            origin_xy=(self.local_map.x_min, self.local_map.y_min),
            value_channels=value_channels,
            use_max_confidence=(value_map_fusion == "max_confidence"),
        )

        # Legacy value map for backward compatibility
        from stretch_mujoco.navigations.VLFM.value_map import LanguageValueMap

        self._legacy_value_map = LanguageValueMap(
            self.local_map,
            diffusion_radius=value_map_diffusion_radius,
        )

        self._acyclic = AcyclicEnforcer()

        # State for sticky frontier tracking
        self._last_frontier: Optional[np.ndarray] = None
        self._last_value: float = float("-inf")
        self._vlm_scores: list[float] = []

        self.diag = VLFMDiagnostics()

    # ------------------------------------------------------------------
    # Public API — image & depth injection
    # ------------------------------------------------------------------
    def inject_observation(
        self,
        rgb_image: bytes,
        depth_image: np.ndarray,
        camera_xy: tuple[float, float],
        camera_yaw: float,
        hfov_rad: float,
        min_depth: float = 0.1,
        max_depth: float = 5.0,
    ) -> float:
        """Score the current camera image with the VLM and update the value map.

        Call this **outside** the MuJoCo physics loop, once per decision step,
        after capturing RGB and depth from the robot's camera.

        Parameters
        ----------
        rgb_image:
            PNG/JPEG bytes of the current egocentric view.
        depth_image:
            2-D depth array ``(H, W)`` in metres.
        camera_xy:
            World ``(x, y)`` of the camera.
        camera_yaw:
            Camera yaw in radians.
        hfov_rad:
            Horizontal FOV in radians.
        min_depth, max_depth:
            Valid depth range (metres).

        Returns
        -------
        float
            The VLM score for this observation (higher = more relevant).
        """
        # Score the image against the instruction
        try:
            vlm_score = self.vlm.score_image(rgb_image, self.instruction)
            self.diag.last_vlm_error = ""
        except Exception as exc:
            self.diag.last_vlm_error = f"{type(exc).__name__}: {exc}"
            if not self.allow_geometric_fallback:
                raise RuntimeError(f"VLM query failed: {exc}") from exc
            warnings.warn(
                f"VLM query failed; using geometric frontier selection: {exc}",
                RuntimeWarning,
            )
            return 0.0

        return self.inject_scored_observation(
            vlm_score,
            depth_image,
            camera_xy,
            camera_yaw,
            hfov_rad,
            min_depth=min_depth,
            max_depth=max_depth,
        )

    def inject_scored_observation(
        self,
        vlm_score: float,
        depth_image: np.ndarray,
        camera_xy: tuple[float, float],
        camera_yaw: float,
        hfov_rad: float,
        *,
        min_depth: float = 0.1,
        max_depth: float = 5.0,
    ) -> float:
        """Fuse a precomputed VLM score, enabling non-blocking inference loops."""
        if not np.isfinite(vlm_score) or not 0.0 <= float(vlm_score) <= 1.0:
            raise ValueError("VLM score_image() must return a finite value in [0, 1]")

        self.diag.vlm_query_count += 1
        self.diag.last_vlm_score = float(vlm_score)
        self._vlm_scores.append(float(vlm_score))

        if self._use_value_map and depth_image is not None:
            self._update_spatial_value_map(
                vlm_score,
                depth_image,
                camera_xy,
                camera_yaw,
                hfov_rad,
                min_depth,
                max_depth,
            )

        self.diag.value_map_updates = self._value_map_updates
        self.diag.value_map_observed_cells = self._value_map_observed_cells
        return float(vlm_score)

    # ------------------------------------------------------------------
    # Override: frontier detection with value-map scoring
    # ------------------------------------------------------------------
    def _do_detect(self) -> None:
        """Detect frontiers, score them via the value map, pick the best."""
        f_mask = detect_frontier_cells(self.local_map)
        clusters = cluster_frontiers(self.local_map, f_mask, self.min_cluster_size)

        self.diag.frontier_count = int(f_mask.sum())
        self.diag.cluster_count = len(clusters)

        # Termination check
        if self.local_map.explored_ratio() >= self.explore_threshold and not clusters:
            self._finish("exploration threshold reached")
            return

        candidates = self._reachable_candidates(clusters)

        if not candidates:
            # Try with smaller clusters
            c2 = cluster_frontiers(self.local_map, f_mask, min_cluster_size=1)
            candidates = self._reachable_candidates(c2)

        if not candidates:
            if self._stagnant_scans >= self.max_stagnant_scans:
                self._finish("no reachable frontiers, map static")
            else:
                self.state = FBEState.SCANNING
            return

        # --- Value-map guided frontier selection ---
        cluster = self._select_frontier(candidates)

        if cluster is None:
            self.state = FBEState.SCANNING
            return

        self._target_cluster = cluster
        self.diag.target_frontier = cluster
        self.state = FBEState.PLANNING

    # ------------------------------------------------------------------
    # Frontier selection (aligned with upstream ITMPolicy._get_best_frontier)
    # ------------------------------------------------------------------
    def _select_frontier(
        self,
        clusters: list[FrontierCluster],
    ) -> Optional[FrontierCluster]:
        """Select the best frontier using value-map scores + acyclic enforcement.

        Corresponds to ``ITMPolicy._get_best_frontier()`` in the upstream.
        """
        if len(clusters) == 1:
            return clusters[0]

        # Convert clusters to waypoints for value-map sorting
        waypoints = np.array([c.centroid_xy for c in clusters])

        # Sort by value-map scores
        if self._use_value_map and self._value_map_updates > 0:
            sorted_pts, sorted_values = self._value_map.sort_waypoints(
                waypoints,
                radius=0.5,
                reduce_fn=self._reduce_values if self._exploration_thresh > 0 else None,
                valid_mask=self.local_map.known_free(),
            )
        else:
            sorted_pts, sorted_values = self._sort_by_legacy_or_geometry(clusters, waypoints)

        robot_xy = self._robot_xy
        best_frontier_idx: Optional[int] = None
        top_two = (
            (sorted_values[0], sorted_values[1])
            if len(sorted_values) >= 2
            else (sorted_values[0], 0.0)
        )
        self.diag.mode = (
            "seek"
            if (self._exploration_thresh > 0 and sorted_values[0] >= self._exploration_thresh)
            else "explore"
        )

        # --- Sticky frontier: prefer the last pursued frontier if still good ---
        if self._sticky_last_frontier and self._last_frontier is not None:
            best_frontier_idx = self._match_last_frontier(sorted_pts, sorted_values)
            if best_frontier_idx is not None:
                self.diag.last_vlm_text = "sticking to last frontier"

        # --- Acyclic check ---
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                if self._acyclic.check_cyclic(robot_xy, frontier, top_two):
                    self.diag.suppressed_cyclic += 1
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            # All frontiers are cyclic — pick the closest
            best_frontier_idx = int(np.argmin(np.linalg.norm(sorted_pts - robot_xy, axis=1)))

        # Commit
        best_pt = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic.add_state_action(robot_xy, best_pt, top_two)
        self._last_value = best_value
        self._last_frontier = best_pt
        self.diag.selected_value = best_value

        # Map back to FrontierCluster
        distances = np.linalg.norm(waypoints - best_pt, axis=1)
        matched_idx = int(np.argmin(distances))
        return clusters[matched_idx]

    # ------------------------------------------------------------------
    # Value-map update
    # ------------------------------------------------------------------
    def _update_spatial_value_map(
        self,
        vlm_score: float,
        depth_img: np.ndarray,
        camera_xy: tuple[float, float],
        camera_yaw: float,
        hfov_rad: float,
        min_depth: float,
        max_depth: float,
    ) -> None:
        """Project a VLM score through the camera FOV onto the value map."""
        if self._exploration_thresh > 0:
            # Two-channel mode: [target_score, exploration_score]
            # The exploration score is 1 - target_score (inverted)
            target_score = float(vlm_score)
            explore_score = 1.0 - target_score
            values = np.array([target_score, explore_score], dtype=np.float32)
        else:
            values = np.array([float(vlm_score)], dtype=np.float32)

        try:
            self._value_map.update(
                values=values,
                depth_img=depth_img,
                camera_xy=camera_xy,
                camera_yaw=camera_yaw,
                hfov_rad=hfov_rad,
                min_depth=min_depth,
                max_depth=max_depth,
                blocked_mask=self.local_map.known_obstacles(),
                free_mask=self.local_map.known_free(),
                obstacle_mask=self.local_map.known_obstacles(),
            )
            self.diag.last_projection_error = ""
        except Exception as exc:
            self.diag.last_projection_error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"Value Map projection failed: {exc}") from exc

    @property
    def _value_map_updates(self) -> int:
        """Number of successfully fused spatial observations."""
        return self._value_map.update_count

    @property
    def _value_map_observed_cells(self) -> int:
        return self._value_map.observed_cell_count

    # ------------------------------------------------------------------
    # Reduction functions
    # ------------------------------------------------------------------
    def _reduce_values(
        self,
        values: list[tuple[float, float]],
    ) -> list[float]:
        """Two-channel reduction (ITMPolicyV3 pattern).

        If the best *target* value across all candidates is below
        ``exploration_thresh``, return *exploration* scores; otherwise
        return *target* scores.
        """
        target_vals = [v[0] for v in values]
        explore_vals = [v[1] for v in values]
        if max(target_vals) < self._exploration_thresh:
            return explore_vals
        return target_vals

    def _sort_by_legacy_or_geometry(
        self,
        clusters: list[FrontierCluster],
        waypoints: np.ndarray,
    ) -> tuple[np.ndarray, list[float]]:
        """Fallback: use legacy value map or geometric scoring."""
        # Try legacy value map first
        if self._vlm_scores:
            last_score = self._vlm_scores[-1]
            for cl in clusters:
                self._legacy_value_map.update(cl, last_score)

        best_cl = self._legacy_value_map.best_cluster(clusters)
        if best_cl is not None:
            # Sort by legacy value
            scored = [
                (self._legacy_value_map.cluster_value(c) or 0.0, i) for i, c in enumerate(clusters)
            ]
            scored.sort(key=lambda x: -x[0])
            order = [i for _, i in scored]
            return waypoints[order], [v for v, _ in scored]

        # Pure geometry
        robot_xy = self._robot_xy
        scores = [
            c.size / (1.0 + float(np.linalg.norm(c.centroid_xy - robot_xy))) for c in clusters
        ]
        order = np.argsort([-s for s in scores])
        return waypoints[order], [scores[i] for i in order]

    # ------------------------------------------------------------------
    # Sticky frontier matching
    # ------------------------------------------------------------------
    def _match_last_frontier(
        self,
        sorted_pts: np.ndarray,
        sorted_values: list[float],
    ) -> Optional[int]:
        """If the last-pursued frontier is still viable, return its index."""
        if self._last_frontier is None:
            return None

        distances = np.linalg.norm(sorted_pts - self._last_frontier, axis=1)
        match_idx = int(np.argmin(distances))
        if float(distances[match_idx]) < 0.5:
            # It's still in the candidate set
            curr_value = sorted_values[match_idx]
            if curr_value + 0.01 >= self._last_value:
                return match_idx
        return None

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def value_map_image(self) -> np.ndarray:
        """Return an RGB visualisation of the spatial value map."""
        return self._value_map.visualize()
