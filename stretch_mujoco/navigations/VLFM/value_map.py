"""Spatial value map that projects VLM scores through FOV onto a 2-D grid.

Aligned with ``vlfm.mapping.value_map.ValueMap`` from the upstream
`rai-opensource/vlfm <https://github.com/rai-opensource/vlfm>`_ repository
(ICRA 2024).

The core idea: at each exploration step the VLM returns a **scalar** score
(cosine similarity between the current camera image and the language
instruction).  That score is projected through the camera's field-of-view
cone onto the grid, weighted by a confidence mask that emphasises central
pixels.  The map is fused incrementally across steps so that regions the
robot has looked at from multiple angles accumulate stable value estimates.
"""

from __future__ import annotations

import math
from typing import Optional, Callable

import cv2
import numpy as np


class ValueMap:
    """Fuse per-image VLM scores into a persistent spatial value map.

    Each :meth:`update` call takes a scalar (or multi-channel) value, a depth
    image, and the camera pose, then projects the value through the visible
    portion of the FOV cone.  Successive updates are fused using either
    *max-confidence* (default) or *confidence-weighted averaging*.

    Parameters
    ----------
    grid_shape:
        ``(rows, cols)`` of the local occupancy map.
    resolution:
        Metres per grid cell.
    origin_xy:
        World ``(x_min, y_min)`` represented by cell ``(0, 0)``. When
        omitted, the map is centred on world ``(0, 0)`` for compatibility.
    value_channels:
        Number of independent value channels (default 1).  Multi-channel maps
        can separate, e.g., *target-seeking* vs *exploration* scores.
    use_max_confidence:
        When True (default), a pixel's value is replaced whenever a new
        observation arrives with higher confidence.  When False, values are
        blended by confidence-weighted averaging.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        resolution: float,
        *,
        origin_xy: Optional[tuple[float, float]] = None,
        value_channels: int = 1,
        use_max_confidence: bool = True,
    ) -> None:
        if len(grid_shape) != 2 or any(
            not isinstance(size, (int, np.integer)) or int(size) <= 0 for size in grid_shape
        ):
            raise ValueError("grid_shape must contain two positive integers")
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and > 0")
        if value_channels < 1:
            raise ValueError("value_channels must be >= 1")
        self._rows, self._cols = int(grid_shape[0]), int(grid_shape[1])
        self._resolution = float(resolution)
        if origin_xy is None:
            self._x_min = -(self._cols // 2) * self._resolution
            self._y_min = -(self._rows // 2) * self._resolution
        else:
            origin = np.asarray(origin_xy, dtype=float)
            if origin.shape != (2,) or not np.all(np.isfinite(origin)):
                raise ValueError("origin_xy must contain two finite coordinates")
            self._x_min, self._y_min = float(origin[0]), float(origin[1])
        self._value_channels = int(value_channels)
        self._use_max_confidence = bool(use_max_confidence)

        # _values:   (rows, cols, value_channels) — accumulated semantic scores
        # _conf_map: (rows, cols) — confidence (0..1) of the best or blended value
        self._values = np.zeros((self._rows, self._cols, value_channels), dtype=np.float32)
        self._conf_map = np.zeros((self._rows, self._cols), dtype=np.float32)
        self._update_count = 0

        # Cached confidence masks keyed by (fov_deg, max_depth_cells)
        self._confidence_mask_cache: dict[tuple[float, int], np.ndarray] = {}

        # Weight for central-vs-peripheral confidence fall-off
        self._min_confidence: float = 0.25

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        values: np.ndarray,
        depth_img: np.ndarray,
        camera_xy: tuple[float, float],
        camera_yaw: float,
        hfov_rad: float,
        min_depth: float,
        max_depth: float,
        blocked_mask: Optional[np.ndarray] = None,
        free_mask: Optional[np.ndarray] = None,
        obstacle_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Project *values* through the FOV cone and fuse into the map.

        Parameters
        ----------
        values:
            Scalar or array of shape ``(value_channels,)`` — the VLM score(s)
            assigned to every pixel visible in this observation.
        depth_img:
            2-D depth array ``(H, W)`` in **metres**.  May be a real depth
            image from MuJoCo or a down-projected occupancy depth.
        camera_xy:
            World ``(x, y)`` of the camera centre.
        camera_yaw:
            Yaw angle (radians) of the camera optical axis (0 = +X).
        hfov_rad:
            Horizontal field-of-view in radians.
        min_depth:
            Minimum valid depth (metres).  Closer points are ignored.
        max_depth:
            Maximum valid depth (metres).  Farther points are clipped.
        """
        if values.shape != (self._value_channels,):
            raise ValueError(
                f"Expected values of shape ({self._value_channels},), " f"got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite")
        camera = np.asarray(camera_xy, dtype=float)
        if camera.shape != (2,) or not np.all(np.isfinite(camera)):
            raise ValueError("camera_xy must contain two finite coordinates")
        if not np.isfinite(camera_yaw):
            raise ValueError("camera_yaw must be finite")
        if not np.isfinite(hfov_rad) or not 0 < hfov_rad < math.pi:
            raise ValueError("hfov_rad must be in (0, pi)")
        if not np.isfinite(min_depth) or min_depth < 0:
            raise ValueError("min_depth must be finite and >= 0")
        if not np.isfinite(max_depth) or max_depth <= min_depth:
            raise ValueError("max_depth must be finite and greater than min_depth")

        # 1. Compute the visible portion of the FOV cone
        local_mask = self._process_local_data(depth_img, hfov_rad, min_depth, max_depth)

        # 2. Rotate to world orientation and place on the full map
        curr_map = self._localize_new_data(local_mask, camera_xy, camera_yaw)
        projection_footprint = curr_map > 0

        if blocked_mask is not None:
            blocked = np.asarray(blocked_mask, dtype=bool)
            if blocked.shape != (self._rows, self._cols):
                raise ValueError("blocked_mask must match the value-map shape")
            curr_map[blocked] = 0.0

        confirmed_free = None
        if free_mask is not None:
            confirmed_free = np.asarray(free_mask, dtype=bool)
            if confirmed_free.shape != (self._rows, self._cols):
                raise ValueError("free_mask must match the value-map shape")
            # Values are meaningful only on cells confirmed free by the map.
            # Unknown and occupied cells must never receive semantic scores.
            curr_map[~confirmed_free] = 0.0

        obstacles = None
        if obstacle_mask is not None:
            obstacles = np.asarray(obstacle_mask, dtype=bool)
            if obstacles.shape != (self._rows, self._cols):
                raise ValueError("obstacle_mask must match the value-map shape")
            curr_map[obstacles] = 0.0

        if confirmed_free is not None:
            # Stale semantic values must not survive if a cell is no longer
            # confirmed free by the occupancy map.
            self._values[~confirmed_free] = 0.0
            self._conf_map[~confirmed_free] = 0.0
            camera_cell = self._world_to_cell_unclipped(np.asarray(camera_xy, dtype=float))
            if camera_cell is None or not confirmed_free[camera_cell]:
                curr_map.fill(0.0)
            else:
                curr_map = _apply_occupancy_visibility(
                    curr_map,
                    camera_cell,
                    confirmed_free,
                    obstacles,
                )

            # Clear old values in this FOV when the current occupancy ray
            # proves that they are unknown, occupied, or behind an obstacle.
            occluded = projection_footprint & (curr_map <= 0)
            self._values[occluded] = 0.0
            self._conf_map[occluded] = 0.0

        # 3. Fuse with existing data
        self._fuse_new_data(curr_map, values)
        self._update_count += 1

    def sort_waypoints(
        self,
        waypoints: np.ndarray,
        radius: float = 0.5,
        *,
        reduce_fn: Optional[Callable[[list[tuple[float, ...]]], list[float]]] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, list[float]]:
        """Sort 2-D waypoints by their spatial value scores.

        Parameters
        ----------
        waypoints:
            ``(N, 2)`` world coordinates.
        radius:
            Search radius in metres for aggregating nearby values.
        reduce_fn:
            When *value_channels* > 1, this callable maps a list of
            ``(ch0, ch1, ...)`` tuples to a flat list of scalars.  Defaults
            to taking the max across channels.

        Returns
        -------
        sorted_waypoints:
            ``(N, 2)`` waypoints sorted best → worst.
        sorted_values:
            Scalar score for each waypoint (same order).
        """
        points = np.asarray(waypoints, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            raise ValueError("waypoints must have shape (N, 2) with N > 0")
        if not np.all(np.isfinite(points)):
            raise ValueError("waypoints must be finite")
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and >= 0")
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != (self._rows, self._cols):
                raise ValueError("valid_mask must match the value-map shape")

        radius_cells = int(radius / self._resolution)
        radius_cells = max(1, radius_cells)

        def _value_at(point: np.ndarray) -> float | tuple[float, ...]:
            row, col = self._world_to_cell(point)
            vals = [
                _pixel_value_within_radius(
                    self._values[..., ch],
                    (row, col),
                    radius_cells,
                    valid_mask=valid_mask,
                )
                for ch in range(self._value_channels)
            ]
            return tuple(vals) if len(vals) > 1 else vals[0]

        raw_values = [_value_at(wp) for wp in waypoints]

        if self._value_channels > 1:
            if reduce_fn is None:
                # Default: max across channels
                reduced = [max(v) if isinstance(v, tuple) else v for v in raw_values]  # type: ignore[arg-type]
            else:
                reduced = reduce_fn(raw_values)  # type: ignore[arg-type]
        else:
            reduced = [float(v) for v in raw_values]

        order = np.argsort([-v for v in reduced])
        sorted_waypoints = np.array([waypoints[i] for i in order])
        sorted_values = [reduced[i] for i in order]

        return sorted_waypoints, sorted_values

    def get_value_at(self, point_xy: np.ndarray, radius: float = 0.0) -> float:
        """Return the scalar value (first channel) at a world point."""
        rc = max(1, int(radius / self._resolution)) if radius > 0 else 1
        row, col = self._world_to_cell(point_xy)
        return float(_pixel_value_within_radius(self._values[..., 0], (row, col), rc))

    def best_point(
        self,
        candidates: np.ndarray,
        radius: float = 0.5,
    ) -> tuple[np.ndarray, float]:
        """Return the candidate with the highest value and its score."""
        sorted_pts, sorted_vals = self.sort_waypoints(candidates, radius=radius)
        return sorted_pts[0], sorted_vals[0]

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._values.fill(0.0)
        self._conf_map.fill(0.0)
        self._update_count = 0

    @property
    def update_count(self) -> int:
        """Number of successfully fused RGB-D observations."""
        return self._update_count

    @property
    def observed_cell_count(self) -> int:
        """Number of cells with non-zero projection confidence."""
        return int(np.count_nonzero(self._conf_map))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def visualize(self, *, reduce_fn: Optional[Callable] = None) -> np.ndarray:
        """Return an RGB visualisation of the value map (inferno colourmap).

        Returns
        -------
        np.ndarray
            ``(rows, cols, 3)`` uint8 image.  Zero-value cells are white.
        """
        if reduce_fn is not None:
            reduced = reduce_fn(self._values)
        elif self._value_channels == 1:
            reduced = self._values[:, :, 0].copy()
        else:
            reduced = np.max(self._values, axis=2)

        # Flip vertically so +Y points up in the image
        map_img = np.flipud(reduced)

        zero_mask = map_img == 0
        if not zero_mask.all():
            # Normalise to [0, 1] for colour mapping
            vmax = np.max(map_img)
            if vmax > 0:
                map_img = map_img / vmax
        # Apply inferno colourmap
        rgb = _inferno_rgb(map_img)
        rgb[zero_mask] = (255, 255, 255)
        return rgb

    # ------------------------------------------------------------------
    # Internal: FOV projection
    # ------------------------------------------------------------------
    def _process_local_data(
        self,
        depth_img: np.ndarray,
        hfov_rad: float,
        min_depth: float,
        max_depth: float,
    ) -> np.ndarray:
        """Build a confidence-weighted mask of the visible FOV cone.

        Returns a ``(2*R+1, 2*R+1)`` float array in camera-local coordinates
        (camera at centre, +X forward).
        """
        depth = np.asarray(depth_img, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2 or depth.shape[0] == 0 or depth.shape[1] < 2:
            raise ValueError("depth_img must be a non-empty 2-D array")

        # MuJoCo depth is distance along the camera optical axis. Following
        # upstream VLFM, the farthest valid pixel in each image column defines
        # the visible ground-plane boundary for that bearing.
        valid = np.isfinite(depth) & (depth > 0)
        depth_row = np.max(np.where(valid, depth, -np.inf), axis=0)
        valid_columns = np.isfinite(depth_row)
        if not np.any(valid_columns):
            empty_radius = max(1, int(math.ceil(max_depth / self._resolution)))
            return np.zeros((2 * empty_radius + 1,) * 2, dtype=np.float32)

        # Interpolate isolated invalid columns instead of turning them into a
        # false near obstacle or allowing NaNs into OpenCV coordinates.
        column_ids = np.arange(depth_row.size)
        depth_row = np.interp(
            column_ids,
            column_ids[valid_columns],
            depth_row[valid_columns],
        )
        depth_row_m = np.clip(depth_row, min_depth, max_depth)

        max_cells = max(1, int(math.ceil(max_depth / self._resolution)))
        cone_size = max_cells * 2 + 1

        # Get blank confidence mask
        cone_mask = self._get_confidence_mask(hfov_rad, max_cells)

        # Compute the depth boundary in camera-local grid coordinates. Depth
        # is axial, so lateral displacement is depth * tan(bearing).
        angles = np.linspace(-hfov_rad / 2, hfov_rad / 2, len(depth_row_m))

        x = depth_row_m
        y = depth_row_m * np.tan(angles)

        px_x = (x / self._resolution + cone_size // 2).astype(int)
        px_y = (y / self._resolution + cone_size // 2).astype(int)

        # Fill only the polygon between the camera and the measured depth
        # boundary. Intersecting it with the confidence cone guarantees that
        # cells beyond an obstacle never inherit the observation score.
        centre = np.array([[cone_size // 2, cone_size // 2]], dtype=np.int32)
        boundary = np.stack((px_x, px_y), axis=1).astype(np.int32)
        visible_polygon = np.concatenate((centre, boundary), axis=0)
        depth_mask = np.zeros_like(cone_mask, dtype=np.uint8)
        cv2.fillPoly(depth_mask, [visible_polygon], 1)

        return (cone_mask * depth_mask).astype(np.float32)

    def _get_confidence_mask(self, hfov_rad: float, max_cells: int) -> np.ndarray:
        """Get (or create and cache) a cone mask with centre-weighted confidence."""
        key = (round(math.degrees(hfov_rad), 2), max_cells)
        if key in self._confidence_mask_cache:
            return self._confidence_mask_cache[key].copy()

        size = max_cells * 2 + 1
        cone_mask = np.zeros((size, size), dtype=np.uint8)
        # Draw the raw FOV cone pointing RIGHT (0° = +X in world coords).
        # OpenCV angles: 0 = right, 90 = up, clockwise.
        cv2.ellipse(
            cone_mask,
            (size // 2, size // 2),
            (size // 2, size // 2),
            0,
            -math.degrees(hfov_rad) / 2,
            math.degrees(hfov_rad) / 2,
            1,
            -1,
        )

        # Apply confidence weighting: cos² of angular distance from centre axis
        adjusted = np.zeros((size, size), dtype=np.float32)
        for row in range(size):
            for col in range(size):
                if not cone_mask[row, col]:
                    continue
                # radial distance from the forward axis (rightwards, +col)
                dy = abs(row - size // 2)
                dx = col - size // 2
                if dx <= 0:
                    continue  # behind the camera
                angle = math.atan2(float(dy), float(dx))
                angle = _remap(angle, 0.0, hfov_rad / 2, 0.0, math.pi / 2)
                confidence = math.cos(angle) ** 2
                confidence = _remap(confidence, 0.0, 1.0, self._min_confidence, 1.0)
                adjusted[row, col] = confidence

        self._confidence_mask_cache[key] = adjusted
        return adjusted.copy()

    # ------------------------------------------------------------------
    # Internal: localisation & fusion
    # ------------------------------------------------------------------
    def _localize_new_data(
        self,
        local_mask: np.ndarray,
        camera_xy: tuple[float, float],
        camera_yaw: float,
    ) -> np.ndarray:
        """Rotate *local_mask* into the world frame and place it on the full grid."""
        # Rotate by -yaw (the mask is in camera frame where +X = forward)
        rotated = _rotate_image(local_mask, -camera_yaw)

        # Camera position in pixel coordinates
        cx, cy = camera_xy
        px = int(round((cx - self._x_min) / self._resolution))
        py = int(round((cy - self._y_min) / self._resolution))

        # Place rotated mask onto a full-size grid
        full = np.zeros((self._rows, self._cols), dtype=np.float32)
        full = _place_image(full, rotated, px, py)
        return full

    def _fuse_new_data(self, new_map: np.ndarray, values: np.ndarray) -> None:
        """Fuse *new_map* confidence and *values* into the persistent maps."""
        if self._use_max_confidence:
            # Replace pixels where new confidence exceeds existing
            higher = new_map > self._conf_map
            self._values[higher] = values
            self._conf_map[higher] = new_map[higher]
        else:
            # Confidence-weighted moving average
            denom = self._conf_map + new_map
            with np.errstate(divide="ignore", invalid="ignore"):
                w1 = np.where(denom > 0, self._conf_map / denom, 0.0)
                w2 = np.where(denom > 0, new_map / denom, 0.0)

            w1_c = np.repeat(w1[:, :, np.newaxis], self._value_channels, axis=2)
            w2_c = np.repeat(w2[:, :, np.newaxis], self._value_channels, axis=2)

            self._values = self._values * w1_c + values * w2_c
            self._conf_map = self._conf_map * w1 + new_map * w2

            self._values = np.nan_to_num(self._values)
            self._conf_map = np.nan_to_num(self._conf_map)

    # ------------------------------------------------------------------
    # Internal: coordinate helpers
    # ------------------------------------------------------------------
    def _world_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        """World (x, y) → (row, col) in the value map."""
        cell = self._world_to_cell_unclipped(point)
        if cell is not None:
            return cell
        x, y = point[0], point[1]
        col = int(round((x - self._x_min) / self._resolution))
        row = int(round((y - self._y_min) / self._resolution))
        return (max(0, min(self._rows - 1, row)), max(0, min(self._cols - 1, col)))

    def _world_to_cell_unclipped(self, point: np.ndarray) -> Optional[tuple[int, int]]:
        """World point to a cell, returning ``None`` outside map bounds."""
        x, y = float(point[0]), float(point[1])
        col = int(round((x - self._x_min) / self._resolution))
        row = int(round((y - self._y_min) / self._resolution))
        if 0 <= row < self._rows and 0 <= col < self._cols:
            return row, col
        return None


# ---------------------------------------------------------------------------
# LanguageValueMap — backward-compatible frontier-cluster value store
# ---------------------------------------------------------------------------


class LanguageValueMap:
    """Fuse VLM scores into frontier cells across exploration decisions.

    This is a lightweight grid-native value map used as a fallback when the
    spatial :class:`ValueMap` is disabled.  Each observation updates the
    selected frontier cells and a configurable neighbourhood using weighted
    averaging.
    """

    def __init__(self, local_map, *, diffusion_radius: int = 2):
        if not isinstance(diffusion_radius, int) or diffusion_radius < 0:
            raise ValueError("diffusion_radius must be a non-negative integer")
        self.values = np.zeros(local_map.shape, dtype=float)
        self.weights = np.zeros(local_map.shape, dtype=float)
        self.diffusion_radius = diffusion_radius

    def update(self, cluster, score: float) -> None:
        """Fuse one semantic score around a frontier cluster."""
        if not np.isfinite(score):
            raise ValueError("value-map score must be finite")
        rows, cols = self.values.shape
        for row, col in cluster.cells:
            for dr in range(-self.diffusion_radius, self.diffusion_radius + 1):
                for dc in range(-self.diffusion_radius, self.diffusion_radius + 1):
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    distance = float(np.hypot(dr, dc))
                    if distance > self.diffusion_radius:
                        continue
                    weight = 1.0 / (1.0 + distance)
                    total = self.weights[nr, nc] + weight
                    self.values[nr, nc] = (
                        self.values[nr, nc] * self.weights[nr, nc] + score * weight
                    ) / total
                    self.weights[nr, nc] = total

    def cluster_value(self, cluster) -> float | None:
        """Return the observed mean value for a cluster, or None."""
        observed = [self.values[cell] for cell in cluster.cells if self.weights[cell] > 0]
        if not observed:
            return None
        return float(np.mean(observed))

    def best_cluster(self, clusters: list) -> object | None:
        """Return the highest-valued observed cluster."""
        valued = [
            (value, cluster)
            for cluster in clusters
            if (value := self.cluster_value(cluster)) is not None
        ]
        if not valued:
            return None
        return max(valued, key=lambda item: item[0])[1]

    @property
    def update_count(self) -> int:
        return int(np.count_nonzero(self.weights))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pixel_value_within_radius(
    channel: np.ndarray,
    centre: tuple[int, int],
    radius_px: int,
    *,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Max value of *channel* within a circular window."""
    row, col = centre
    r_min = max(0, row - radius_px)
    r_max = min(channel.shape[0], row + radius_px + 1)
    c_min = max(0, col - radius_px)
    c_max = min(channel.shape[1], col + radius_px + 1)
    patch = channel[r_min:r_max, c_min:c_max]
    if patch.size == 0:
        return 0.0
    patch_rows, patch_cols = np.ogrid[r_min:r_max, c_min:c_max]
    selection = (patch_rows - row) ** 2 + (patch_cols - col) ** 2 <= radius_px**2
    if valid_mask is not None:
        selection &= valid_mask[r_min:r_max, c_min:c_max]
    if not np.any(selection):
        return 0.0
    return float(np.max(patch[selection]))


def _apply_occupancy_visibility(
    projected: np.ndarray,
    camera_cell: tuple[int, int],
    confirmed_free: np.ndarray,
    obstacles: Optional[np.ndarray],
) -> np.ndarray:
    """Remove projected cells whose grid ray crosses unknown or occupied space."""
    visible = projected.copy()
    candidates = np.argwhere(visible > 0)
    for target_array in candidates:
        target = int(target_array[0]), int(target_array[1])
        ray = _bresenham_cells(camera_cell, target)
        # Include the target: every written cell and every cell leading to it
        # must be confirmed free. Obstacles are checked explicitly for clarity
        # even though they should already be absent from confirmed_free.
        blocked = any(not confirmed_free[cell] for cell in ray[1:])
        if not blocked and obstacles is not None:
            blocked = any(obstacles[cell] for cell in ray[1:])
        if blocked:
            visible[target] = 0.0
    return visible


def _bresenham_cells(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return inclusive grid cells along a line using Bresenham traversal."""
    row, col = start
    end_row, end_col = end
    delta_col = abs(end_col - col)
    delta_row = -abs(end_row - row)
    step_col = 1 if col < end_col else -1
    step_row = 1 if row < end_row else -1
    error = delta_col + delta_row
    cells = []

    while True:
        cells.append((row, col))
        if row == end_row and col == end_col:
            return cells
        doubled_error = 2 * error
        if doubled_error >= delta_row:
            error += delta_row
            col += step_col
        if doubled_error <= delta_col:
            error += delta_col
            row += step_row


def _remap(value: float, lo_in: float, hi_in: float, lo_out: float, hi_out: float) -> float:
    """Linear map from one interval to another."""
    return (value - lo_in) * (hi_out - lo_out) / (hi_in - lo_in) + lo_out


def _rotate_image(img: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a 2-D array about its centre by *angle_rad* (counter-clockwise)."""
    h, w = img.shape
    centre = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(centre, math.degrees(angle_rad), 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)


def _place_image(
    full: np.ndarray,
    patch: np.ndarray,
    cx: int,
    cy: int,
) -> np.ndarray:
    """Overlay *patch* onto *full* so that patch centre lands at ``(cx, cy)``.

    Pixel coordinates use ``(x=col, y=row)`` convention, i.e.
    ``full[py, px]``.  The origin of *full* is at ``(0, 0)`` (top-left).
    """
    ph, pw = patch.shape
    fh, fw = full.shape

    # Top-left of where the patch should be placed
    px0 = cx - pw // 2
    py0 = cy - ph // 2

    # Intersection with the full image
    src_x0 = max(0, -px0)
    src_y0 = max(0, -py0)
    src_x1 = min(pw, fw - px0)
    src_y1 = min(ph, fh - py0)

    dst_x0 = max(0, px0)
    dst_y0 = max(0, py0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        full[dst_y0:dst_y1, dst_x0:dst_x1] = patch[src_y0:src_y1, src_x0:src_x1]

    return full


def _inferno_rgb(values: np.ndarray) -> np.ndarray:
    """Map a 2-D float array in ``[0, 1]`` to OpenCV's Inferno RGB map."""
    normalized = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    grayscale = np.rint(normalized * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(grayscale, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
