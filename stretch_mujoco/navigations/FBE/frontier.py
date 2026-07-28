"""Frontier detection and clustering for FBE.

A *frontier cell* is a free cell (0) that has at least one unknown (-1)
neighbour.  Adjacent frontier cells are grouped into clusters, each of
which becomes a candidate exploration target.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from stretch_mujoco.navigations.FBE.local_map import LocalOccupancyGrid

# 8-connected neighbourhood offsets
_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass
class FrontierCluster:
    """A contiguous group of frontier cells."""

    cells: list[tuple[int, int]]  # cell coordinates
    centroid_xy: np.ndarray  # world (x, y) of the cluster centre
    size: int  # number of cells

    @property
    def centroid_cell(self) -> tuple[int, int]:
        """Frontier cell closest to the (possibly off-frontier) centroid."""
        centre = np.mean(np.asarray(self.cells, dtype=float), axis=0)
        return min(
            self.cells,
            key=lambda cell: float(np.sum((np.asarray(cell, dtype=float) - centre) ** 2)),
        )


# ---------------------------------------------------------------------------
def detect_frontier_cells(local_map: LocalOccupancyGrid) -> np.ndarray:
    """Return a boolean mask where True marks frontier cells.

    A cell is a frontier when it is free (0) AND at least one of its
    8-neighbours is unknown (-1).
    """
    free = local_map.data == 0
    unknown = local_map.data == -1

    rows, cols = local_map.data.shape
    frontier = np.zeros((rows, cols), dtype=bool)

    # Check each free cell's neighbours for unknown
    for dr, dc in _NEIGHBOURS:
        # Shift the unknown mask and AND with free
        r_src_start = max(0, -dr)
        r_src_end = rows + min(0, -dr)
        c_src_start = max(0, -dc)
        c_src_end = cols + min(0, -dc)

        r_dst_start = max(0, dr)
        c_dst_start = max(0, dc)

        neigh_unknown = unknown[r_src_start:r_src_end, c_src_start:c_src_end]
        local_free = free[
            r_dst_start : r_dst_start + neigh_unknown.shape[0],
            c_dst_start : c_dst_start + neigh_unknown.shape[1],
        ]

        frontier[
            r_dst_start : r_dst_start + neigh_unknown.shape[0],
            c_dst_start : c_dst_start + neigh_unknown.shape[1],
        ] |= (
            local_free & neigh_unknown
        )

    return frontier


def cluster_frontiers(
    local_map: LocalOccupancyGrid,
    frontier_mask: Optional[np.ndarray] = None,
    min_cluster_size: int = 3,
) -> list[FrontierCluster]:
    """Group adjacent frontier cells into clusters via BFS.

    Parameters
    ----------
    local_map:
        The local occupancy grid.
    frontier_mask:
        Pre-computed frontier mask (computed if None).
    min_cluster_size:
        Clusters with fewer cells are discarded.

    Returns
    -------
    list[FrontierCluster]
        Clusters sorted by size descending.
    """
    if frontier_mask is None:
        frontier_mask = detect_frontier_cells(local_map)

    rows, cols = frontier_mask.shape
    visited = np.zeros((rows, cols), dtype=bool)
    clusters: list[FrontierCluster] = []

    for r0 in range(rows):
        for c0 in range(cols):
            if not frontier_mask[r0, c0] or visited[r0, c0]:
                continue

            # BFS to collect this cluster
            cells: list[tuple[int, int]] = []
            queue = deque([(r0, c0)])
            visited[r0, c0] = True

            while queue:
                r, c = queue.popleft()
                cells.append((r, c))

                for dr, dc in _NEIGHBOURS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if frontier_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            if len(cells) >= min_cluster_size:
                # Compute world centroid
                world_pts = np.array([local_map.cell_to_world(c) for c in cells])
                centroid = world_pts.mean(axis=0)
                clusters.append(
                    FrontierCluster(
                        cells=cells,
                        centroid_xy=centroid,
                        size=len(cells),
                    )
                )

    # Sort largest → smallest
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


def best_frontier(
    clusters: list[FrontierCluster],
    robot_xy: np.ndarray,
    *,
    prefer_large: bool = True,
    min_dist: float = 0.5,
) -> Optional[FrontierCluster]:
    """Pick the best frontier to explore.

    Balances cluster size against distance from the robot.
    Clusters closer than *min_dist* are filtered out first unless
    they are the only ones available.
    """
    if not clusters:
        return None

    # Prefer clusters that are at least min_dist away
    far = [c for c in clusters if float(np.linalg.norm(c.centroid_xy - robot_xy)) >= min_dist]
    candidates = far if far else clusters

    # Approximate information gain per travel cost. A larger frontier is
    # useful, while an otherwise equivalent closer frontier is preferred.
    def _score(cl: FrontierCluster) -> float:
        dist = float(np.linalg.norm(cl.centroid_xy - robot_xy))
        if not prefer_large:
            return -dist
        return cl.size / (1.0 + dist)

    return max(candidates, key=_score)
