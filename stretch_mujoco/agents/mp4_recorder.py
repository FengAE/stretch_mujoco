"""Lightweight MP4 recorder for multi-agent office simulations.

Renders a 2-D top-down view of the office floor plan with agent positions,
status overlays, and a running clock. Does not require a MuJoCo model —
all layout data comes from the semantic-world manifest or explicit positions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Default colour palette for agents (BGR)
AGENT_COLORS = [
    (239, 128, 64),   # warm orange
    (200, 80, 80),    # steel blue
    (80, 180, 80),    # green
    (220, 180, 60),   # purple-ish
]

STATUS_COLORS = {
    "work": (200, 180, 100),
    "use_computer": (180, 200, 100),
    "rest": (120, 200, 200),
    "eat": (100, 180, 220),
    "drink": (100, 150, 220),
    "meeting": (220, 120, 200),
    "move_to": (180, 180, 180),
    "idle": (140, 140, 140),
    "request_robot": (100, 100, 220),
}

# Office layout colours (BGR)
FLOOR_COLOR = (245, 240, 235)
WALL_COLOR = (160, 160, 155)
ZONE_COLORS = {
    "work": (230, 235, 240),
    "meeting": (225, 220, 235),
    "lounge": (220, 235, 225),
    "snack": (235, 225, 210),
}


class OfficeMp4Recorder:
    """Record a 2-D animation of agent activity over a working day."""

    def __init__(
        self,
        output_path: str | Path,
        scene_manifest: str | Path | None = None,
        *,
        width_m: float = 18.0,
        depth_m: float = 12.0,
        fps: int = 10,
        pixels_per_meter: int = 80,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter

        # Canvas dimensions
        margin = 60  # px
        self.canvas_w = int(width_m * pixels_per_meter) + 2 * margin
        self.canvas_h = int(depth_m * pixels_per_meter) + 2 * margin
        self.margin = margin
        self._origin_x = margin
        self._origin_y = margin
        self._width_m = width_m
        self._depth_m = depth_m

        # MuJoCo coords -> image coords:  x_img = origin_x + (x_mj + width/2) * ppm
        #                            y_img = origin_y + (depth/2 - y_mj) * ppm
        self._x_offset = width_m / 2
        self._y_offset = depth_m / 2

        # Zone definitions (from manifest or defaults)
        self._zones: list[dict[str, Any]] = []
        if scene_manifest is not None:
            self._load_zones(Path(scene_manifest))

        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0
        self._base_canvas: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.canvas_w, self.canvas_h),
        )
        self._base_canvas = self._draw_floor_plan()

    def record_frame(
        self,
        minute_of_day: float,
        agents: dict[str, dict[str, Any]],
        *,
        events: list[str] | None = None,
    ) -> None:
        """Render one frame and write it to the video."""
        if self._writer is None or self._base_canvas is None:
            raise RuntimeError("Call start() before record_frame()")

        canvas = self._base_canvas.copy()

        # --- draw agents ---
        for idx, (agent_id, state) in enumerate(agents.items()):
            color = AGENT_COLORS[idx % len(AGENT_COLORS)]
            pos_x, pos_y = state.get("position", (0.0, 0.0))
            px, py = self._world_to_pixel(pos_x, pos_y)
            action = state.get("action", "idle")
            label = state.get("label", agent_id)

            # Agent dot + halo
            cv2.circle(canvas, (px, py), 12, (255, 255, 255), -1)
            cv2.circle(canvas, (px, py), 9, color, -1)

            # Status ring
            status_color = STATUS_COLORS.get(action, (140, 140, 140))
            cv2.circle(canvas, (px, py), 15, status_color, 2)

            # Label
            cv2.putText(
                canvas, label, (px + 18, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA,
            )
            # Action
            cv2.putText(
                canvas, action, (px + 18, py + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1, cv2.LINE_AA,
            )

        # --- clock ---
        hour = int(minute_of_day) // 60
        minute = int(minute_of_day) % 60
        clock_text = f"{hour:02d}:{minute:02d}"
        cv2.putText(
            canvas, clock_text, (self.canvas_w - 120, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "Office Sim", (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, cv2.LINE_AA,
        )

        # --- event feed ---
        if events:
            y_line = self.canvas_h - 15
            for evt in reversed(events[-4:]):
                cv2.putText(
                    canvas, evt, (15, y_line),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1, cv2.LINE_AA,
                )
                y_line -= 16

        self._writer.write(canvas)
        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_pixel(self, x_m: float, y_m: float) -> tuple[int, int]:
        px = int(self._origin_x + (x_m + self._x_offset) * self.pixels_per_meter)
        py = int(self._origin_y + (self._y_offset - y_m) * self.pixels_per_meter)
        return px, py

    def _draw_floor_plan(self) -> np.ndarray:
        canvas = np.full(
            (self.canvas_h, self.canvas_w, 3),
            FLOOR_COLOR[::-1],
            dtype=np.uint8,
        )

        # Walls
        x0, y0 = self._origin_x, self._origin_y
        w = int(self._width_m * self.pixels_per_meter)
        h = int(self._depth_m * self.pixels_per_meter)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), WALL_COLOR[::-1], 3)

        # Front-wall gap (door): gap around y_mj = -depth/2, x_mj = 0
        gap_center_x = int(x0 + self._x_offset * self.pixels_per_meter)
        gap_bottom = int(y0 + h)
        gap_width_px = int(1.8 * self.pixels_per_meter)
        cv2.line(canvas, (gap_center_x - gap_width_px // 2, gap_bottom),
                 (gap_center_x + gap_width_px // 2, gap_bottom),
                 FLOOR_COLOR[::-1], 3)

        # Zones
        for zone in self._zones:
            xmin, xmax, ymin, ymax = zone["bounds"]
            px1, py1 = self._world_to_pixel(xmin, ymax)
            px2, py2 = self._world_to_pixel(xmax, ymin)
            zone_color = ZONE_COLORS.get(zone.get("type", ""), (230, 230, 225))
            cv2.rectangle(canvas, (px1, py1), (px2, py2),
                          zone_color[::-1], -1)
            # Zone label
            cx, cy = self._world_to_pixel((xmin + xmax) / 2, (ymin + ymax) / 2)
            cv2.putText(canvas, zone.get("type", "").upper(), (cx - 20, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 175), 1, cv2.LINE_AA)

        # Redraw walls on top of zones
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), WALL_COLOR[::-1], 3)
        cv2.line(canvas, (gap_center_x - gap_width_px // 2, gap_bottom),
                 (gap_center_x + gap_width_px // 2, gap_bottom),
                 FLOOR_COLOR[::-1], 3)

        # Key furniture markers
        furniture_positions = {
            "workstation": [(-3.0, -3.0), (3.0, -3.0)],
            "meeting": [(-5.5, 3.0)],
            "snack": [(6.8, 3.0)],
            "storage": [(5.2, 5.5)],
            "sofa": [(1.2, 5.2), (1.2, 0.8)],
        }
        furniture_colors = {
            "workstation": (120, 120, 200),
            "meeting": (200, 120, 200),
            "snack": (100, 180, 200),
            "storage": (180, 180, 160),
            "sofa": (100, 180, 120),
        }
        for ftype, positions in furniture_positions.items():
            color = furniture_colors.get(ftype, (150, 150, 150))
            for fx, fy in positions:
                px, py = self._world_to_pixel(fx, fy)
                size = 6
                cv2.rectangle(canvas, (px - size, py - size),
                              (px + size, py + size), color[::-1], -1)

        # Robot starting position marker
        rx, ry = self._world_to_pixel(-7.7, -0.7)
        cv2.circle(canvas, (rx, ry), 7, (80, 80, 200), -1)
        cv2.putText(canvas, "Robot", (rx + 12, ry + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 200), 1, cv2.LINE_AA)

        return canvas

    def _load_zones(self, manifest_path: Path) -> None:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._zones = data.get("zones", [])
        dims = data.get("dimensions_m")
        if dims:
            self._width_m = float(dims[0])
            self._depth_m = float(dims[1])
            self._x_offset = self._width_m / 2
            self._y_offset = self._depth_m / 2
            margin = self.margin
            self.canvas_w = int(self._width_m * self.pixels_per_meter) + 2 * margin
            self.canvas_h = int(self._depth_m * self.pixels_per_meter) + 2 * margin
