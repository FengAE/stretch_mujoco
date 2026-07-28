"""Record all enabled robot RGBD cameras as one MP4 mosaic."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RobotVideoRecorder:
    def __init__(self, sim: Any, cameras: Any, path: str | Path, fps: float = 15.0) -> None:
        self.sim = sim
        self.cameras = list(cameras) if isinstance(cameras, (list, tuple)) else [cameras]
        self.recorded_cameras: list[str] = []
        self.path = Path(path)
        self.fps = fps
        self.frames = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="RobotVideoRecorder", daemon=True)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=5.0)
        return {
            "path": str(self.path),
            "cameras": self.recorded_cameras,
            "fps": self.fps,
            "frames": self.frames,
            "error": self.error,
        }

    def _run(self) -> None:
        writer: cv2.VideoWriter | None = None
        period = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                try:
                    snapshot = self.sim.pull_camera_data()
                    tiles = []
                    for camera in self.cameras:
                        try:
                            data = snapshot.get_camera_data(camera, auto_correct_rgb=True)
                        except ValueError:
                            continue
                        tiles.append(
                            _video_tile(data, camera.name, getattr(camera, "is_depth", False))
                        )
                        if camera.name not in self.recorded_cameras:
                            self.recorded_cameras.append(camera.name)
                    if not tiles:
                        raise ValueError("No camera frame is ready")
                    frame = _mosaic(tiles)
                except ValueError:
                    self._stop.wait(min(period, 0.05))
                    continue

                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(self.path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV could not open the MP4 video writer")
                writer.write(frame)
                self.frames += 1
                self._stop.wait(max(0.0, period - (time.perf_counter() - started)))
        except Exception as exc:
            self.error = str(exc)
        finally:
            if writer is not None:
                writer.release()


def _video_tile(data: np.ndarray, label: str, is_depth: bool) -> np.ndarray:
    if is_depth:
        finite = np.isfinite(data)
        if finite.any():
            low, high = np.percentile(data[finite], (2, 98))
            scaled = np.clip((data - low) / max(float(high - low), 1e-6), 0, 1)
            tile = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        else:
            tile = np.zeros((*data.shape[:2], 3), dtype=np.uint8)
    else:
        tile = data if data.ndim == 3 else cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        tile = np.asarray(tile, dtype=np.uint8)
    cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return tile


def _mosaic(tiles: list[np.ndarray]) -> np.ndarray:
    cell_width = min(640, max(tile.shape[1] for tile in tiles))
    cell_height = min(480, max(tile.shape[0] for tile in tiles))
    resized = [cv2.resize(tile, (cell_width, cell_height)) for tile in tiles]
    columns = min(2, len(resized))
    blank = np.zeros_like(resized[0])
    rows = []
    for offset in range(0, len(resized), columns):
        row = resized[offset : offset + columns]
        rows.append(np.hstack(row + [blank] * (columns - len(row))))
    return np.vstack(rows)
