"""Online Qwen2.5-VL pixel-goal selector.

Calls Qwen2.5-VL through an OpenAI-compatible chat-completions endpoint
(DashScope by default; any compatible server such as vLLM or SiliconFlow also
works) and asks it to point at the pixel the robot should move toward to make
progress on a natural-language instruction.

The heavy VLM runs in the cloud — no local weights and no GPU are needed for
the language part.  The output is a **normalised** image coordinate in ``[0, 1]``
so it is independent of the model's internal image resizing; the caller scales
it to the actual captured image resolution.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Optional, Tuple

import numpy as np


class QwenPixelGoalSelector:
    """Point a robot at a goal pixel using an online Qwen2.5-VL model."""

    def __init__(
        self,
        *,
        model: str = "qwen-vl-max",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        detail: str = "low",
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and > 0")
        if detail not in {"low", "high", "auto"}:
            raise ValueError("detail must be one of 'low', 'high', 'auto'")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get(
            "QWEN_API_KEY"
        ) or ""
        self.timeout = float(timeout)
        self.detail = detail

    def validate_environment(self) -> None:
        """Raise when the OpenAI client or an API key is unavailable."""
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "QwenPixelGoalSelector requires the 'openai' package "
                "(pip install openai)"
            ) from exc
        if not self.api_key:
            raise ValueError(
                "No API key. Set DASHSCOPE_API_KEY (or QWEN_API_KEY), or pass api_key=."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_goal(
        self,
        image_bytes: bytes,
        instruction: str,
        image_size: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """Return the ``(u, v)`` goal pixel in the captured image.

        Returns ``None`` when the model reports the goal has been reached
        (``STOP``).
        """
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("image_bytes must contain encoded image bytes")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")

        normalized = self._query_normalized(bytes(image_bytes), instruction.strip())
        if normalized is None:
            return None
        width = int(image_size[0])
        height = int(image_size[1])
        u = int(round(float(normalized[0]) * (width - 1)))
        v = int(round(float(normalized[1]) * (height - 1)))
        return (max(0, min(u, width - 1)), max(0, min(v, height - 1)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_normalized(
        self,
        image_bytes: bytes,
        instruction: str,
    ) -> Optional[Tuple[float, float]]:
        """Call the online model and return normalised ``(x, y)`` or ``None``."""
        self.validate_environment()
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")

        prompt = (
            "You are an autonomous navigation assistant on a mobile robot.\n"
            f'The robot must follow this instruction: "{instruction}"\n\n'
            "Look at the attached camera image and decide where the robot should "
            "head next to make progress toward the instruction.\n"
            "Reply with the NORMALIZED image coordinates of that point as two "
            "numbers in [0, 1], separated by a space:\n"
            "  x y\n"
            "where x = fraction across the width (0 = left, 1 = right) and "
            "y = fraction down the height (0 = top, 1 = bottom).\n"
            "Reply with ONLY the two numbers.\n"
            "If the goal described by the instruction has already been reached, "
            "reply with exactly: STOP"
        )
        messages = [
            {"role": "system", "content": "You are a robot navigation goal pointer."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": self.detail},
                    },
                ],
            },
        ]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=16,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        return self.parse_response(raw)

    @staticmethod
    def parse_response(raw: str) -> Optional[Tuple[float, float]]:
        """Parse the model reply into normalised ``(x, y)`` or ``None`` (STOP)."""
        text = (raw or "").strip()
        if re.search(r"\bSTOP\b", text, re.IGNORECASE):
            return None
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        if len(numbers) < 2:
            raise RuntimeError(f"Qwen did not return coordinates: {raw!r}")
        x = float(np.clip(float(numbers[0]), 0.0, 1.0))
        y = float(np.clip(float(numbers[1]), 0.0, 1.0))
        return (x, y)
