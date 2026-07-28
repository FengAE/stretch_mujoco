"""Event-only LLM request gate; never called from physics or animation updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class LLMTrigger(str, Enum):
    DAY_START = "day_start"
    NEW_TASK = "new_task"
    DIALOGUE = "dialogue"
    REPEATED_FAILURE = "repeated_failure"
    UNEXPECTED_CHANGE = "unexpected_change"
    REINTERPRET_PLAN = "reinterpret_plan"


@dataclass(frozen=True)
class LLMRequest:
    trigger: LLMTrigger
    agent_id: str
    day: int
    minute_of_day: float
    context: dict[str, Any] = field(default_factory=dict)


class EventDrivenLLMGateway:
    """Queue allowed LLM events and enforce a per-agent daily call budget."""

    def __init__(self, daily_budget: int = 30) -> None:
        if daily_budget <= 0:
            raise ValueError("LLM daily budget must be positive")
        self.daily_budget = daily_budget
        self._pending: list[LLMRequest] = []
        self._calls: dict[tuple[int, str], int] = {}
        self._issued: dict[tuple[int, str], int] = {}

    def queue(
        self,
        trigger: LLMTrigger,
        agent_id: str,
        day: int,
        minute_of_day: float,
        context: dict[str, Any] | None = None,
    ) -> bool:
        key = (day, agent_id)
        if self._issued.get(key, 0) >= self.daily_budget:
            return False
        self._pending.append(LLMRequest(trigger, agent_id, day, minute_of_day, context or {}))
        self._issued[key] = self._issued.get(key, 0) + 1
        return True

    def drain_requests(self) -> tuple[LLMRequest, ...]:
        requests = tuple(self._pending)
        self._pending.clear()
        return requests

    def process(
        self,
        request: LLMRequest,
        provider: Callable[[LLMRequest], dict[str, Any]],
    ) -> dict[str, Any]:
        key = (request.day, request.agent_id)
        if self._calls.get(key, 0) >= self.daily_budget:
            raise RuntimeError("LLM daily budget exhausted")
        response = provider(request)
        self._calls[key] = self._calls.get(key, 0) + 1
        return response

    def calls_for(self, day: int, agent_id: str) -> int:
        return self._calls.get((day, agent_id), 0)
