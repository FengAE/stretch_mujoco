"""Seeded utility scoring with personality weights and repetition penalties."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import random
from typing import TYPE_CHECKING

from stretch_mujoco.semantics import ObjectType, SemanticWorld

if TYPE_CHECKING:
    from .employee import EmployeeAgent
    from .models import ScheduleItem


class UtilityGoal(str, Enum):
    WORK = "work"
    EAT = "eat"
    DRINK = "drink"
    REST = "rest"
    REQUEST_ROBOT = "request_robot"
    MEETING = "meeting"
    WAIT = "wait"


@dataclass(frozen=True)
class UtilityScore:
    goal: UtilityGoal
    score: float
    factors: dict[str, float]


class UtilitySystem:
    """Compute constrained goal scores and choose among near-equivalent goals."""

    def evaluate(
        self,
        agent: "EmployeeAgent",
        world: SemanticWorld,
        schedule_item: "ScheduleItem | None",
    ) -> tuple[UtilityScore, ...]:
        conscientiousness = agent.profile.personality.get("conscientiousness", 0.5)
        patience = agent.profile.personality.get("patience", 0.5)
        work_urgency = 0.85 if schedule_item and schedule_item.activity == "work" else 0.18
        meeting_urgency = (
            0.95
            if schedule_item
            and schedule_item.activity == "meeting"
            and (
                not agent.planner.recent_goals
                or agent.planner.recent_goals[-1] != UtilityGoal.MEETING.value
            )
            else 0.0
        )
        preferred_snack = agent.profile.preferences.get("snack", "bread_snack")
        preferred_drink = agent.profile.preferences.get("drink", "soda_can")
        snack_availability = self._availability(agent, world, preferred_snack)
        drink_availability = self._availability(agent, world, preferred_drink)
        leaving_cost = work_urgency * conscientiousness * 0.45
        snack_request_value = (
            agent.needs.hunger * (1.0 - snack_availability)
            if self._is_requestable(world, preferred_snack)
            else 0.0
        )
        drink_request_value = (
            agent.needs.thirst * (1.0 - drink_availability)
            if self._is_requestable(world, preferred_drink)
            else 0.0
        )
        remote_need = max(
            snack_request_value,
            drink_request_value,
            meeting_urgency * 0.7,
        )

        raw = {
            UtilityGoal.WORK: (
                work_urgency * (0.65 + 0.35 * conscientiousness) - agent.needs.fatigue * 0.45,
                {"urgency": work_urgency, "fatigue": agent.needs.fatigue},
            ),
            UtilityGoal.EAT: (
                agent.needs.hunger * snack_availability,
                {"hunger": agent.needs.hunger, "availability": snack_availability},
            ),
            UtilityGoal.DRINK: (
                agent.needs.thirst * drink_availability,
                {"thirst": agent.needs.thirst, "availability": drink_availability},
            ),
            UtilityGoal.REST: (
                agent.needs.fatigue * (1.1 - 0.2 * conscientiousness),
                {"fatigue": agent.needs.fatigue},
            ),
            UtilityGoal.REQUEST_ROBOT: (
                remote_need * (0.9 + 0.1 * patience) - leaving_cost,
                {"task_value": remote_need, "leaving_cost": leaving_cost},
            ),
            UtilityGoal.MEETING: (
                meeting_urgency,
                {"urgency": meeting_urgency},
            ),
            UtilityGoal.WAIT: (0.08, {"baseline": 0.08}),
        }
        scores = []
        for goal, (base_score, factors) in raw.items():
            repetition_penalty = agent.planner.recent_goals.count(goal.value) * 0.08
            factors = dict(factors)
            factors["repetition_penalty"] = repetition_penalty
            scores.append(UtilityScore(goal, max(0.0, base_score - repetition_penalty), factors))
        return tuple(sorted(scores, key=lambda item: item.score, reverse=True))

    def choose(
        self,
        scores: tuple[UtilityScore, ...],
        *,
        agent_id: str,
        day: int,
        decision_index: int,
        seed: int,
    ) -> UtilityScore:
        maximum = scores[0].score
        candidates = [
            score for score in scores if score.score > 0.0 and score.score >= maximum - 0.12
        ]
        if not candidates:
            candidates = [scores[0]]
        digest = hashlib.sha256(
            f"{seed}:{agent_id}:{day}:{decision_index}".encode("utf-8")
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        weights = [max(0.02, candidate.score) for candidate in candidates]
        return rng.choices(candidates, weights=weights, k=1)[0]

    @staticmethod
    def _availability(agent: "EmployeeAgent", world: SemanticWorld, object_id: str) -> float:
        semantic_object = world.object(object_id)
        if semantic_object.get("consumed", False) or not semantic_object.get("available", True):
            return 0.0
        location = world.location_of(object_id)
        if agent.state.held_object == object_id:
            return 1.0
        if location is not None and location.object_id == agent.state.location:
            return 0.95
        if semantic_object.object_type == ObjectType.SNACK:
            return 0.48
        return 0.30

    @staticmethod
    def _is_requestable(world: SemanticWorld, object_id: str) -> bool:
        semantic_object = world.object(object_id)
        return not semantic_object.get("consumed", False) and semantic_object.get("available", True)
