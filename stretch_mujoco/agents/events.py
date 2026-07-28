"""Seeded daily office events constrained by registered semantic objects."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
from typing import Any

from stretch_mujoco.semantics import ObjectType, RelationType, SemanticWorld


@dataclass(frozen=True)
class DailyOfficeEvent:
    event_type: str
    target: str
    details: dict[str, Any] = field(default_factory=dict)


class DailyOfficeEventGenerator:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def generate(
        self, day: int, world: SemanticWorld, employee_ids: tuple[str, ...]
    ) -> tuple[DailyOfficeEvent, ...]:
        digest = hashlib.sha256(f"{self.seed}:office:{day}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        events: list[DailyOfficeEvent] = []
        if rng.random() < 0.65:
            snacks = [obj.object_id for obj in world.objects_of_type(ObjectType.SNACK)]
            if snacks:
                events.append(
                    DailyOfficeEvent(
                        "snack_unavailable",
                        rng.choice(snacks),
                        {"available": False},
                    )
                )
        if rng.random() < 0.75:
            documents = [obj.object_id for obj in world.objects_of_type(ObjectType.DOCUMENT)]
            if documents and employee_ids:
                document = rng.choice(documents)
                employee = rng.choice(employee_ids)
                events.append(
                    DailyOfficeEvent(
                        "urgent_document",
                        document,
                        {"requester": employee, "urgent": True},
                    )
                )
        if not events:
            documents = [obj.object_id for obj in world.objects_of_type(ObjectType.DOCUMENT)]
            if documents and employee_ids:
                events.append(
                    DailyOfficeEvent(
                        "document_review",
                        rng.choice(documents),
                        {"requester": rng.choice(employee_ids), "urgent": False},
                    )
                )
        return tuple(events)

    @staticmethod
    def apply(event: DailyOfficeEvent, world: SemanticWorld) -> None:
        world.object(event.target).attributes.update(event.details)
        if event.event_type in {"urgent_document", "document_review"}:
            world.add_relation(
                event.target,
                RelationType.REQUESTED_BY,
                event.details["requester"],
            )
