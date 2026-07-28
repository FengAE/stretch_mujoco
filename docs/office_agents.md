# Office Employee Agents

`OfficeAgentRuntime` is the deterministic, LLM-free execution layer for employee
agents. An optional LLM may produce `ActionCommand` JSON outside the simulation,
but it cannot introduce new action names or bypass runtime validation.

## Agent Components

Each configured employee owns independent profile, needs, schedule, memory,
planner, perception, executor, and `EmployeeState` instances. Add employees under
`employees` in `stretch_mujoco/models/office_agents.json`; each ID must also be a
semantic `Employee` object.

Schedule start times receive deterministic per-day jitter. The same seed reproduces
an experiment, while different days do not repeat the exact same timeline.

## Validation Order

Before an action starts, the runtime checks:

1. Agent identity and executor availability.
2. Target and parameter existence.
3. Required location and object type.
4. Action-specific preconditions and perception.
5. Resource reservations and conflicts.
6. `ALLOWED_FOR` permissions for confidential objects.

Rejected actions produce `action_rejected` events and do not mutate the world.

## Simulator Loop

```python
runtime = sim.create_office_agent_runtime(auto_plan=True)
sim.start(headless=True)

while sim.is_running():
    snapshot = sim.pull_semantic_state()
    events = runtime.tick(0.1, snapshot)

    for task in runtime.pending_robot_tasks():
        robot_controller.submit(task)
```

The runtime executes logical actions and verifies semantic effects. Navigation,
grasping, and handover controllers remain responsible for physical execution.
After a controller reaches a terminal state, report it explicitly:

```python
runtime.complete_robot_task(
    task.task_id,
    success=True,
    semantic_snapshot=sim.pull_semantic_state(),
)
```

Successful completion updates `ON` and `REQUESTED_BY`, releases the reservation,
records memory, and emits `robot_task_completed`. A failed controller result does
not claim that the object moved. When a snapshot is supplied, success is downgraded
to failure unless the object is physically near the destination.

## Closed Action Set

The allowed actions are `idle`, `move_to`, `sit`, `work`, `rest`, `eat`, `drink`,
`pick_up`, `put_down`, `request_robot`, `use_computer`, `open_cabinet`, `handover`,
and `attend_meeting`.
