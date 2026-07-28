# Low-Compute Autonomous Behavior

The office runtime separates behavior into three layers.

## Layer 1: Deterministic State Machine

The state machine runs at 4 Hz by default. It executes only actions already placed
in a validated plan queue. Physics, animation, action timing, reservations, and
result verification never call an LLM.

## Layer 2: Schedule and Utility

Needs update at 0.5 Hz. Utility decisions occur at seeded intervals between 5 and
15 simulation seconds. Scores combine schedule urgency, needs, availability,
personality, the cost of leaving work, and a recent-goal repetition penalty.

Near-equivalent goals use seeded weighted selection. Plans remain constrained by
the semantic world. For example, meeting preparation chooses between:

- moving to the cabinet, opening it, taking the report, and attending;
- requesting report delivery, then moving to the meeting table.

## Layer 3: Event-Only LLM

`tick()` only queues `LLMRequest` records. It never invokes a provider. Allowed
triggers are day start, new task, dialogue, repeated failure, unexpected change,
and plan reinterpretation. Calls must be processed explicitly outside the
simulation loop:

```python
for request in runtime.drain_llm_requests():
    response = runtime.llm.process(request, provider)
    runtime.apply_llm_response(request, response)
```

The default hard limit is 30 issued requests per employee per day. Daily schedule
jitter, weighted plan variants, repetition penalties, seeded office events, and
recent memory reduce exact routine repetition without unconstrained wandering.
LLM responses can only replace a validated schedule draft or submit a member of the
closed `ActionType` set; both paths retain normal location, reservation, and
permission checks.

## Local LLM Configuration

Put credentials in the ignored local file:

`stretch_mujoco/models/office_llm.local.json`

Set `enabled` to `true`, then fill in `api_key`, `base_url`, and `model`. Use
`api_mode: "responses"` for a Responses-compatible endpoint, or
`api_mode: "chat_completions"` for a Chat Completions-compatible endpoint. Keep
the file private:

```bash
chmod 600 stretch_mujoco/models/office_llm.local.json
```

The adjacent `office_llm.example.json` is the tracked, credential-free template.
The local file is excluded from Git and source distributions. Provider calls remain
explicit and outside `tick()`:

```python
provider = runtime.create_llm_provider()
for request in runtime.drain_llm_requests():
    response = runtime.llm.process(request, provider)
    runtime.apply_llm_response(request, response)
```

The runtime rejects missing placeholders, invalid URLs, and permissive credential
file modes when the provider is enabled. Diagnostic output redacts the API key.

Run one accelerated 09:00-18:00 day for `employee_01` with the configured provider:

```bash
.venv/bin/python examples/office_llm_day.py --output /tmp/office_llm_day.json
```

The LLM creates the high-level schedule and handles event requests. The local
utility planner and state machine execute routine actions. In this behavior-only
runner, pending Stretch deliveries receive a semantic success result; replace that
line with the physical robot controller handoff for an embodied run.

Visually replay the saved workday in MuJoCo, compressed to two minutes:

```bash
.venv/bin/python examples/office_day_replay.py --duration 120
```

Use `--loop` to repeat it or change `--duration` to control playback speed. The
replay drives the SMPL-X root between registered office interaction sites and uses
the baked walk, sit, work, eat, and idle clips. Root motion and walking scale with
the ratio between the office timeline and replay duration, so travel takes the same
amount of office time in short and long replays. Work and eating gestures retain a
natural local animation speed. During lunch, the NPC walks to the snack area,
performs a hand-to-mouth animation, then marks the food consumed and disables its
render and collision geoms. A looped replay restores the food at the next day start.

On a headless machine, render the same replay to a 960x540 MP4 using EGL:

```bash
.venv/bin/python examples/office_day_replay.py \
  --duration 120 --fps 24 --output office_day_replay.mp4
```

The fixed overview camera captures the whole office. Each frame includes the office
clock, current activity, semantic location, and schedule item.

Run a deterministic trace:

```bash
uv run examples/office_autonomy.py --seconds 60
```

## GraspGen Office Grasp

Install the optional lightweight client dependencies:

```bash
uv sync --extra graspgen
```

Run RGB-D detection, constrained Stretch IK, a horizontal approach followed by
vertical cereal-box carry, and headless MP4 recording against the persistent
GraspGen service:

```bash
.venv/bin/python examples/office_graspgen.py \
  --host 10.29.150.95 --port 5557 \
  --object-id cereal_box --prompt "cereal box" \
  --output office_graspgen.mp4
```

The overview video includes the rotated D435i feed and detection box. The motion
sequence observes, plans, approaches, closes, retracts beyond the snack-counter
edge, rotates the cleared gripper vertically downward, and only then raises the
object. The D435i inset is refreshed every video frame. Request data, response
poses, selected IK candidate, contact metrics, and measured object displacement
are saved under `output/office_graspgen/`. Attachment and lifting are permitted
only after MuJoCo reports contact between the target object and both the left and
right gripper fingers; semantic displacement alone is not treated as a successful
grasp. After that physical-contact gate passes, kinematic attachment is used to
keep the carried object stable. Use `--dry-run` to stop after reachability planning.
