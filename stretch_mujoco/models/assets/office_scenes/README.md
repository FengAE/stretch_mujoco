# Generated open-plan office scenes

Ten textured, single-area MuJoCo offices. Every layout contains a multi-workstation work zone,
a meeting zone, a lounge, a snack counter, and a Hello Robot Stretch placed at a scene-specific
clear starting point. There are no internal room walls; floor bands and furniture define zones.

Open a scene with:

```bash
.venv/bin/python examples/generated_office_scene.py --scene 1
```

The viewer starts in free-camera mode. Use left-drag to rotate, right-drag to pan, and the mouse
wheel to zoom. The scenes reference the centralized `office_assets` library and include a
scene-specific generated Stretch XML that sets the robot's initial freejoint pose.

## Adding or moving a placed object

Don't hand-compute world coordinates. Two tools do that for you:

- **Placing something on top of a surface** (a monitor on a table, a snack on the counter): use
  `measure_top_z(asset, yaw)` to get the surface's true rendered height instead of trusting
  `AssetInfo.bounds` for it — see `furnish_meeting_zone()`. For several small items on the same
  surface (like the snack counter), use `place_on_surface()`: give each item a `width` (its
  spacing slot) and a `lift` (how far its own mesh center sits above the surface); adding an item
  is one more dict in the list, and every position is recomputed automatically.
- **Moving something**: prefer offsets relative to `zone.center` / `zone.bounds` over hardcoded
  absolute coordinates, matching the existing `furnish_*_zone()` functions.

After any change, regenerate and audit before committing:

```bash
.venv/bin/python tools/generate_office_scenes.py
.venv/bin/python tools/audit_office_scenes.py
```

`audit_office_scenes.py` compiles all ten scenes and flags any placed object whose visual mesh
doesn't actually rest on its supporting surface (floating) or sinks into it (penetrating). A clean
run prints `ok` for every scene — that's the bar, not a visual spot-check.
