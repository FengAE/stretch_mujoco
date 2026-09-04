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

- **Placing something on top of a surface** (a monitor on a table or a graspable item on the
  counter): use `measure_top_z(asset, yaw)` for imported furniture. For an `InteractiveAsset`,
  call `Furnisher.place_interactive()` with `z=surface_top_z - asset.mesh_min_z`; this is the
  actual mesh offset used by the workstation and snack-zone placers.
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
