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
