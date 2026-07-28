# Humanoid assets

The active office NPC is generated locally from the licensed SMPL-X v1.1
parameters. Generated meshes are ignored by Git and must be recreated with the
commands in `private/README.md` after obtaining the official model.

The generated character contains:

- a 1.72 meter SMPL-X neutral body;
- one aligned body mesh per frame with UV-baked skin, shirt, pants, and shoe regions;
- four Idle frames, eight Walk frames, and one Sit frame;
- a lightweight runtime mesh player with no Torch dependency.

## Legacy preview assets

`cesium_man.glb` is the animated, skinned, and textured CesiumMan sample from
the Khronos glTF Sample Assets repository. Copyright 2017 Cesium, licensed
under CC BY 4.0 with the trademark limitations documented by the source:

https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMan

`cesium_man_idle.obj` is frame 36 of the bundled animation, converted with
Blender and scaled by `office_scene.xml` to approximately 1.72 meters.
`cesium_man.png` is its embedded 1024 x 1024 texture converted to the format
accepted by MuJoCo.

The smaller RiggedFigure files are retained as a CC BY 4.0 skeleton reference:

https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/RiggedFigure

Both GLB files retain their original skeletons and animations for connecting
to a MuJoCo articulated humanoid in a later iteration.
