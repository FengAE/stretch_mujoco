# Stanford TidyBot Description (MJCF)

> [!IMPORTANT]
> Requires MuJoCo 3.1.0 or later.

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for a full history of changes.

## Overview

This package contains a simplified robot description (MJCF) of the [Stanford TidyBot](https://tidybot.cs.princeton.edu) developed by Jimmy Wu and collaborators from the [Interactive Perception and Robot Learning Lab](https://iprl.stanford.edu) at Stanford University.

<p float="left">
  <img src="tidybot.png" width="400">
  <img src="tidybot_base.png" width="400">
</p>

## MJCF Model

* The mobile base MJCF was created from a CAD model of the base.
* The Kinova Gen3 MJCF is taken from Menagerie. See [kinova_gen3](../kinova_gen3/).
* The Robotiq 2F-85 MJCF is taken from Menagerie. See [robotiq_2f85](../robotiq_2f85/).

## Local camera configuration

The base camera uses the public calibration for TidyBot robot #2's Logitech
C930e (serial `44251E9E`), scaled from 1280x720 to the 640x360 stream used by
the official robot code. The wrist camera comes from the Menagerie simulation
model and is marked as simulation-only by the Python interface. The official
project's two ceiling cameras are not embedded here because only planar image
alignment and height are public, not reusable six-degree-of-freedom scene
extrinsics.

## License

The TidyBot base is released under an MIT License. The Kinova Gen3 and Robotiq 2F-85 MJCFs retain
their original licenses. For more information, see the [LICENSE](LICENSE) file.

## Publications

If you use this work in an academic context, please cite the following publication:

```bibtex
@article{wu2023tidybot,
  title = {TidyBot: Personalized Robot Assistance with Large Language Models},
  author = {Wu, Jimmy and Antonova, Rika and Kan, Adam and Lepert, Marion and Zeng, Andy and Song, Shuran and Bohg, Jeannette and Rusinkiewicz, Szymon and Funkhouser, Thomas},
  journal = {Autonomous Robots},
  year = {2023}
}
```
