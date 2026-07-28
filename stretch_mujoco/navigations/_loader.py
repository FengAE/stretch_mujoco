"""Internal helper for loading modules from subdirectories whose names
contain characters that are invalid in Python identifiers (e.g. ``A*``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType
from typing import Optional


def _load_module(subdirectory: str, module_name: str) -> ModuleType:
    """Load a Python module from a subdirectory of *this* package.

    Parameters
    ----------
    subdirectory:
        Name of the subdirectory (relative to the ``navigations`` package
        root) that contains the module file.
    module_name:
        Base name of the ``.py`` file (without extension).

    Returns
    -------
    ModuleType
        The loaded module.
    """
    package_dir = os.path.dirname(__file__)
    filepath = os.path.join(package_dir, subdirectory, f"{module_name}.py")

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Module file not found: {filepath}")

    # Use a fully-qualified name so that the module appears sensibly in
    # tracebacks and the sys.modules cache.
    full_name = f"stretch_mujoco.navigations.{subdirectory}.{module_name}"

    # Return cached module if already loaded
    existing: Optional[ModuleType] = sys.modules.get(full_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(full_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module
