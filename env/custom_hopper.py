# env/custom_hopper.py
"""Custom Hopper environment built on the shared DR base.

This file:
- defines CustomHopper as a thin subclass of CustomMujocoDRBase
- registers the Gymnasium IDs for source/target
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np

try:
    CAMERA_TRACKING = mujoco.mjtCamera.mjCAMERA_TRACKING  # type: ignore[attr-defined]
except Exception:
    # Fallback: MuJoCo enum value for TRACKING (works across versions)
    CAMERA_TRACKING = 2

from env.custom_mujoco_dr_base import CustomMujocoDRBase

DEFAULT_HOPPER_CAMERA_CONFIG = {
    "type": CAMERA_TRACKING,
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class CustomHopper(CustomMujocoDRBase):
    ASSET_XML = "hopper.xml"
    DEFAULT_CAMERA_CONFIG = DEFAULT_HOPPER_CAMERA_CONFIG

    # Hopper: apply the domain shift on the torso body.
    # If your XML uses a different name, change only this constant.
    DOMAIN_SHIFT_BODY = "torso"
    DOMAIN_SHIFT_DELTA_KG = -1.0

    # Hopper: randomize only these bodies (as in your original code).
    # If names differ, call get_extra_param() and inspect body_names.
    UDR_TARGET_BODIES = ("thigh", "leg", "foot")
    ADR_TARGET_BODIES = UDR_TARGET_BODIES

    # Hopper health defaults
    HEALTHY_Z_RANGE = (0.7, float("inf"))
    HEALTHY_ANGLE_RANGE = (-0.2, 0.2)


# ---------- Gymnasium registration (Hopper) ----------

gym.register(
    id="CustomHopper-v0",
    entry_point=f"{__name__}:CustomHopper",
    max_episode_steps=500,
)

gym.register(
    id="CustomHopper-source-v0",
    entry_point=f"{__name__}:CustomHopper",
    max_episode_steps=500,
    kwargs={"domain": "source"},
)

gym.register(
    id="CustomHopper-target-v0",
    entry_point=f"{__name__}:CustomHopper",
    max_episode_steps=500,
    kwargs={"domain": "target"},
)
