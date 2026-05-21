# env/custom_walker2d.py
"""Custom Walker2d environment built on the shared DR base.

This file:
- defines CustomWalker2d as a thin subclass of CustomMujocoDRBase
- registers the Gymnasium IDs for source/target

['world', 'torso', 'thigh', 'leg', 'foot', 'thigh_left', 'leg_left', 'foot_left']
"""

from __future__ import annotations

import gymnasium as gym
import mujoco

try:
    CAMERA_TRACKING = mujoco.mjtCamera.mjCAMERA_TRACKING  # type: ignore[attr-defined]
except Exception:
    # Fallback: MuJoCo enum value for TRACKING (works across versions)
    CAMERA_TRACKING = 2

import numpy as np

from env.custom_mujoco_dr_base import CustomMujocoDRBase

DEFAULT_WALKER2D_CAMERA_CONFIG = {
    # Camera values are not critical for training; tweak only for nicer rendering.
    # "fixedcamid": 0,
    "type": CAMERA_TRACKING,
    "trackbodyid": 1,
    "distance": 5.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
    "orthographic": 0,
}


class CustomWalker2d(CustomMujocoDRBase):
    ASSET_XML = "walker2d.xml"
    DEFAULT_CAMERA_CONFIG = DEFAULT_WALKER2D_CAMERA_CONFIG

    # Walker2d: apply the domain shift on the torso/pelvis body.
    # IMPORTANT: confirm the exact body name via get_extra_param()["body_names"].
    # Common names are "torso" or "pelvis" depending on the XML.
    DOMAIN_SHIFT_BODY = "torso"
    DOMAIN_SHIFT_DELTA_KG = -1.0

    # Walker2d: randomize leg segment masses on BOTH legs.
    # IMPORTANT: confirm the exact names via get_extra_param()["body_names"].
    # Typical names you might see:
    #   "thigh", "leg", "foot", "thigh_left", "leg_left", "foot_left"
    # Some XMLs use right/left naming differently; adjust this tuple accordingly.
    UDR_TARGET_BODIES = (
        "thigh",
        "leg",
        "foot",
        "thigh_left",
        "leg_left",
        "foot_left",
    )
    ADR_TARGET_BODIES = UDR_TARGET_BODIES

    # Walker2d health defaults (may vary slightly across versions; tune if needed)
    HEALTHY_Z_RANGE = (0.8, 2.0)
    HEALTHY_ANGLE_RANGE = (-1.0, 1.0)
    # Discourage hopping / unstable gaits (Walker2d-specific)
    VERTICAL_VEL_COST_WEIGHT = 5e-2
    ANGLE_COST_WEIGHT = 1e-3

    def _get_rew(self, x_velocity: float, action: np.ndarray):
        # Forward progress (same idea as base)
        forward_reward = self._forward_reward_weight * x_velocity

        # Healthy reward (survival bonus)
        healthy_reward = self.healthy_reward

        # Control cost (already handled in base)
        ctrl_cost = self.control_cost(action)

        # Walker2d anti-hopping terms:
        # qvel[1] is usually vertical velocity of the torso/pelvis in Walker2d obs
        # If your indexing differs, this still generally works, but you can confirm via obs structure.
        y_velocity = float(self.data.qvel[1])
        vertical_vel_cost = self.VERTICAL_VEL_COST_WEIGHT * (y_velocity**2)

        height = float(self.data.qpos[1])
        height_cost = 1e-3 * (
            (height - 1.3) ** 2
        )  # 1.3 is a typical comfortable height

        # Penalize torso angle magnitude (qpos[2] = torso angle in most Walker2d models)
        torso_angle = float(self.data.qpos[2])
        angle_cost = self.ANGLE_COST_WEIGHT * (torso_angle**2)

        reward = (
            (forward_reward + healthy_reward)
            - ctrl_cost
            - vertical_vel_cost
            - angle_cost
            - height_cost
        )

        info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_vertical_vel": -vertical_vel_cost,
            "reward_angle": -angle_cost,
        }
        return float(reward), info


# ---------- Gymnasium registration (Walker2d) ----------

gym.register(
    id="CustomWalker2d-v0",
    entry_point=f"{__name__}:CustomWalker2d",
    max_episode_steps=1000,
)

gym.register(
    id="CustomWalker2d-source-v0",
    entry_point=f"{__name__}:CustomWalker2d",
    max_episode_steps=1000,
    kwargs={"domain": "source"},
)

gym.register(
    id="CustomWalker2d-target-v0",
    entry_point=f"{__name__}:CustomWalker2d",
    max_episode_steps=1000,
    kwargs={"domain": "target"},
)
