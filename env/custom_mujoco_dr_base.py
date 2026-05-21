"""Shared MuJoCo base env with (future) Domain Randomization.

Design goals:
- Keep the DR interface stable for the 3 settings: no-DR / UDR / ADR
- Maximize code reuse between Hopper and Walker2d
- Randomize masses by *body name* (robust across envs)

Subclasses should define:
- ASSET_XML
- DEFAULT_CAMERA_CONFIG
- DOMAIN_SHIFT_BODY / DOMAIN_SHIFT_DELTA_KG (optional)
- UDR_TARGET_BODIES (optional)
- HEALTHY_* defaults (optional) or override is_healthy
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


@dataclass(frozen=True)
class UDRConfig:
    enabled: bool = False
    ratio: float = 0.20
    debug: bool = False


@dataclass(frozen=True)
class ADRConfig:
    enabled: bool = False
    debug: bool = False
    initial_scale: float = 0.20
    scale_step: float = 0.05
    success_threshold: float = 800.0
    min_scale: float = 0.05
    max_scale: float = 0.80


class CustomMujocoDRBase(MujocoEnv, utils.EzPickle):
    """Reusable MuJoCo env base with (future) Domain Randomization."""

    # Shared: override in subclasses
    ASSET_XML: str = ""  # e.g., "hopper.xml" / "walker2d.xml"
    DEFAULT_CAMERA_CONFIG: Dict[str, Union[float, int]] = {}

    # Shared: health defaults (can be overridden)
    HEALTHY_STATE_RANGE: Tuple[float, float] = (-100.0, 100.0)
    HEALTHY_Z_RANGE: Tuple[float, float] = (0.7, float("inf"))
    HEALTHY_ANGLE_RANGE: Tuple[float, float] = (-0.2, 0.2)

    # Per-env: apply the "source" domain shift on this body (if set)
    DOMAIN_SHIFT_BODY: Optional[str] = None
    DOMAIN_SHIFT_DELTA_KG: float = -1.0

    # Per-env: which bodies to randomize (by name)
    UDR_TARGET_BODIES: Tuple[str, ...] = ()

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
        "render_fps": 125,
    }

    def __init__(
        self,
        xml_file: Optional[str] = None,
        frame_skip: int = 4,
        default_camera_config: Optional[Dict[str, Union[float, int]]] = None,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] | None = None,
        healthy_z_range: Tuple[float, float] | None = None,
        healthy_angle_range: Tuple[float, float] | None = None,
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        # Shared DR interface (supports: no-DR / UDR / ADR)
        domain: Optional[str] = None,
        udr_enabled: bool = False,
        udr_ratio: float = 0.20,
        debug_udr: bool = False,
        adr_enabled: bool = False,
        debug_adr: bool = False,
        adr_initial_scale: float = 0.20,
        adr_scale_step: float = 0.05,
        adr_success_threshold: float = 800.0,
        adr_min_scale: float = 0.05,
        adr_max_scale: float = 0.80,
        # optional DR logging path (train.py passes logs_dir/dr_stats.csv)
        dr_log_path: Optional[str] = None,
        **kwargs,
    ):
        if not self.ASSET_XML:
            raise ValueError("ASSET_XML must be set in subclass")

        if xml_file is None:
            xml_file = self.ASSET_XML

        if default_camera_config is None:
            default_camera_config = dict(self.DEFAULT_CAMERA_CONFIG)

        # EzPickle (reproducibility / Gymnasium convention)
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
            udr_enabled,
            udr_ratio,
            debug_udr,
            adr_enabled,
            debug_adr,
            adr_initial_scale,
            adr_scale_step,
            adr_success_threshold,
            adr_min_scale,
            adr_max_scale,
            dr_log_path,
            **kwargs,
        )

        print(f"ctrl_cost_weight: {ctrl_cost_weight:.6f}")

        # Store parameters
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range or self.HEALTHY_STATE_RANGE
        self._healthy_z_range = healthy_z_range or self.HEALTHY_Z_RANGE
        self._healthy_angle_range = healthy_angle_range or self.HEALTHY_ANGLE_RANGE
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.domain = domain

        # DR log path
        self._dr_log_path = dr_log_path
        self._dr_log_header_written = False

        # Resolve xml path if using the default filename
        if xml_file == self.ASSET_XML:
            xml_file = os.path.join(
                os.path.dirname(__file__), f"assets/{self.ASSET_XML}"
            )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # Metadata FPS from dt
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # Observation space
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - int(exclude_current_positions_from_observation)
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        self.observation_structure = {
            "skipped_qpos": int(exclude_current_positions_from_observation),
            "qpos": self.data.qpos.size
            - int(exclude_current_positions_from_observation),
            "qvel": self.data.qvel.size,
        }

        # DR configs
        self.udr = UDRConfig(enabled=udr_enabled, ratio=udr_ratio, debug=debug_udr)
        self.adr = ADRConfig(
            enabled=adr_enabled,
            debug=debug_adr,
            initial_scale=adr_initial_scale,
            scale_step=adr_scale_step,
            success_threshold=adr_success_threshold,
            min_scale=adr_min_scale,
            max_scale=adr_max_scale,
        )
        self._adr_scale = float(self.adr.initial_scale)
        self._episode_return = 0.0
        self._episode_idx = 0

        # Snapshot nominal masses by body name (robust across envs)
        self._body_name_to_id = self._build_body_name_to_id()
        self._nominal_body_masses = {
            name: float(self.model.body_mass[self._body_id(name)])
            for name in self._all_body_names()
        }

        # Apply domain shift (source) if configured
        if self.domain == "source" and self.DOMAIN_SHIFT_BODY:
            bid = self._body_id(self.DOMAIN_SHIFT_BODY)
            self.model.body_mass[bid] += float(self.DOMAIN_SHIFT_DELTA_KG)

        # IMPORTANT: Always define ADR baseline masses (if ADR_TARGET_BODIES exists).
        # Baseline is the *current* mass after any domain shift.
        self._adr_baseline_masses = {
            name: float(self.model.body_mass[self._body_id(name)])
            for name in getattr(self, "ADR_TARGET_BODIES", ())
        }

        print("ADR enabled:", self.adr.enabled)
        print("ADR_TARGET_BODIES:", getattr(self, "ADR_TARGET_BODIES", None))
        print("UDR_TARGET_BODIES:", getattr(self, "UDR_TARGET_BODIES", None))

    # ---------- shared helpers ----------

    def _all_body_names(self) -> Tuple[str, ...]:
        return tuple(self.model.body(i).name for i in range(self.model.nbody))

    def _build_body_name_to_id(self) -> Dict[str, int]:
        return {self.model.body(i).name: i for i in range(self.model.nbody)}

    def _body_id(self, name: str) -> int:
        try:
            return self._body_name_to_id[name]
        except KeyError as e:
            raise KeyError(
                f"Body '{name}' not found. Available: {sorted(self._body_name_to_id.keys())}"
            ) from e

    # ---------- shared reward/health (override if needed) ----------

    @property
    def healthy_reward(self) -> float:
        return float(self.is_healthy) * self._healthy_reward

    def control_cost(self, action: np.ndarray) -> float:
        return self._ctrl_cost_weight * float(np.sum(np.square(action)))

    @property
    def is_healthy(self) -> bool:
        # Default style (Hopper-like): z + angle + state range
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        return bool(all((healthy_state, healthy_z, healthy_angle)))

    def _get_obs(self) -> np.ndarray:
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def _get_rew(self, x_velocity: float, action: np.ndarray):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        ctrl_cost = self.control_cost(action)
        reward = (forward_reward + healthy_reward) - ctrl_cost
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }
        return float(reward), info

    # ---------- shared step/reset ----------

    def step(self, action: np.ndarray):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        obs = self._get_obs()
        reward, reward_info = self._get_rew(float(x_velocity), action)
        self._episode_return += float(reward)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy

        info = {
            "x_position": float(x_position_after),
            "z_distance_from_origin": float(self.data.qpos[1] - self.init_qpos[1]),
            "x_velocity": float(x_velocity),
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, bool(terminated), False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        # DR hook at episode reset (ADR/UDR)
        self.apply_domain_randomization()

        # Reset episode return AFTER updating ADR based on last episode
        self._episode_return = 0.0

        self.set_state(qpos, qvel)
        return self._get_obs()

    # ---------- DR logging ----------

    def _maybe_write_dr_header(self) -> None:
        if not self._dr_log_path or self._dr_log_header_written:
            return
        out_dir = os.path.dirname(self._dr_log_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Write header only if file doesn't exist or is empty
        need_header = True
        try:
            if (
                os.path.exists(self._dr_log_path)
                and os.path.getsize(self._dr_log_path) > 0
            ):
                need_header = False
        except Exception:
            need_header = True

        if need_header:
            with open(self._dr_log_path, "w", encoding="utf-8") as f:
                f.write(
                    "episode,timestep,episode_return,adr_scale,body,baseline_mass,sampled_mass,delta_mass\n"
                )
        self._dr_log_header_written = True

    def _log_adr_sample(self, *, body: str, baseline: float, sampled: float) -> None:
        if not self._dr_log_path:
            return
        self._maybe_write_dr_header()
        delta = sampled - baseline
        try:
            with open(self._dr_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{self._episode_idx},{int(self.data.time / self.dt) if self.dt > 0 else 0},"
                    f"{self._episode_return:.6f},{self._adr_scale:.6f},{body},"
                    f"{baseline:.8f},{sampled:.8f},{delta:.8f}\n"
                )
        except Exception:
            # Non bloccare il training per problemi di IO
            pass

    # ---------- shared DR (masses by name) ----------

    def apply_domain_randomization(self):
        """Apply DR once per episode reset.

        Project rule of thumb:
          - no DR: udr.enabled=False and adr.enabled=False
          - UDR:   udr.enabled=True  and adr.enabled=False
          - ADR:   adr.enabled=True  (UDR typically False)

        Priority: ADR overrides UDR (if both True) to avoid ambiguity.
        """
        self._episode_idx += 1
        if self.adr.enabled:
            self.apply_adr()
            return
        if self.udr.enabled:
            self.apply_udr_masses()

    def apply_adr(self):
        """Simple ADR: adapt a single scale based on previous episode return,
        then randomize masses by ±scale around baseline masses.
        """
        # Update scale based on last episode performance
        if self._episode_return >= self.adr.success_threshold:
            self._adr_scale += self.adr.scale_step
        else:
            self._adr_scale = max(
                self.adr.min_scale, self._adr_scale - self.adr.scale_step
            )

        # Cap scale
        self._adr_scale = min(self._adr_scale, self.adr.max_scale)

        bodies = getattr(self, "ADR_TARGET_BODIES", ())
        if not bodies:
            return

        for name in bodies:
            baseline = self._adr_baseline_masses.get(
                name, float(self.model.body_mass[self._body_id(name)])
            )
            lo = (1.0 - self._adr_scale) * baseline
            hi = (1.0 + self._adr_scale) * baseline
            sampled = float(self.np_random.uniform(lo, hi))
            self.model.body_mass[self._body_id(name)] = sampled

            # NEW: log baseline + sampled + delta (for report plots)
            self._log_adr_sample(body=name, baseline=baseline, sampled=sampled)

        if self.adr.debug:
            print(
                f"[ADR] episode_return={self._episode_return:.2f} scale={self._adr_scale:.3f}"
            )

    def apply_udr_masses(self):
        """Randomize masses for the bodies listed in UDR_TARGET_BODIES."""
        if not self.UDR_TARGET_BODIES:
            return

        new_masses: Dict[str, float] = {}
        for name in self.UDR_TARGET_BODIES:
            nominal = self._nominal_body_masses[name]
            lo = (1.0 - self.udr.ratio) * nominal
            hi = (1.0 + self.udr.ratio) * nominal
            new_masses[name] = float(self.np_random.uniform(lo, hi))

        for name, mass in new_masses.items():
            self.model.body_mass[self._body_id(name)] = mass

        if self.udr.debug:
            print("[UDR] masses updated:", new_masses)

    # ---------- shared debug/inspection ----------

    def get_extra_param(self):
        model = self.model
        return {
            "n_bodies": model.nbody,
            "body_names": [model.body(i).name for i in range(model.nbody)],
            "body_masses": model.body_mass.tolist(),
            "n_dofs": model.nv,
            "body_dofs": model.body_dofnum.tolist(),
            "n_actuators": model.nu,
        }

    def get_parameters(self):
        # Compatibility: all masses except the 'world' body
        return np.array(self.model.body_mass[1:], dtype=np.float64)

    def set_parameters(self, task: np.ndarray):
        # Compatibility: assumes the same length as body_mass[1:]
        self.model.body_mass[1:] = task
