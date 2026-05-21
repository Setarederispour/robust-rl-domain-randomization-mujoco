"""Test a trained policy on Custom MuJoCo envs (Hopper / Walker2d).

What this script does:
- Loads a Stable-Baselines3 PPO model (saved by train.py).
- Runs one episode on source and one on target.
- Optionally renders (human window) OR runs headless (no rendering).

Notes:
- Rendering may fail on some Wayland/NixOS/conda setups due to OpenGL context issues.
  Use --render false for a reliable numeric test.

Usage examples:
  # Headless evaluation (recommended if rendering is problematic)
  python test.py --env_family hopper --model models/hopper/source/udr_r0.20/ppo_lr3e-04_ns2048_nenv8_seed0_ts1000000.zip

  # Render (may require X11 / proper GL setup)
  python test.py --env_family hopper --model models/hopper/source/udr_r0.20/ppo_lr3e-04_ns2048_nenv8_seed0_ts1000000.zip --render

  # Specify which domains to run
  python test.py --env_family walker2d --model <path> --domains source target
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import gymnasium as gym
from stable_baselines3 import PPO

# Import modules for side-effect registration of gym env IDs
import env.custom_hopper  # noqa: F401

try:
    import env.custom_walker2d  # noqa: F401
except Exception:
    print("WARNING: Unable to import env.custom_walker2d (Walker2d not available yet).")


def build_env_id(env_family: str, domain: str) -> str:
    """Map (family, domain) -> gym registered ID."""
    family_to_prefix = {
        "hopper": "CustomHopper",
        "walker2d": "CustomWalker2d",
    }
    if env_family not in family_to_prefix:
        raise ValueError(
            f"Unknown env_family={env_family}. Choose from: {list(family_to_prefix)}"
        )
    if domain not in {"source", "target"}:
        raise ValueError("domain must be 'source' or 'target'")
    return f"{family_to_prefix[env_family]}-{domain}-v0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test a trained PPO policy (Hopper/Walker2d)."
    )

    p.add_argument("--env_family", choices=["hopper", "walker2d"], default="hopper")
    p.add_argument(
        "--domains",
        nargs="+",
        choices=["source", "target"],
        default=["source", "target"],
        help="Domains to test (default: both).",
    )

    # Model path: accept both with/without .zip
    p.add_argument(
        "--model", required=True, help="Path to the saved SB3 model (.zip or stem)."
    )

    # Rendering
    p.add_argument(
        "--render",
        action="store_true",
        help="Render with a human window (may fail on some setups).",
    )

    # Determinism / seed
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", default=True)

    # Max steps safety (optional)
    p.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Optional hard cap on steps (0 = no cap).",
    )

    return p.parse_args()


def normalize_model_path(model_arg: str) -> str:
    """Normalize SB3 model path.

    SB3 accepts both:
      - 'path/to/model' (will append .zip internally)
      - 'path/to/model.zip'

    But avoid the common mistake: '...zip.zip'.
    """
    p = Path(model_arg)
    if p.suffix == ".zip":
        return str(p)
    # If user passed something like '..._udr.zip.zip', strip extra
    if model_arg.endswith(".zip.zip"):
        return model_arg[:-4]
    return model_arg


def run_episode(
    env_id: str,
    model: PPO,
    render: bool,
    deterministic: bool,
    seed: int,
    max_steps: int = 0,
):
    """Run a single episode and print return/length."""
    render_mode = "human" if render else None

    # Important: pass render_mode only when needed
    if render_mode is None:
        env = gym.make(env_id)
    else:
        env = gym.make(env_id, render_mode=render_mode)

    try:
        obs, _info = env.reset(seed=seed)
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)

            ep_return += float(reward)
            ep_len += 1

            if max_steps > 0 and ep_len >= max_steps:
                break

        print(f"{env_id}: return={ep_return:.2f}, len={ep_len}")

    finally:
        env.close()


def main() -> None:
    args = parse_args()

    model_path = normalize_model_path(args.model)
    model = PPO.load(model_path)

    env = gym.make("CustomWalker2d-source-v0")
    print(env.unwrapped.get_extra_param()["body_names"])

    for i, domain in enumerate(args.domains):
        env_id = build_env_id(args.env_family, domain)
        # Make domains use different seeds (nice for quick sanity checks)
        seed = args.seed + 10 * i
        run_episode(
            env_id,
            model=model,
            render=args.render,
            deterministic=args.deterministic,
            seed=seed,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
