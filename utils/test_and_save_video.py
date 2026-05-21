"""
Record a trained policy on Custom MuJoCo envs (Hopper / Walker2d) to MP4.

Typical usage (from project root):
  python test_and_save_video.py --env_family hopper --model ppo_hopper_source_lr0.0003_ns2048_udr --domain source
  python test_and_save_video.py --env_family hopper --model ppo_hopper_source_lr0.0003_ns2048_udr --domain target

Notes:
- Uses render_mode="rgb_array" to avoid GLFW/Wayland window issues.
- For headless setups you may need: MUJOCO_GL=osmesa (or egl if supported).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

# Import modules for side-effect registration of gym env IDs
import env.custom_hopper  # noqa: F401
import env.custom_walker2d  # noqa: F401


def build_env_id(env_family: str, domain: str) -> str:
    """Map (family, domain) -> gym registered ID."""
    family_to_prefix = {
        "hopper": "CustomHopper",
        "walker2d": "CustomWalker2d",
    }
    if env_family not in family_to_prefix:
        raise ValueError(
            f"Unknown env_family={env_family}. Choose from {list(family_to_prefix)}"
        )
    if domain not in {"source", "target"}:
        raise ValueError("domain must be 'source' or 'target'")
    return f"{family_to_prefix[env_family]}-{domain}-v0"


def normalize_model_path(model_path: str) -> str:
    """Accept both 'name' and 'name.zip' and return a path usable by SB3."""
    p = Path(model_path)
    if p.suffix == ".zip":
        return str(p)
    # If user passed a path without extension, SB3 usually appends '.zip'.
    # We keep it extension-less to match SB3 behavior and avoid '.zip.zip' mistakes.
    return str(p)


def run_episode_to_video(
    env_id: str,
    model_path: str,
    out_video: str,
    env_kwargs: dict,
    deterministic: bool = True,
    seed: int = 0,
    fps: int = 30,
    max_steps: int | None = None,
) -> None:
    """Run one episode and write frames directly to an MP4 file."""
    # Load model first (fail fast if path is wrong)
    model = PPO.load(normalize_model_path(model_path))

    env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    obs, info = env.reset(seed=seed)

    out_path = Path(out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ep_return = 0.0
    ep_len = 0

    # Write frames incrementally (no huge RAM usage)
    with imageio.get_writer(out_video, fps=fps) as writer:
        done = False
        while not done:
            frame = env.render()
            if frame is None:
                raise RuntimeError("env.render() returned None (expected rgb_array).")
            writer.append_data(np.asarray(frame))

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            ep_return += float(reward)
            ep_len += 1

            if max_steps is not None and ep_len >= max_steps:
                break

        # Final frame (optional but nice)
        frame = env.render()
        if frame is not None:
            writer.append_data(np.asarray(frame))

    env.close()
    print(f"{env_id}: return={ep_return:.2f}, len={ep_len}, saved={out_video}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record a trained PPO policy to MP4 (Hopper/Walker2d)."
    )

    p.add_argument("--env_family", choices=["hopper", "walker2d"], default="hopper")
    p.add_argument("--domain", choices=["source", "target"], default="source")

    p.add_argument("--model", required=True, help="Model path (with or without .zip).")
    p.add_argument("--out", default="", help="Output MP4 path (optional).")

    # DR switches for the evaluation env (optional)
    udr_group = p.add_mutually_exclusive_group()
    udr_group.add_argument(
        "--udr", dest="udr_enabled", action="store_true", help="Enable UDR in eval env"
    )
    udr_group.add_argument(
        "--no-udr",
        dest="udr_enabled",
        action="store_false",
        help="Disable UDR in eval env",
    )
    p.set_defaults(udr_enabled=False)  # default: evaluate without DR
    p.add_argument("--udr_ratio", type=float, default=0.20)
    p.add_argument(
        "--debug-udr",
        action="store_true",
        help="Print sampled UDR masses (if env supports it)",
    )

    # ADR placeholder (will raise if enabled in env base)
    p.add_argument(
        "--adr",
        action="store_true",
        help="Enable ADR in eval env (requires env support)",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--max_steps", type=int, default=0, help="Optional cap; 0 means no cap."
    )
    det_group = p.add_mutually_exclusive_group()
    det_group.add_argument("--deterministic", dest="deterministic", action="store_true")
    det_group.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.set_defaults(deterministic=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.adr:
        raise SystemExit("ADR requested, but ADR is not implemented yet. Remove --adr.")

    env_id = build_env_id(args.env_family, args.domain)

    # Keep kwargs aligned with the env base interface
    env_kwargs = {
        "udr_enabled": args.udr_enabled,
        "udr_ratio": args.udr_ratio,
        "debug_udr": args.debug_udr,
        "adr_enabled": bool(args.adr),
    }

    # Auto-name output if not provided
    if args.out:
        out_video = args.out
    else:
        out_dir = Path("videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = str(out_dir / f"{args.env_family}_{args.domain}.mp4")

    max_steps = args.max_steps if args.max_steps > 0 else None

    run_episode_to_video(
        env_id=env_id,
        model_path=args.model,
        out_video=out_video,
        env_kwargs=env_kwargs,
        deterministic=args.deterministic,
        seed=args.seed,
        fps=args.fps,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    main()
