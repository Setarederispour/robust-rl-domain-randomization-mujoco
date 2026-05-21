"""Training script for Custom MuJoCo envs (Hopper / Walker2d).

Design goals:
- Minimal changes vs the original lab-4 train.py
- Support the course project extension grid:
    2 envs (hopper, walker2d) × 3 settings (no DR, UDR, ADR)
- Organize outputs neatly:
    models/<env_family>/<domain>/<setting>/...
    logs/<env_family>/<domain>/<setting>/<run_id>/...
- Keep interfaces stable for your refactored env base:
    env kwargs: domain, udr_enabled, udr_ratio, debug_udr, adr_enabled, debug_adr

Usage examples:
  # Hopper, no DR
  python train.py --env_family hopper --domain source --no-udr

  # Hopper, UDR
  python train.py --env_family hopper --domain source --udr --udr_ratio 0.20

  # Walker2d, no DR
  python train.py --env_family walker2d --domain source --no-udr

  # ADR
  python train.py --env_family hopper --domain source --adr
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # workaround for OMP Error #15
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gymnasium as gym

# Import modules for side-effect registration of gym env IDs
import env.custom_hopper  # noqa: F401

try:
    import env.custom_walker2d  # noqa: F401
except Exception:
    env_custom_walker2d = None  # type: ignore
    print("WARNING: Unable to import env.custom_walker2d (Walker2d not available yet).")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def make_single_env(env_id: str, seed: int, env_kwargs: dict | None = None):
    """Create a single env for evaluation/inspection."""
    env_kwargs = env_kwargs or {}
    env = gym.make(env_id, **env_kwargs)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


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
        description="Train PPO on Custom MuJoCo envs (Hopper/Walker2d)."
    )

    # Core experiment switches
    p.add_argument("--env_family", choices=["hopper", "walker2d"], default="hopper")
    p.add_argument("--domain", choices=["source", "target"], default="source")

    # DR switches
    udr_group = p.add_mutually_exclusive_group()
    udr_group.add_argument(
        "--udr", dest="udr_enabled", action="store_true", help="Enable UDR"
    )
    udr_group.add_argument(
        "--no-udr", dest="udr_enabled", action="store_false", help="Disable UDR"
    )
    p.set_defaults(udr_enabled=True)
    p.add_argument("--udr_ratio", type=float, default=0.20)
    p.add_argument(
        "--debug_udr",
        action="store_true",
        help="Print UDR samples (if env supports it)",
    )

    # ADR flag
    p.add_argument(
        "--adr",
        dest="adr_enabled",
        action="store_true",
        help="Enable ADR (requires env support)",
    )
    p.add_argument(
        "--debug_adr", action="store_true", help="Print ADR debug (if implemented)"
    )
    # ADR hyperparameters (simple ADR like last year's "Auto")
    p.add_argument("--adr_initial_scale", type=float, default=0.20)
    p.add_argument("--adr_scale_step", type=float, default=0.05)
    p.add_argument("--adr_success_threshold", type=float, default=800.0)

    # PPO + run params
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    # Eval params
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--deterministic", action="store_true", default=True)

    # Output
    p.add_argument("--run_name", type=str, default="", help="Optional custom run name")
    p.add_argument("--models_dir", type=str, default="models")
    p.add_argument("--logs_dir", type=str, default="logs")

    return p.parse_args()


def format_setting_name(udr_enabled: bool, udr_ratio: float, adr_enabled: bool) -> str:
    """Create a readable setting name for folder structure."""
    # ADR overrides UDR to avoid ambiguous runs
    if adr_enabled:
        return "adr"
    if udr_enabled:
        return f"udr_r{udr_ratio:.2f}"
    return "nodr"


class EpisodeStatsCallback(BaseCallback):
    """
    Collect episode reward/length during training and write:
      - logs_dir/episode_stats.csv
      - logs_dir/training_episode_rewards.png
      - logs_dir/training_episode_lengths.png

    Requires Monitor/VecMonitor so infos contain info["episode"] with keys r, l.
    """

    def __init__(
        self,
        out_dir,
        rolling_window=50,
        max_episodes_plot=3000,
        save_every_episodes=200,
        verbose=0,
    ):
        super().__init__(verbose)
        self.out_dir = Path(out_dir)
        self.rolling_window = int(rolling_window)
        self.max_episodes_plot = int(max_episodes_plot)
        self.save_every_episodes = int(save_every_episodes)

        self.rewards = []
        self.lengths = []
        self.timesteps = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = (info or {}).get("episode", None)
            if ep is None:
                continue
            self.rewards.append(float(ep.get("r", 0.0)))
            self.lengths.append(int(ep.get("l", 0)))
            self.timesteps.append(int(self.num_timesteps))

        if self.save_every_episodes > 0 and len(self.rewards) > 0:
            if len(self.rewards) % self.save_every_episodes == 0:
                self._save()
        return True

    def _on_training_end(self) -> None:
        self._save()

    def _save(self) -> None:
        import matplotlib.pyplot as plt
        import pandas as pd

        self.out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "episode": range(1, len(self.rewards) + 1),
                "timesteps": self.timesteps,
                "reward": self.rewards,
                "length": self.lengths,
            }
        )
        df.to_csv(self.out_dir / "episode_stats.csv", index=False)

        if self.max_episodes_plot > 0:
            df_plot = df[df["episode"] <= self.max_episodes_plot].copy()
        else:
            df_plot = df.copy()
        w = max(1, self.rolling_window)
        df_plot["reward_rm"] = df_plot["reward"].rolling(w, min_periods=1).mean()
        df_plot["length_rm"] = df_plot["length"].rolling(w, min_periods=1).mean()

        plt.figure()
        plt.plot(
            df_plot["episode"],
            df_plot["reward"],
            alpha=0.3,
            label="Episode reward",
        )
        plt.plot(
            df_plot["episode"],
            df_plot["reward_rm"],
            label=f"Rolling mean (w={w})",
        )
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode rewards over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / "training_episode_rewards.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(
            df_plot["episode"],
            df_plot["length"],
            alpha=0.3,
            label="Episode length",
        )
        plt.plot(
            df_plot["episode"],
            df_plot["length_rm"],
            label=f"Rolling mean (w={w})",
        )
        plt.xlabel("Episode")
        plt.ylabel("Episode length")
        plt.title("Episode lengths over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / "training_episode_lengths.png", dpi=200)
        plt.close()


def save_eval_results_csv(
    out_dir: Path,
    *,
    env_family: str,
    train_domain: str,
    setting: str,
    run_id: str,
    seed: int,
    total_timesteps: int,
    n_envs: int,
    n_steps: int,
    learning_rate: float,
    eval_episodes: int,
    deterministic: bool,
    mean_s: float,
    std_s: float,
    mean_t: float,
    std_t: float,
) -> Path:
    """
    Save evaluation results (source and target) in a dedicated CSV inside logs/<...>/<run_id>/.

    Produces: eval_results.csv (one row)
    """
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results.csv"

    df = pd.DataFrame(
        [
            {
                "env_family": env_family,
                "train_domain": train_domain,
                "setting": setting,
                "run_id": run_id,
                "seed": seed,
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
                "n_steps": n_steps,
                "learning_rate": learning_rate,
                "eval_episodes": eval_episodes,
                "deterministic": bool(deterministic),
                "s_to_s_mean": float(mean_s),
                "s_to_s_std": float(std_s),
                "s_to_t_mean": float(mean_t),
                "s_to_t_std": float(std_t),
            }
        ]
    )
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()

    env_id = build_env_id(args.env_family, args.domain)

    # Keep env kwargs consistent across env families
    env_kwargs = {
        "udr_enabled": bool(args.udr_enabled) and (not bool(args.adr_enabled)),
        "udr_ratio": float(args.udr_ratio),
        "debug_udr": bool(args.debug_udr),
        "adr_enabled": bool(args.adr_enabled),
        "debug_adr": bool(args.debug_adr),
        "adr_initial_scale": float(args.adr_initial_scale),
        "adr_scale_step": float(args.adr_scale_step),
        "adr_success_threshold": float(args.adr_success_threshold),
    }

    # Env-specific default: Walker2d benefits from a slightly higher control cost
    # to discourage "pogo"/one-leg hopping solutions.
    if args.env_family == "walker2d":
        env_kwargs["ctrl_cost_weight"] = 1e-2
    else:
        env_kwargs["ctrl_cost_weight"] = 1e-3

    setting = format_setting_name(
        env_kwargs["udr_enabled"], env_kwargs["udr_ratio"], env_kwargs["adr_enabled"]
    )

    # Run id: unique and human-readable
    if args.run_name:
        run_id = args.run_name
    else:
        run_id = (
            f"lr{args.learning_rate}_ns{args.n_steps}_"
            f"nenv{args.n_envs}_seed{args.seed}_ts{args.total_timesteps}"
        )

    # Organized output directories
    models_dir = Path(args.models_dir) / args.env_family / args.domain / setting
    logs_dir = Path(args.logs_dir) / args.env_family / args.domain / setting / run_id
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Model path (SB3 adds .zip automatically)
    model_stem = models_dir / f"ppo_{run_id}"

    # give the env a place to write ADR logs (if ADR enabled)
    # This is just a path; the actual writing happens in CustomMujocoDRBase.
    env_kwargs["dr_log_path"] = str(logs_dir / "dr_stats.csv")

    # Quick inspection env
    env = gym.make(env_id, **env_kwargs)
    print("Env:", env_id)
    print("Setting:", setting)
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    # Optional: print dynamics params if the env exposes it
    if hasattr(env.unwrapped, "get_parameters"):
        try:
            print("Dynamics parameters:", env.unwrapped.get_parameters())
        except Exception:
            print("Dynamics parameters: <unavailable>")
    env.close()

    # TRAIN
    train_env = make_vec_env(
        env_id,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(logs_dir),
        env_kwargs=env_kwargs,
    )

    # Debug prints for DR
    try:
        print("UDR enabled (train env):", train_env.get_attr("udr_enabled")[0])
        print("UDR ratio (train env):", train_env.get_attr("udr_ratio")[0])
        print("ADR enabled (train env):", train_env.get_attr("adr_enabled")[0])
        print("DR log path (train env):", train_env.get_attr("dr_log_path")[0])
    except Exception:
        pass

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        # tensorboard_log=str(logs_dir),  # enable if you want TB
    )

    tic = time.perf_counter()
    cb = EpisodeStatsCallback(
        out_dir=logs_dir,
        rolling_window=50,
        max_episodes_plot=3000,
        save_every_episodes=200,
    )
    model.learn(total_timesteps=args.total_timesteps, callback=cb)
    toc = time.perf_counter()

    model.save(str(model_stem))
    print(f"Saved model: {model_stem}.zip")
    print(f"Logs directory: {logs_dir}")
    print(f"Time required for training: {toc - tic:.2f} s")

    # TEST on both source and target for the same env family
    env_source_id = build_env_id(args.env_family, "source")
    env_target_id = build_env_id(args.env_family, "target")

    # Evaluation should usually be deterministic and without DR.
    # Here we disable DR in eval to compare source vs target fairly.
    eval_env_kwargs = {
        **env_kwargs,
        "udr_enabled": False,
        "adr_enabled": False,
    }

    env_source = make_single_env(
        env_source_id, seed=args.seed + 10, env_kwargs=eval_env_kwargs
    )
    env_target = make_single_env(
        env_target_id, seed=args.seed + 20, env_kwargs=eval_env_kwargs
    )

    mean_s, std_s = evaluate_policy(
        model,
        env_source,
        n_eval_episodes=args.eval_episodes,
        deterministic=args.deterministic,
    )
    mean_t, std_t = evaluate_policy(
        model,
        env_target,
        n_eval_episodes=args.eval_episodes,
        deterministic=args.deterministic,
    )

    print(f"Train domain: {args.domain}")
    print(f"{args.domain} -> source:", mean_s, "+/-", std_s)
    print(f"{args.domain} -> target:", mean_t, "+/-", std_t)

    # NEW: persist evaluation results to a dedicated CSV in this run's logs folder
    eval_csv_path = save_eval_results_csv(
        logs_dir,
        env_family=args.env_family,
        train_domain=args.domain,
        setting=setting,
        run_id=run_id,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        eval_episodes=args.eval_episodes,
        deterministic=args.deterministic,
        mean_s=mean_s,
        std_s=std_s,
        mean_t=mean_t,
        std_t=std_t,
    )
    print(f"Saved eval results: {eval_csv_path}")

    env_source.close()
    env_target.close()
    train_env.close()


if __name__ == "__main__":
    main()
