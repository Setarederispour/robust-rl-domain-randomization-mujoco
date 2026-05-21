#!/usr/bin/env python3
"""
Generate organized ADR plots from logs.

Expected layout:
  logs/<env_family>/<domain>/adr/<run_id>/episode_stats.csv
  logs/<env_family>/<domain>/adr/<run_id>/dr_stats.csv

dr_stats.csv expected columns:
  episode,timestep,episode_return,adr_scale,body,baseline_mass,sampled_mass,delta_mass

Outputs:
  logs/<...>/<run_id>/figures/
    adr_training_reward.png
    adr_training_length.png
    adr_scale.png
    adr_scale_vs_return.png
    adr_mass_delta.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def rolling_mean(s: pd.Series, w: int) -> pd.Series:
    w = max(1, int(w))
    return s.rolling(w, min_periods=1).mean()


def plot_episode_stats(run_dir: Path, rolling_window: int, max_episodes: int) -> None:
    csv_path = run_dir / "episode_stats.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    if max_episodes > 0:
        df = df[df["episode"] <= max_episodes].copy()

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Reward plot
    df["reward_rm"] = rolling_mean(df["reward"], rolling_window)
    plt.figure()
    plt.plot(df["episode"], df["reward"], alpha=0.25, label="Episode reward")
    plt.plot(df["episode"], df["reward_rm"], label=f"Rolling mean (w={rolling_window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("ADR training rewards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "adr_training_reward.png", dpi=200)
    plt.close()

    # Length plot
    df["length_rm"] = rolling_mean(df["length"], rolling_window)
    plt.figure()
    plt.plot(df["episode"], df["length"], alpha=0.25, label="Episode length")
    plt.plot(df["episode"], df["length_rm"], label=f"Rolling mean (w={rolling_window})")
    plt.xlabel("Episode")
    plt.ylabel("Episode length")
    plt.title("ADR training episode lengths")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "adr_training_length.png", dpi=200)
    plt.close()


def plot_adr_stats(run_dir: Path, rolling_window: int) -> None:
    adr_csv = run_dir / "dr_stats.csv"
    if not adr_csv.exists():
        return

    df = pd.read_csv(adr_csv)
    if df.empty:
        return

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- ADR scale over time ---
    plt.figure()
    plt.plot(df["episode"], df["adr_scale"], label="ADR scale")
    plt.plot(
        df["episode"],
        rolling_mean(df["adr_scale"], rolling_window),
        label=f"Rolling mean (w={rolling_window})",
    )
    plt.xlabel("Episode")
    plt.ylabel("ADR scale")
    plt.title("ADR scale over training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "adr_scale.png", dpi=200)
    plt.close()

    # --- ADR scale vs episode return ---
    plt.figure()
    plt.scatter(df["adr_scale"], df["episode_return"], s=8, alpha=0.6)
    plt.xlabel("ADR scale")
    plt.ylabel("Episode return")
    plt.title("ADR scale vs episode return")
    plt.tight_layout()
    plt.savefig(fig_dir / "adr_scale_vs_return.png", dpi=200)
    plt.close()

    # --- Mass delta plot (one figure, all bodies) ---
    plt.figure()
    for body, g in df.groupby("body"):
        plt.plot(g["episode"], g["delta_mass"], alpha=0.7, label=body)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Mass delta (kg)")
    plt.title("ADR mass perturbations (baseline + delta)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "adr_mass_delta.png", dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", type=str, default="logs")
    ap.add_argument("--rolling_window", type=int, default=50)
    ap.add_argument("--max_episodes", type=int, default=3000)
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        raise SystemExit(f"logs_dir not found: {logs_dir}")

    episode_csvs = sorted(logs_dir.glob("**/adr/**/episode_stats.csv"))
    if not episode_csvs:
        print("No ADR runs found (no episode_stats.csv under **/adr/**).")
        return

    n = 0
    for ep_csv in episode_csvs:
        run_dir = ep_csv.parent
        plot_episode_stats(
            run_dir, rolling_window=args.rolling_window, max_episodes=args.max_episodes
        )
        plot_adr_stats(run_dir, rolling_window=args.rolling_window)
        n += 1
        print(f"[OK] Plots generated in: {run_dir / 'figures'}")

    print(f"Done. Processed {n} ADR runs.")


if __name__ == "__main__":
    main()
