#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_run_info(csv_path: Path, logs_root: Path):
    """
    Expected layout:
      logs/<env_family>/<domain>/<setting>/<run_id>/episode_stats.csv
    """
    rel = csv_path.relative_to(logs_root)
    parts = rel.parts
    if len(parts) < 5:
        return None

    env_family = parts[0]
    domain = parts[1]
    setting = parts[2]
    run_id = parts[3]
    return env_family, domain, setting, run_id


def summarize_episode_stats(csv_path: Path, last_n: int):
    df = pd.read_csv(csv_path)
    # expected columns: episode,timesteps,reward,length
    if df.empty:
        return None

    # Use ALL episodes by default (last_n == 0).
    df_used = df.tail(last_n) if last_n and last_n > 0 else df

    # Use sample std (ddof=1) when possible; 0.0 if only one episode exists.
    std_reward = float(df_used["reward"].std(ddof=1)) if len(df_used) > 1 else 0.0
    std_length = float(df_used["length"].std(ddof=1)) if len(df_used) > 1 else 0.0

    out = {
        "episodes_total": int(df["episode"].iloc[-1]),
        "timesteps_last": int(df["timesteps"].iloc[-1]),
        "episodes_used": int(len(df_used)),
        "avg_reward": float(df_used["reward"].mean()),
        "std_reward": std_reward,
        "avg_length": float(df_used["length"].mean()),
        "std_length": std_length,
    }
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Summarize training episode_stats.csv under logs/ (global mean/std by default)."
    )
    ap.add_argument(
        "--logs_root",
        type=Path,
        default=Path("logs"),
        help="Root folder that contains the training logs (default: logs).",
    )
    ap.add_argument(
        "--last_n",
        type=int,
        default=0,
        help="Compute mean/std over last N episodes (default: 0 = use all episodes).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("training_summary.csv"),
        help="Output CSV path (default: training_summary.csv).",
    )
    args = ap.parse_args()

    logs_root = args.logs_root.resolve()
    if not logs_root.exists():
        raise SystemExit(f"logs_root not found: {logs_root}")

    rows = []
    for csv_path in logs_root.rglob("episode_stats.csv"):
        info = parse_run_info(csv_path, logs_root)
        if info is None:
            continue
        env_family, domain, setting, run_id = info

        stats = summarize_episode_stats(csv_path, args.last_n)
        if stats is None:
            continue

        rows.append(
            {
                "env_family": env_family,
                "domain": domain,
                "setting": setting,
                "run_id": run_id,
                "csv_path": str(csv_path),
                **stats,
            }
        )

    if not rows:
        raise SystemExit(f"No episode_stats.csv found under: {logs_root}")

    out_df = pd.DataFrame(rows).sort_values(
        by=["env_family", "domain", "setting", "run_id"]
    )

    out_df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}  ({len(out_df)} runs)")
    if args.last_n == 0:
        print("Stats computed over ALL episodes (global mean/std).")
    else:
        print(f"Stats computed over last_n={args.last_n} episodes.")


if __name__ == "__main__":
    main()
