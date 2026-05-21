#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def setting_label(setting: str) -> str:
    if setting == "nodr":
        return "No DR"
    if setting.startswith("udr_r"):
        return "UDR"
    if setting == "adr":
        return "ADR"
    return setting


def fmt(x: float | int | None, nd: int = 2) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def load_all_eval(logs_root: Path) -> pd.DataFrame:
    rows = []
    for p in logs_root.rglob("eval_results.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        r = df.iloc[0].to_dict()
        r["eval_path"] = str(p)
        rows.append(r)
    if not rows:
        raise SystemExit(f"No eval_results.csv found under {logs_root}")
    out = pd.DataFrame(rows)
    return out


def pick_one(df: pd.DataFrame, env_family: str, total_timesteps: int, setting_key: str) -> dict | None:
    sub = df[
        (df["env_family"] == env_family)
        & (df["total_timesteps"] == total_timesteps)
        & (df["setting"] == setting_key)
    ]
    if sub.empty:
        return None
    # If multiple runs match (e.g., different run_id), pick the first deterministically
    sub = sub.sort_values(by=["run_id"])
    return sub.iloc[0].to_dict()


def make_table(env_family: str, total_timesteps: int, df: pd.DataFrame, label: str) -> str:
    # expected settings folders: nodr, udr_r0.xx, adr
    # we’ll try to find one udr setting (the one present most frequently), else fall back.
    udr_candidates = sorted({s for s in df["setting"].unique() if str(s).startswith("udr_r")})
    udr_key = udr_candidates[0] if udr_candidates else "udr_r0.20"

    order = [("nodr", "No DR"), (udr_key, "UDR"), ("adr", "ADR")]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{{env_family.capitalize()} (training budget: {total_timesteps//1_000_000}M timesteps). "
        r"Policies are trained on the source domain and evaluated on both source (S$\rightarrow$S) and target (S$\rightarrow$T).}}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{DR Strategy} & \textbf{Domain} & \textbf{Avg. Reward} & \textbf{Std. Dev.} \\")
    lines.append(r"\hline")

    for key, pretty in order:
        row = pick_one(df, env_family, total_timesteps, key)
        if row is None:
            # keep empty placeholders so the table structure is stable
            lines.append(rf"{pretty} & S$\rightarrow$S &  &  \\")
            lines.append(rf"{pretty} & S$\rightarrow$T &  &  \\")
            continue

        lines.append(
            rf"{pretty} & S$\rightarrow$S & {fmt(row.get('s_to_s_mean'))} & {fmt(row.get('s_to_s_std'))} \\"
        )
        lines.append(
            rf"{pretty} & S$\rightarrow$T & {fmt(row.get('s_to_t_mean'))} & {fmt(row.get('s_to_t_std'))} \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", type=Path, default=Path("logs"))
    ap.add_argument("--copy", action="store_true", help="Copy output to clipboard via pbcopy (macOS).")
    args = ap.parse_args()

    logs_root = args.logs_root.resolve()
    df = load_all_eval(logs_root)

    # Ensure numeric type (CSV might read as int/float already, but be safe)
    df["total_timesteps"] = pd.to_numeric(df["total_timesteps"], errors="coerce").astype("Int64")

    # Build the three tables you asked for:
    tex_parts = []
    tex_parts.append(make_table("hopper", 1_000_000, df, label="tab:hopper_1m_eval"))
    tex_parts.append("")
    tex_parts.append(make_table("walker2d", 1_000_000, df, label="tab:walker2d_1m_eval"))
    tex_parts.append("")
    tex_parts.append(make_table("walker2d", 3_000_000, df, label="tab:walker2d_3m_eval"))

    out_tex = "\n".join(tex_parts)

    print(out_tex)

    if args.copy:
        import subprocess
        subprocess.run(["pbcopy"], input=out_tex.encode("utf-8"), check=True)
        print("\n[Copied LaTeX to clipboard with pbcopy]")


if __name__ == "__main__":
    main()
