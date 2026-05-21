#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="logs"
DST_DIR="images"

mkdir -p "$DST_DIR"

# ---------- helpers ----------

budget_from_runid() {
  # input: run_id like lr0.0003_ns2048_nenv8_seed0_ts3000000
  local run_id="$1"
  if [[ "$run_id" =~ ts([0-9]+) ]]; then
    local ts="${BASH_REMATCH[1]}"
    if (( ts % 1000000 == 0 )); then
      echo "$((ts / 1000000))M"
    else
      # fallback
      echo "${ts}"
    fi
  else
    echo "unknown"
  fi
}

should_add_budget_suffix() {
  # we follow your screenshot:
  # - walker2d has 1M and 3M -> add suffix
  # - hopper has only 1M -> no suffix
  local env_family="$1"
  if [[ "$env_family" == "walker2d" ]]; then
    return 0
  fi
  return 1
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "[OK] $(basename "$dst")"
  fi
}

# ---------- 1) NoDR + UDR: training plots at run root ----------

# We want:
# training_episode_rewards_<env>_<nodr|udr>[_<budget>].png
# training_episode_lengths_<env>_<nodr|udr>[_<budget>].png
#
# from:
# logs/<env>/<domain>/<setting>/<run_id>/training_episode_rewards.png
# logs/<env>/<domain>/<setting>/<run_id>/training_episode_lengths.png

find "$SRC_ROOT" -type f \
  \( -path "*/nodr/*/training_episode_rewards.png" -o -path "*/nodr/*/training_episode_lengths.png" \
     -o -path "*/udr_r*/ */training_episode_rewards.png" -o -path "*/udr_r*/*/training_episode_lengths.png" \) \
  2>/dev/null | while read -r f; do

  # Example:
  # logs/walker2d/source/udr_r0.20/<run_id>/training_episode_rewards.png
  env_family=$(echo "$f" | cut -d'/' -f2)
  domain=$(echo "$f" | cut -d'/' -f3)     # not used in name for now
  setting_dir=$(echo "$f" | cut -d'/' -f4)
  run_id=$(echo "$f" | cut -d'/' -f5)

  # setting short name
  if [[ "$setting_dir" == "nodr" ]]; then
    setting="nodr"
  else
    setting="udr"
  fi

  budget="$(budget_from_runid "$run_id")"
  suffix=""
  if should_add_budget_suffix "$env_family"; then
    suffix="_${budget}"
  fi

  base="$(basename "$f")"
  if [[ "$base" == "training_episode_rewards.png" ]]; then
    out="training_episode_rewards_${env_family}_${setting}${suffix}.png"
  else
    out="training_episode_lengths_${env_family}_${setting}${suffix}.png"
  fi

  cp "$f" "$DST_DIR/$out"
  echo "[OK] $out"
done

# ---------- 2) ADR: figures/*.png ----------

# We want only ADR-related plots from:
# logs/<env>/<domain>/adr/<run_id>/figures/*.png
# and rename them cleanly:
# adr_scale_<env>_<budget>.png
# adr_mass_delta_<env>_<budget>.png
# adr_training_reward_<env>_<budget>.png
# adr_training_length_<env>_<budget>.png
# adr_scale_vs_return_<env>_<budget>.png

find "$SRC_ROOT" -type f -path "*/adr/*/figures/*.png" 2>/dev/null | while read -r f; do
  env_family=$(echo "$f" | cut -d'/' -f2)
  domain=$(echo "$f" | cut -d'/' -f3)     # not used
  run_id=$(echo "$f" | cut -d'/' -f5)
  budget="$(budget_from_runid "$run_id")"

  stem="$(basename "$f" .png)"  # e.g. adr_scale, adr_mass_delta, ...
  out="${stem}_${env_family}_${budget}.png"

  cp "$f" "$DST_DIR/$out"
  echo "[OK] $out"
done

echo
echo "Done. Collected images in: ./$DST_DIR/"
