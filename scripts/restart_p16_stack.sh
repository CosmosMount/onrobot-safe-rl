#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
log_dir="${1:-$repo_root/saved/experiments/p16_stack}"
mkdir -p "$log_dir"
log_dir="$(readlink -m "$log_dir")"

pkill -f 'python -m runtime.inference.runtime' 2>/dev/null || true
pkill -f 'runtime/control/go2/build/go2_control' 2>/dev/null || true
pkill -f './unitree_mujoco -r go2 -s scene_empty.xml -i 1 -n lo' 2>/dev/null || true
sleep 2

(
  cd /home/xyz/code/unitree_mujoco/simulate/build
  nohup ./unitree_mujoco -r go2 -s scene_empty.xml -i 1 -n lo \
    >"$log_dir/mujoco.log" 2>&1 &
)
sleep 3

(
  cd "$repo_root"
  nohup runtime/control/go2/build/go2_control runtime/control/go2/go2.yaml \
    >"$log_dir/controller.log" 2>&1 &
)
sleep 3

(
  cd "$repo_root"
  nohup micromamba run -n oss python -m runtime.inference.runtime \
    --config config/go2_50hz_safe.yaml \
    >"$log_dir/runtime.log" 2>&1 &
)
sleep 5

pgrep -af './unitree_mujoco -r go2 -s scene_empty.xml -i 1 -n lo'
pgrep -af 'runtime/control/go2/build/go2_control'
pgrep -af 'python -m runtime.inference.runtime'
