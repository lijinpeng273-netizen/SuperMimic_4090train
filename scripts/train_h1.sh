#!/usr/bin/env bash
# RTX 4090 profile, measured with CPU cores 16-23 and four numerical-library threads.
# Keep k=8, H=32, original network widths and the default learning rates.
# Compile the world-model loss and imagined step; the first update may compile for minutes.
# Use the original 10 world-model batches per update to fit dynamics before policy updates.
# local.sh selects this project's environment and restricts CPU cores/threads.
# Train on every NPZ currently in data/npz/h1, including newly converted motions.
# Disable clip/manifest/limit filters and packed-library overrides by default.
# Feed back reference-relative pose and penalise global drift; retain the arm fix.
# Legacy 480-d checkpoints are expanded to 489 inputs automatically on resume.
# Arguments supplied by the caller override the defaults below.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$project_dir/scripts/local.sh" env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  "$project_dir/.venv/bin/python" -B -m supermimic.scripts.train \
  --run-name h1_all_motions --data.npz-dir data/npz/h1 \
  --data.clips None --data.limit None --data.manifest None --data.library None \
  --env.num-envs 4096 --env.replay-capacity 256 --env.anchor-obs \
  --wm-trainer.batch-size 4096 --wm-trainer.batches-per-update 10 --wm-trainer.window 8 \
  --policy-trainer.batch-size 4096 --policy-trainer.horizon 32 --policy-trainer.lr 0.0002 \
  --wm-trainer.compile default --policy-trainer.compile default \
  --rewards.w-arm-joint-pos 1.0 --rewards.std-arm-joint-pos 0.5 \
  --rewards.w-arm-target-limit -0.2 \
  --rewards.w-planar-drift -0.5 --rewards.planar-drift-scale 1.0 --rewards.w-heading 0.5 \
  "$@"
