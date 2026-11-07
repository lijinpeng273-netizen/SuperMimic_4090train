#!/bin/bash
# Fetch H1 checkpoints from the explicitly supplied host and evaluate local H1 clips.
# Usage: scripts/pull_and_eval.sh <remote_host> <run_name> [num_envs] [npz_dir]
set -euo pipefail
host=${1:?remote_host required}
run=${2:?run_name required}
nenv=${3:-32}
npz_dir=${4:-data/npz/h1}
cd "$(dirname "$0")/.."
mkdir -p "logs/$run"
shopt -s nullglob
while true; do
  rsync -a --include='ckpt_0*.pt' --include='config.json' --exclude='*' "$host:~/SuperMimic/logs/$run/" "logs/$run/"
  for ck in "logs/$run"/ckpt_0*.pt; do
    stem=$(basename "$ck" .pt)
    out="logs/$run/benchmark_$stem.json"
    [ -f "$out" ] && continue
    env -u PYTHONPATH -u PYTHONHOME .venv/bin/python scripts/eval_benchmark.py \
      --ckpt "$ck" --npz-dir "$npz_dir" --num-envs "$nenv" --out "$out" \
      > "logs/$run/eval_$stem.log" 2>&1
  done
  sleep 180
done

# Original G1/BONES-SEED helper retained below:
# #!/bin/bash
# # Pull new checkpoints of a remote run and evaluate them locally on the BONES-SEED 2000-clip subset.
# #   scripts/pull_and_eval.sh <remote_host> <run_name> [num_envs]
# host=$1; run=$2; nenv=${3:-1024}
# cd "$(dirname "$0")/.." || exit 1
# mkdir -p logs/$run
# while true; do
#   rsync -a --include='ckpt_0*.pt' --include='config.json' --include='eval_summary.txt' --exclude='*' $host:~/SuperMimic/logs/$run/ logs/$run/ 2>/dev/null
#   for ck in $(ls logs/$run/ckpt_0*.pt 2>/dev/null | sort); do
#     out=logs/$run/benchmark_$(basename $ck .pt).json
#     [ -f "$out" ] && continue
#     env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt $ck --npz-dir data/npz/bones_seed --clip-list data/npz/bones_seed/subset2000.txt --num-envs $nenv > logs/$run/eval_$(basename $ck .pt).log 2>&1
#     echo "$(date +%H:%M) $(basename $ck): $(grep -E 'clips \|' logs/$run/eval_$(basename $ck .pt).log)" >> logs/$run/eval_summary_local.txt
#   done
#   sleep 180
# done
