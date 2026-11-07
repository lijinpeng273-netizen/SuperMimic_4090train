#!/usr/bin/env bash
# Use this project's environment with the workstation's CPU restrictions.
set -euo pipefail

if (( $# == 0 )); then
  printf 'Usage: %s <command> [args...]\n' "$0" >&2
  exit 2
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
mkdir -p "$project_dir/.cache/tmp"

exec taskset -c 16-23 env -u PYTHONPATH -u PYTHONHOME \
  VIRTUAL_ENV="$project_dir/.venv" \
  PYTHONDONTWRITEBYTECODE=1 \
  TMPDIR="$project_dir/.cache/tmp" \
  XDG_CACHE_HOME="$project_dir/.cache" \
  WARP_CACHE_PATH="$project_dir/.cache/warp" \
  TORCHINDUCTOR_CACHE_DIR="$project_dir/.cache/torchinductor" \
  TRITON_CACHE_DIR="$project_dir/.cache/triton" \
  CUDA_CACHE_PATH="$project_dir/.cache/cuda" \
  MPLCONFIGDIR="$project_dir/.cache/matplotlib" \
  PATH="$project_dir/.venv/bin:$PATH" \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  TORCHINDUCTOR_COMPILE_THREADS=4 \
  UV_CONCURRENT_DOWNLOADS=4 \
  UV_CONCURRENT_INSTALLS=4 \
  UV_CONCURRENT_BUILDS=1 \
  CMAKE_BUILD_PARALLEL_LEVEL=4 \
  MAX_JOBS=4 \
  "$@"
