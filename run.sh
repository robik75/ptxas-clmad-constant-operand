#!/usr/bin/env bash
# Builds repro.cu for sm_120 at each ptxas optimisation level, and once as PTX only so the driver
# JIT-compiles it, keeps the PTX and each SASS build's listing under out/, and runs every build.
# Each run exits 0 when every step matches the host, 1 when one doesn't and 2 on a CUDA error.
# NVCC overrides the compiler; extra arguments go to every nvcc call (e.g. -DN=18).
set -u
cd "$(dirname "$0")"
nvcc=${NVCC:-nvcc}
bin=$(dirname "$(command -v "$nvcc")")
mkdir -p out

"$nvcc" --version | grep release | sed "s/^/nvcc: /"
"$bin/cuobjdump" --version | grep release | sed "s/^/cuobjdump: /"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

"$nvcc" -arch=sm_120 -ptx "$@" repro.cu -o out/repro.ptx || exit 2
for o in 3 2 1 0; do
  "$nvcc" -arch=sm_120 -Xptxas -O$o "$@" repro.cu -o out/repro_O$o || exit 2
  "$bin/cuobjdump" -sass out/repro_O$o > out/repro_O$o.sass
  echo "== ptxas -O$o"
  ./out/repro_O$o
  echo "exit $?"
done
"$nvcc" -gencode arch=compute_120,code=compute_120 "$@" repro.cu -o out/repro_jit || exit 2
echo "== PTX only, JIT-compiled by the driver"
CUDA_CACHE_DISABLE=1 ./out/repro_jit
echo "exit $?"
