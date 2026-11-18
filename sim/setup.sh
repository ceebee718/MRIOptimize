#!/usr/bin/env bash
# One-time setup on a RunPod GPU pod. Everything goes into /workspace so it
# survives pod restarts. Run from the repo folder:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"
WS=${WS:-/workspace}
JULIA_VERSION=${JULIA_VERSION:-1.11.5}
mkdir -p "$WS"
if [ ! -d "$WS" ] || [ "$(df -P "$WS" | tail -1 | awk '{print $6}')" = "/" ]; then
  echo "Note: $WS is on the container disk. It is erased when the pod stops, so setup will need to run again."
fi

echo "== Python packages"
python3 -m pip install -q -r requirements.txt 2>/dev/null \
  || python3 -m pip install -q --break-system-packages -r requirements.txt

echo "== Julia $JULIA_VERSION"
if [ ! -x "$WS/julia/bin/julia" ]; then
  MINOR=${JULIA_VERSION%.*}
  curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/${MINOR}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" -o /tmp/julia.tgz
  mkdir -p "$WS/julia" && tar -xzf /tmp/julia.tgz -C "$WS/julia" --strip-components=1 --no-same-owner
fi
export PATH="$WS/julia/bin:$PATH"
export JULIA_DEPOT_PATH="$WS/.julia"

echo "== KomaMRI and server packages (first time: 10-20 minutes)"
julia --project=server -e '
  using Pkg
  Pkg.add(["KomaMRIBase", "KomaMRICore", "KomaMRIFiles", "CUDA", "HTTP", "JSON3"])
  Pkg.precompile()
  using CUDA
  println("CUDA functional: ", CUDA.functional())
'
echo "== Setup complete. Start the server with: bash start.sh"
