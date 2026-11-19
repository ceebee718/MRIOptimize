#!/usr/bin/env bash
# Start the KomaMRI worker (port 8011, internal) and the API (port 8000, exposed).
# Optional: export PROAQMR_API_KEY=some-secret   (the page must send the same key)
#           export ALLOWED_ORIGINS=https://your-site.netlify.app
set -uo pipefail
cd "$(dirname "$0")"
WS=${WS:-/workspace}
export PATH="$WS/julia/bin:$PATH"
export JULIA_DEPOT_PATH="$WS/.julia"
export PROAQMR_WORK="$WS/proaqmr_tmp"; mkdir -p "$PROAQMR_WORK"
export PROAQMR_WARMUP="$PROAQMR_WORK/warm"
export KOMA_PORT=${KOMA_PORT:-8011}   # RunPod's own proxy uses 8001 inside the pod

# stop any worker or API left from an earlier start (patterns match only this server)
pkill -f "server/koma_worker.jl" 2>/dev/null || true
pkill -f "uvicorn server.app:app" 2>/dev/null || true
sleep 1

if command -v julia >/dev/null 2>&1; then
  python3 -m server.warmup "$PROAQMR_WARMUP" || echo "warm-up files not written"
  echo "Starting KomaMRI worker (log: $WS/koma.log)"
  nohup julia --project=server server/koma_worker.jl > "$WS/koma.log" 2>&1 &
else
  echo "Julia not found: only the fast engine will be available. Run setup.sh for KomaMRI."
fi

echo "Starting API on port 8000 (log: $WS/api.log)"
exec python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8000 2>&1 | tee -a "$WS/api.log"
