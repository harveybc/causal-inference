#!/bin/bash
# FS-CAUSAL durable driver for worker_b (run as a systemd --user unit; see fs_causal_unit.sh).
# Stage 1: the three-rung ladder over every chunk (resumable), then finalize.
# Stage 2: PCMCI+ discovery comparator over every chunk (resumable), then re-finalize to merge it.
# Every heavy step runs under crispdm-run with the cap passed in FS_CAUSAL_CAP (1.25x the measured pilot peak).
# Never touches GPU jobs, MT5/lts services or /tmp; writes only under $ROOT.
set -uo pipefail

ROOT="${FS_CAUSAL_ROOT:-$HOME/.local/state/canonical_20261003/fs_causal}"
RUN="$ROOT/run"
PY="${FS_CAUSAL_PYTHON:-$HOME/.local/state/envs/fs_causal/bin/python}"
CAP="${FS_CAUSAL_CAP:-2G}"
WALL="${FS_CAUSAL_WALL:-6h}"
PEAK_EV="${FS_CAUSAL_PEAK_EVIDENCE:-}"
PEAK_BYTES="${FS_CAUSAL_PEAK_BYTES:-}"
LOG="$ROOT/driver.log"
export PYTHONPATH="$ROOT/code/provider/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
MAXFAIL=6

run_capped() {  # name, command...
  local name=$1; shift
  local extra=()
  if [ -n "$PEAK_EV" ] && [ -n "$PEAK_BYTES" ]; then extra=(-E "$PEAK_EV" -P "$PEAK_BYTES"); fi
  "$HOME/.local/bin/crispdm-run" -m "$CAP" -t "$WALL" -n "$name" -q -W 7200 "${extra[@]}" -- "$@"
}

log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

log "DRIVER START cap=$CAP wall=$WALL"
fails=0
# ---- stage 1: ladder
until [ -f "$RUN/READY" ]; do
  if [ -f "$RUN/DRIVER_FAILED" ]; then log "DRIVER_FAILED marker present; stopping"; exit 1; fi
  if "$PY" - "$RUN" <<'PY'
import json, os, sys
out = sys.argv[1]
plan = json.load(open(os.path.join(out, "plan.json")))
sys.exit(0 if all(os.path.exists(os.path.join(out, c["id"], "READY")) for c in plan["chunks"]) else 1)
PY
  then
    log "all chunks READY; finalize"
    if run_capped fs-causal-finalize "$PY" -m causal_inference_provider.fs_causal_batch finalize --out "$RUN" >> "$LOG" 2>&1; then
      log "FINALIZED"; fails=0
    else
      fails=$((fails + 1)); log "finalize failed rc=$? ($fails)"; sleep 60
    fi
  else
    if run_capped fs-causal-ladder "$PY" -m causal_inference_provider.fs_causal_batch run --out "$RUN" >> "$LOG" 2>&1; then
      log "ladder pass finished"; fails=0
    else
      rc=$?; fails=$((fails + 1)); log "ladder pass ended rc=$rc ($fails consecutive)"; sleep 60
    fi
  fi
  if [ "$fails" -ge "$MAXFAIL" ]; then log "too many consecutive failures; marking DRIVER_FAILED"; echo "$(date -u +%FT%TZ) stage1" > "$RUN/DRIVER_FAILED"; exit 1; fi
done
# ---- stage 2: PCMCI+ comparator
fails=0
until [ -f "$RUN/READY_PCMCI" ]; do
  if run_capped fs-causal-pcmci "$PY" -m causal_inference_provider.fs_causal_discovery pcmci --out "$RUN" >> "$LOG" 2>&1; then
    log "pcmci pass finished"; fails=0
  else
    rc=$?; fails=$((fails + 1)); log "pcmci pass ended rc=$rc ($fails consecutive)"; sleep 60
  fi
  if [ "$fails" -ge "$MAXFAIL" ]; then log "pcmci: too many consecutive failures; leaving comparator PENDING"; echo "$(date -u +%FT%TZ) stage2" > "$RUN/DRIVER_FAILED"; break; fi
done
if [ -f "$RUN/READY_PCMCI" ]; then
  rm -f "$RUN/READY"  # re-finalize merges the comparator into the evidence lines (states are unchanged by it)
  run_capped fs-causal-refinalize "$PY" -m causal_inference_provider.fs_causal_batch finalize --out "$RUN" >> "$LOG" 2>&1 && log "REFINALIZED with PCMCI+"
fi
log "DRIVER END"
exit 0
