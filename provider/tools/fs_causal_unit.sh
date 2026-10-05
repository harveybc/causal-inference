#!/bin/bash
# Start (or restart) the FS-CAUSAL driver as a durable systemd --user unit on worker_b.
#   FS_CAUSAL_CAP=<1.25x measured pilot peak> FS_CAUSAL_PEAK_EVIDENCE=<record.json> FS_CAUSAL_PEAK_BYTES=<n> tools/fs_causal_unit.sh
# The unit restarts the driver on failure (the driver itself is resumable per chunk) and survives the ssh session.
set -euo pipefail
ROOT="${FS_CAUSAL_ROOT:-$HOME/.local/state/canonical_20261003/fs_causal}"
UNIT="${FS_CAUSAL_UNIT:-fs-causal-driver}"
: "${FS_CAUSAL_CAP:?set FS_CAUSAL_CAP to 1.25x the measured pilot cgroup peak}"
if systemctl --user is-active --quiet "$UNIT.service"; then
  echo "$UNIT already active"; systemctl --user status "$UNIT.service" --no-pager | head -5; exit 0
fi
systemctl --user reset-failed "$UNIT.service" 2>/dev/null || true
systemd-run --user --unit="$UNIT" --description="FS-CAUSAL ladder + PCMCI+ driver (worker_b CPU, crispdm-run capped)" \
  -p Restart=on-failure -p RestartSec=120s \
  --setenv=FS_CAUSAL_ROOT="$ROOT" --setenv=FS_CAUSAL_CAP="$FS_CAUSAL_CAP" \
  --setenv=FS_CAUSAL_PEAK_EVIDENCE="${FS_CAUSAL_PEAK_EVIDENCE:-}" --setenv=FS_CAUSAL_PEAK_BYTES="${FS_CAUSAL_PEAK_BYTES:-}" \
  --setenv=FS_CAUSAL_WALL="${FS_CAUSAL_WALL:-6h}" --setenv=PATH="$PATH" \
  "$ROOT/code/provider/tools/fs_causal_driver.sh"
sleep 2
systemctl --user status "$UNIT.service" --no-pager | head -8
