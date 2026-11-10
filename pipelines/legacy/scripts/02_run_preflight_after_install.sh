#!/usr/bin/env bash
set -u
WMSR_ROOT=/root/wmsr-repro
WMSR_SETUP_PID="$(cat "$WMSR_ROOT/records/environment_setup.pid")"
while kill -0 "$WMSR_SETUP_PID" 2>/dev/null; do
    if grep -q '^ENVIRONMENT_SETUP_COMPLETE' "$WMSR_ROOT/logs/environment_setup.log"; then break; fi
    sleep 5
done
if ! grep -q '^ENVIRONMENT_SETUP_COMPLETE' "$WMSR_ROOT/logs/environment_setup.log"; then
    printf 'ENVIRONMENT_SETUP_FAILED; see logs/environment_setup.log\n'
    exit 1
fi
export XDG_CACHE_HOME="$WMSR_ROOT/.cache"
export MPLCONFIGDIR="$WMSR_ROOT/.cache/matplotlib"
export TRITON_CACHE_DIR="$WMSR_ROOT/.cache/triton"
"$WMSR_ROOT/.venv/bin/python" "$WMSR_ROOT/scripts/02_model_preflight.py" > "$WMSR_ROOT/logs/model_preflight.log" 2>&1
WMSR_MODEL_STATUS=$?
printf 'MODEL_PREFLIGHT_EXIT=%s\n' "$WMSR_MODEL_STATUS"
"$WMSR_ROOT/.venv/bin/python" "$WMSR_ROOT/scripts/03_inspect_sst_sample.py" > "$WMSR_ROOT/logs/first_sst_sample.log" 2>&1
WMSR_SAMPLE_STATUS=$?
printf 'SAMPLE_CHECK_EXIT=%s\n' "$WMSR_SAMPLE_STATUS"
if [ "$WMSR_MODEL_STATUS" -eq 0 ] && [ "$WMSR_SAMPLE_STATUS" -eq 0 ]; then
    printf 'BOOTSTRAP_VERIFICATION_COMPLETE\n'
else
    printf 'BOOTSTRAP_VERIFICATION_NEEDS_FIX\n'
    exit 1
fi
