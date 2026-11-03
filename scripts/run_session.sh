#!/usr/bin/env bash
# run_session.sh: one cost session on one device, every pipeline back to back.
#
# Comparisons between models are made within one session, on that session's
# idle power (docs/protocol.md, P2). Input: the 1080-line ladder rung (P6),
# batch 1 and batch 32, paced at 25 fps, and a 10-minute sustained hold.
#
# Desktop (container):  scripts/dev.sh scripts/run_session.sh
# Jetson (native):      HARNESS="$HOME/.venvs/cslr/bin/python" scripts/run_session.sh
#
# Environment:
#   KEYPOINT=1|0          the MSKA keypoint pipeline (HRNet-w48 + flip, FP32)   default 1
#   RGB_MODELS            "vac-smkd corrnet twostream-slr twostream-kp"          default "vac-smkd"
#   RGB_PRECISIONS        "fp16 fp32"                                           default "fp16 fp32"
#   RGB_HEAD_MAX_FRAMES   e.g. 160, for a head engine too large for the device  default unset
#   NO_BATCH_SWEEP=1      skip the batch-32 row where it does not fit
#   OUT                   results file        default results/cost_session_<host>.jsonl
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
HARNESS="${HARNESS:-python}"
ONNX_DIR="${ONNX_DIR:-${CSLR_DATA:-data}/onnx}"
OUT="${OUT:-results/cost_session_$(hostname -s).jsonl}"
LOG_DIR="${LOG_DIR:-logs/session_$(hostname -s)}"
KEYPOINT="${KEYPOINT:-1}"
RGB_MODELS="${RGB_MODELS:-vac-smkd}"
RGB_PRECISIONS="${RGB_PRECISIONS:-fp16 fp32}"
RGB_HEAD_MAX_FRAMES="${RGB_HEAD_MAX_FRAMES:-}"
EXTRA=()
[ "${NO_BATCH_SWEEP:-0}" = 1 ] && EXTRA+=(--no-batch-sweep)

# Rows record the commit that produced them; a dirty tree makes that a lie.
[ -z "$(git status --porcelain 2>/dev/null)" ] || { echo "tree dirty; refusing (rows record the commit)"; exit 1; }
mkdir -p "$LOG_DIR" "$(dirname "$OUT")"

COMMON=(--rungs 1080p --clips 1 --frames 128
        --seconds 60 --idle 10 --settle 20 --warmup 5 --warmup-clips 2
        --batch 32 --paced-fps 25 --sustained-minutes 10
        --runtime tensorrt --onnx-dir "$ONNX_DIR" --out "$OUT")

run() {   # run <tag> <harness args...>
    local tag="$1"; shift
    local log="$LOG_DIR/$(date -u +%Y%m%dT%H%M%SZ)_${tag}.log"
    echo "=== $(date '+%F %T') $tag -> $log"
    if $HARNESS -m cslr_cost.harness.cli ladder "${COMMON[@]}" "$@" "${EXTRA[@]+"${EXTRA[@]}"}" >"$log" 2>&1; then
        grep -E "^(all|[0-9]+/[0-9]+ runs)" "$log" | tail -2 | sed 's/^/    /' || true
    else
        echo "    FAILED (exit $?), see the log" >&2
        echo "$tag" >> "$LOG_DIR/failures.txt"
    fi
}

echo "=== session on $(hostname -s), commit $(git rev-parse --short HEAD 2>/dev/null), out $OUT"

if [ "$KEYPOINT" = 1 ]; then
    # MSKA as released: HRNet-w48 DARK 384x288 with flip test, four-head ensemble, beam 5
    run tensorrt_mska_hrnet-w48_fp32_1080p --pose-scale hrnet-w48 --precision fp32 --beam 5
fi

for m in $RGB_MODELS; do
    for p in $RGB_PRECISIONS; do
        beam=10; case "$m" in twostream-*) beam=5 ;; esac     # each model's own decoding
        run "tensorrt_${m}_${p}_1080p" --rgb-model "$m" --precision "$p" --beam "$beam" \
            ${RGB_HEAD_MAX_FRAMES:+--rgb-head-max-frames "$RGB_HEAD_MAX_FRAMES"}
    done
done
echo "=== session done $(date '+%F %T')"
