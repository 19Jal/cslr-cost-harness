#!/usr/bin/env bash
# harness_dev.sh — run a command inside the project container.
#
#   scripts/dev.sh                     interactive shell
#   scripts/dev.sh pytest              run the test suite
#   scripts/dev.sh python -m cslr_cost.harness.cli ...
#
# Your working copy is mounted live at /workspace, so edits on the host take
# effect immediately inside — no rebuild for code changes. Rebuild only when
# the system layer changes (apt packages, SCTK, ctcdecode, or the lockfile):
#
#   docker build -t cslr-cost-harness:dev .
#
# Why each flag is here:
#
#   --gpus all              CUDA devices; without it torch.cuda.is_available()
#                           is False and every measurement is meaningless.
#   -v $REPO:/workspace     your code, live. The image's venv lives at
#                           /opt/venv, deliberately outside this mount, so the
#                           bind mount cannot shadow it. (An earlier version
#                           kept it at /workspace/.venv and preserved it with an
#                           anonymous volume — but that made Docker create a
#                           root-owned /workspace/.venv on the host in any
#                           checkout lacking one, which then blocked `uv`.)
#   --user $(id -u):$(id -g)  files written into the repo stay yours. Requires
#                           the interpreter to live outside /root — see the
#                           UV_PYTHON_INSTALL_DIR note in the Dockerfile.
#   --ipc=host              torch DataLoader workers need shared memory; the
#                           default 64MB /dev/shm causes obscure worker crashes.
#   -v .../powercap:ro      Intel RAPL energy counters, for CPU-side energy.
#                           Mounted at /host/powercap rather than under /sys:
#                           Docker mounts its own sysfs over /sys, which
#                           silently shadows any bind mount at a /sys subpath —
#                           the class symlinks are visible inside but dangle.
#   --group-add <users gid> RAPL's energy_uj is 0440 root:users on the host, and
#                           the container's gid would not otherwise match, so
#                           the file is there but unreadable.
#   -e CSLR_DATA            where the data and checkpoints are (docs/data.md).
#   -e CSLR_GIT_SHA/DIRTY   resolved on the HOST. In a worktree, .git is a file
#                           pointing at <main repo>/.git/worktrees/<name>, which
#                           is outside the bind mount — so `git rev-parse` fails
#                           inside and every results row loses the field that
#                           ties it to the code that produced it.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-cslr-cost-harness:dev}"
# Datasets, checkpoints and ONNX files (see docs/data.md). Mounted at the SAME
# absolute path inside the container, so symlinks inside it resolve on both
# sides. Default: ./data in this checkout.
SHARED_DATA="${CSLR_DATA:-$REPO/data}"

# Provenance for the results rows, resolved here because the container cannot.
GIT_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)"
GIT_DIRTY=false
[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ] && GIT_DIRTY=true

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "error: image '$IMAGE' not found. Build it first:" >&2
  echo "  docker build -t $IMAGE $REPO" >&2
  exit 1
fi

GPU_FLAG=()
# `docker info` can fail transiently -- once, right after another container
# exited, it did, and a queued job ran without a GPU. So
# ask a few times, and when the host plainly has a GPU, refuse rather than run
# a GPU job on CPU: a warning scrolls past, a CPU fallback does not announce
# itself. CSLR_ALLOW_NO_GPU=1 runs GPU-less on purpose.
for _ in 1 2 3; do
  if docker info 2>/dev/null | grep -qi "runtimes:.*nvidia"; then
    GPU_FLAG=(--gpus all)
    break
  fi
  sleep 2
done
if [ ${#GPU_FLAG[@]} -eq 0 ]; then
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
     && [ "${CSLR_ALLOW_NO_GPU:-0}" != "1" ]; then
    echo "error: this host has a GPU but no NVIDIA container runtime was detected" \
         "(docker info, 3 tries); refusing to run without it." \
         "Set CSLR_ALLOW_NO_GPU=1 to run GPU-less on purpose." >&2
    exit 1
  fi
  echo "warning: no NVIDIA container runtime detected — running without GPU." >&2
fi

# Allocate a TTY only when there is one. Passing -t without a terminal fails
# with "the input device is not a TTY", which would make this unusable from
# scripts, cron and CI.
TTY_FLAG=(-i)
[ -t 0 ] && [ -t 1 ] && TTY_FLAG=(-i -t)

# Intel RAPL, if it has been made group-readable (see docs/devices.md).
# Absent or unreadable is not an error: the harness records CPU energy as
# unavailable with a stated reason rather than reporting it as zero.
RAPL_FLAGS=()
HOST_POWERCAP=/sys/devices/virtual/powercap
if [ -d "$HOST_POWERCAP" ]; then
  RAPL_FLAGS+=(-v "$HOST_POWERCAP":/host/powercap:ro -e CSLR_RAPL_ROOT=/host/powercap)
  USERS_GID="$(getent group users | cut -d: -f3 || true)"
  [ -n "$USERS_GID" ] && RAPL_FLAGS+=(--group-add "$USERS_GID")
fi

exec docker run --rm "${TTY_FLAG[@]}" \
  "${GPU_FLAG[@]}" \
  --ipc=host \
  "${RAPL_FLAGS[@]}" \
  --user "$(id -u):$(id -g)" \
  -v "$REPO":/workspace \
  -v "$SHARED_DATA":"$SHARED_DATA":ro \
  -e CSLR_DATA="$SHARED_DATA" \
  -e CSLR_GIT_SHA="$GIT_SHA" \
  -e CSLR_GIT_DIRTY="$GIT_DIRTY" \
  -w /workspace \
  "$IMAGE" \
  "${@:-bash}"
