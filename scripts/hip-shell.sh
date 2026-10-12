#!/usr/bin/env bash
# Run a command inside the ad-hoc Orderit dev image with the GPU and the source
# tree attached.
#
#   scripts/hip-shell.sh                 # interactive bash in the dev image
#   scripts/hip-shell.sh 'cmake --build build'
#   scripts/hip-shell.sh --build         # rebuild the dev image first
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ORDERIT_IMAGE:-orderit-dev:rocm10}"

if [ "${1:-}" = "--build" ]; then
  podman build -f "$ROOT/deploy/Dockerfile.dev" -t "$IMAGE" "$ROOT/deploy"
  shift
fi

TTY=()
if [ -t 0 ] && [ -t 1 ]; then TTY=(-it); fi

# The checkpoints are not in the repo. Anything that exists on the host under
# these roots is passed through read-only.
HOST_MOUNTS=()
for dir in ${ORDERIT_MOUNTS:-/mnt/fanxiang "$HOME/models"}; do
  [ -d "$dir" ] && HOST_MOUNTS+=(-v "$dir:$dir:ro")
done

if [ "$#" -eq 0 ]; then set -- /bin/bash; fi
exec podman run --rm "${TTY[@]}" \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --ulimit memlock=-1:-1 \
  -v "$ROOT:/src:Z" \
  "${HOST_MOUNTS[@]+"${HOST_MOUNTS[@]}"}" \
  -w /src \
  "$IMAGE" \
  /bin/bash -lc "$*"