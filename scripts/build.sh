#!/usr/bin/env bash
# Build Orderit inside the dev image. The image is ephemeral, so everything is
# written under ./build on the host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ORDERIT_BUILD:-$ROOT/build}"
CONFIG="${1:-release}"
exec "$ROOT/scripts/hip-shell.sh" "cmake -S /src -B /src/build/$CONFIG -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build /src/build/$CONFIG -j 16"