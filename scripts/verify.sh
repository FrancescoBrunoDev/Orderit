#!/usr/bin/env bash
# Verify Orderit from scratch. Nothing here trusts a file this repository
# shipped: the fixtures are regenerated from a llama.cpp started right now, and
# the tests are rerun against those.
#
#   scripts/verify.sh                 # uses the defaults below
#   REF_IMAGE=... REF_PORT=... scripts/verify.sh
#
# Exits non-zero if any step fails, and says which.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${ORDERIT_MODEL:-/mnt/fanxiang/models/halogen-stack/small/Qwen3-Embedding-4B-Q8_0.gguf}"
REF_IMAGE="${REF_IMAGE:-docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0}"
REF_PORT="${REF_PORT:-8091}"
REF_NAME="${REF_NAME:-orderit-verify-ref}"
REF_URL="http://127.0.0.1:${REF_PORT}"
FAILED=0

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
fail() { printf '   FAILED: %s\n' "$*"; FAILED=1; }

# ---------------------------------------------------------------- 0. the host
step "host"
if [ ! -e /dev/kfd ] || [ ! -e /dev/dri ]; then
  fail "no /dev/kfd or /dev/dri; this only runs on the AMD GPU"
fi
echo "   $(lscpu | awk -F: '/Model name/{print $2}' | xargs)"
echo "   $(rocm-smi --showproductname --csv 2>/dev/null | tail -1 || echo 'rocm-smi not on the host')"
if [ ! -f "$MODEL" ]; then
  fail "checkpoint not found: $MODEL  (set ORDERIT_MODEL)"
  echo "cannot continue without a checkpoint"; exit 1
fi
echo "   checkpoint: $MODEL ($(du -h "$MODEL" | cut -f1))"

# ---------------------------------------------------------- 1. build from zero
step "build"
if ! ./scripts/build.sh > /tmp/orderit-build.log 2>&1; then
  fail "build failed, see /tmp/orderit-build.log"
  tail -20 /tmp/orderit-build.log
  exit 1
fi
echo "   ok: $(ls build/release/orderit-bench build/release/tests/model_test 2>/dev/null | wc -l) binaries"

# --------------------------------------------------- 2. an independent oracle
step "reference implementation"
STARTED_REF=0
if curl -sf --max-time 5 "${REF_URL}/health" > /dev/null 2>&1; then
  echo "   using the llama.cpp already listening on ${REF_URL}"
else
  echo "   starting $REF_IMAGE on ${REF_URL} (CPU, read-only, -ngl 0)"
  podman rm -f "$REF_NAME" > /dev/null 2>&1
  if ! podman run -d --name "$REF_NAME" \
        --group-add keep-groups \
        -p "127.0.0.1:${REF_PORT}:${REF_PORT}" \
        -v "$(dirname "$MODEL"):/models:ro" \
        "$REF_IMAGE" \
        llama-server -m "/models/$(basename "$MODEL")" -ngl 0 --embeddings \
        --pooling last --host 0.0.0.0 --port "${REF_PORT}" > /dev/null; then
    fail "cannot start the reference container"
    exit 1
  fi
  STARTED_REF=1
  printf '   waiting for it to load'
  for _ in $(seq 1 120); do
    if curl -sf --max-time 3 "${REF_URL}/health" > /dev/null 2>&1; then break; fi
    printf '.'; sleep 2
  done
  echo
  curl -sf --max-time 5 "${REF_URL}/health" > /dev/null 2>&1 || {
    fail "the reference never became healthy; podman logs $REF_NAME"
    exit 1
  }
fi
curl -sf --max-time 20 "${REF_URL}/v1/models" | head -c 120; echo

# --------------------------------------------- 3. regenerate the fixtures live
step "regenerate fixtures from that server"
# They are overwritten, not appended: if the committed ones were wrong, this is
# what replaces them.
if python3 tools/make_tokenizer_fixture.py "${REF_URL}/tokenize" > /tmp/fix-tok.log 2>&1; then
  echo "   tokenizer.txt regenerated ($(grep -c . tests/fixtures/tokenizer.txt) lines)"
else
  fail "tokenizer fixture"; tail -5 /tmp/fix-tok.log
fi
if python3 tools/make_parity_fixture.py "${REF_URL}/embedding" > /tmp/fix-par.log 2>&1; then
  echo "   parity.txt regenerated ($(grep -vc '^#' tests/fixtures/parity.txt) cases)"
else
  fail "parity fixture"; tail -5 /tmp/fix-par.log
fi

# ------------------------------------------------------- 4. rerun the tests
step "tests, against the fixtures just written"
./scripts/hip-shell.sh "ctest --test-dir /src/build/release --output-on-failure" 2>&1 | tail -12

# ------------------------------------------------------ 5. the machine itself
step "the machine, measured again"
./scripts/hip-shell.sh 'build/release/orderit-bench info' 2>&1 | grep -E "gcnArchName|multiProcessor|totalGlobalMem|warpSize"
./scripts/hip-shell.sh 'build/release/orderit-bench mem'  2>&1 | tail -6
./scripts/hip-shell.sh 'build/release/orderit-bench wmma' 2>&1 | tail -3

# --------------------------------------------------------------- 6. the CPU
step "the baseline this is meant to beat"
START=$(date +%s%N)
if curl -sf --max-time 120 "${REF_URL}/embedding" -H 'Content-Type: application/json' \
      -d "{\"content\":\"$(python3 -c 'print("The quick brown fox. " * 200)')\"}" > /dev/null; then
  END=$(date +%s%N)
  echo "   llama.cpp CPU, ~2.2k tokens: $(( (END - START) / 1000000 )) ms"
fi

[ "$STARTED_REF" = 1 ] && { echo; echo "   stopping $REF_NAME"; podman rm -f "$REF_NAME" > /dev/null 2>&1; }

step "verdict"
if [ "$FAILED" = 0 ]; then
  echo "   everything passed"
else
  echo "   something did not pass; see above"
fi
exit "$FAILED"