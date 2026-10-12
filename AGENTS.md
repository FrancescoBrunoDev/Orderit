# Working on Orderit

## Platform

Linux x86-64, one AMD GPU, `gfx1151`. The build refuses any other architecture on purpose: every number in `docs/BASELINE.md` is a statement about this silicon and this ROCm, and a kernel tuned here is not portable, so pretending otherwise would only cost time later.

Never commit a checkpoint, a logit dump, or an embedding file. The fixtures under `tests/fixtures/` are small, reproducible and belong in the repository; everything else is generated and is not.

## The two things that make results trustworthy

**Every number states its conditions.** Working set size, batch size, tile shape, ROCm version, and what it was compared against. A speed figure without conditions is not a measurement.

**Tolerances are contracts, not suggestions.** The gates are set in `tests/` and are not raised to make a change pass. If a kernel is faster and the cosine fell, the kernel is wrong - that is the finding, not an obstacle to it.

## Before claiming a kernel is done

1. It agrees with `src/reference.hpp` (double precision, host) for the operator in isolation.
2. It runs twice on the same input and produces byte-identical output. gufo's attention tests do this and it catches stale registers in a prefetch path, which no tolerance catches.
3. The end-to-end cosine against `tests/fixtures/parity.txt` is above 0.999.
4. Its resource usage is recorded: VGPRs, LDS bytes, scratch. gufo gates on zero scratch with a baseline file because a compiler update can silently make a kernel spill.

## Numerics

Storage and accumulation are decided per stage and written down in `src/reference.hpp` and the kernel that implements them. Do not change a reduction order to make something faster or to make a number prettier; gufo keeps `ggml_gemm_parity` and a `logit_comparator` precisely so that a change of order is visible rather than silent.

fp16 and bf16 conversion is hand written in `src/half.hpp` rather than using compiler intrinsics, because the device compiler's rounding is a build flag and a conversion that depends on a flag cannot carry a stated tolerance.

## Style

Google style, two-space indent, 80 columns, matching gufo's `.clang-format`. `snake_case` files, `PascalCase` types, `kCamelCase` constants, namespaces named after the directory.

Comments explain why, never what. A comment that restates the line above it is removed; a comment that records a measurement, a rejected alternative, or a trap stays.

## Layout

`src/` holds the engine. `kernels.hip` is one file on purpose: seven kernels do not need seven files, and splitting them makes the resource budget harder to see. `tools/` is development-only and never linked into the binary. `docs/BASELINE.md` is the record of what this machine does; extend it before extending the code.

## Tests

```sh
scripts/build.sh
scripts/hip-shell.sh 'ctest --test-dir build/release --output-on-failure'
```

`reference_test` is host-only and needs no checkpoint. `model_test` needs one; pass it as `ORDERIT_MODEL`. Labels on the CTest tests carry the phase (`f1`, `f2`, `f3`) so a regression can be bisected to the phase that introduced it.

To regenerate the fixtures, run a llama-server against the same GGUF and use `tools/make_tokenizer_fixture.py` or `tools/make_parity_fixture.py`. They are the only sanctioned way to produce a reference, and they overwrite the files rather than appending.