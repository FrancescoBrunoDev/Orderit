# Baseline — Orderit F0

Every number here was measured on this machine with the toolchain this project
ships, by `tools/bench.hip`. They are the roofline that every later kernel is
judged against. gufo measured on ROCm 7.2.3; these are ROCm 10.0 / clang 23 on
the same silicon, so they are the ones that count here.

Reproduce with:

```sh
scripts/build.sh          # builds into build/release
scripts/hip-shell.sh 'build/release/orderit-bench all'
```

## Machine

| | |
|---|---|
| GPU | AMD Radeon 8060S, `gfx1151`, integrated |
| CUs | **20** as reported by `hipGetDeviceProperties` (rocminfo reports 40; see below) |
| Threads per CU | 2048, max 64 resident blocks per CU |
| Wave | 32 lanes; hardware wave64 available via `-mwavefrontsize64` |
| Registers | 196608 per CU, 64 KiB LDS per block, 2 MiB L2, 32 MiB L3 |
| Clock | 2900 MHz |
| Memory visible to the GPU | **124.00 GiB** (host RAM over the fabric; VRAM carve-out is only 512 MiB) |
| Async engine count | 8 |

`rocminfo` reports 40 compute units and `hipDeviceProp_t` reports 20. Grid
sizing that depends on this is calibrated in the bandwidth sweep rather than
trusted from either number.

`hostRegisterReadOnlySupported` reports **0** and `unifiedAddressing` reports
**0**, but `hipHostRegister(..., hipHostRegisterMapped | hipHostRegisterReadOnly)`
and `hipHostGetDevicePointer` both work anyway — see below. The capability bits
on this device are conservative, not descriptive.

## Memory bandwidth

Streaming `float4` reads, 4 repeats, best of:

| Working set | GB/s | Where |
|---|---|---|
| 4 MiB | 108.8 | L3 |
| 16 MiB | 743.4 | L3 |
| 32 MiB | 902.9 | L3 (fits exactly) |
| 64 MiB | 235.5 | RAM |
| 256 MiB | 237.6 | RAM |
| 1 GiB | 240.0 | RAM |
| 4 GiB | 238.7 | RAM |
| 8 GiB | 239.0 | RAM |

**The cliff is exactly at the 32 MiB L3.** Below it, bandwidth is ~900 GB/s;
above it, a flat ~239 GB/s.

Consequences that fix the design:

1. **Weights never live in cache.** An int8 Q8_0 model is ~3.7 GiB, so every
   forward pass streams ~3.7 GiB from RAM: `3.7 / 239 = 15.5 ms` as a hard floor
   per pass, whatever the batch size.
2. **Everything that fits in 32 MiB should be sized to fit.** Activations for a
   2560-wide hidden state are 10 KiB per token in fp32, so 8192 tokens of
   activation is 80 MiB — already out of L3. Batch sizes past ~3000 tokens stop
   benefiting from any cache residency, which is the point where the batch is
   large enough to saturate the GPU anyway.
3. Arithmetic intensity at the crossover: a pass streams `W` bytes of weights
   and does `2 * 3.6e9 * T` flops over `T` tokens. With `W = 3.7 GiB` and
   `47 TFLOPS`, compute exceeds the memory floor from about **T = 11500
   tokens**. Below that the pass is bandwidth-bound, above it compute-bound.

## WMMA register ceiling

Pure register matrix instructions, operands changed every iteration so the loop
cannot be hoisted, 8 waves per block, `multiProcessorCount * 8` blocks:

| Path | Throughput |
|---|---|
| `wmma_i32_16x16x16_iu8_w32` | **51.1 TOPS** |
| `wmma_f32_16x16x16_bf16_w32` | **47.4 TFLOPS** |

**int8 is 1.08× bf16 — nothing.** This is the single most consequential number
in the baseline, and it holds on a different ROCm from gufo's 55.07/55.05
measurement. It means:

- Quantising weights to int8 buys **no throughput** on this GPU.
- Therefore int8 is chosen for **RAM only**, never for speed.
- Therefore the extra machinery gufo needs for it (the tiled `Q8_1` activation
  layout, in-kernel dequant, `QuantSub16`) is the entire cost of the choice, and
  it must buy a RAM reduction or it is not worth writing.

Signatures that work on this toolchain (there is no `hip_wmma.h` in ROCm 10):

```cpp
using v16h  = __attribute__((__vector_size__(32))) _Float16;     // 16 x fp16
using v8f   = __attribute__((__vector_size__(32))) float;         // 8 x fp32
using v16i8 = __attribute__((__vector_size__(16))) signed char;   // 16 x int8
using v8i   = __attribute__((__vector_size__(32))) int;           // 8 x int32

__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 (cb, a, cb, b, c, sc);  // 6 args
__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 (a, b, c);            // 3 args
```

## Weight mapping and cold start

A 4 GiB file, `mmap(MAP_PRIVATE, PROT_READ)`, then `hipHostRegister` read-only:

| Step | Result |
|---|---|
| `hipHostRegister` of 4 GiB | 0.498 s → **8.6 GB/s** commit rate |
| Device alias == host pointer | yes |
| GPU reads 1 GiB, pages never touched by the host | 0.033 s → **32.2 GB/s** |
| GPU reads 1 GiB, after the host prefaulted the mapping | 0.005 s → **227.5 GB/s** |

**Serving weights straight out of a file mapping costs 7× on the first read.**
This is the fault cost being paid inside the first kernel that touches the
weights, and it is invisible unless you measure it.

The loader therefore has to fault the whole mapping in from the host before it
declares itself ready — gufo does the same in `CopyMappedWeights`, with 16 MiB
chunks, `MADV_POPULATE_READ` before each `memcpy` and `MADV_DONTNEED` after so
the file mapping keeps no page table entries. At 8.6 GB/s, a 3.7 GiB model costs
about 0.43 s to commit plus the fault pass; both are startup, not request time.

## What F0 settles

| Question | Answer |
|---|---|
| Does a kernel run on this machine? | yes |
| Is `hipHostRegister` read-only usable despite the capability bit saying no? | yes, and the device pointer aliases the host pointer |
| Does int8 pay for itself in speed? | **no, 1.08×** — it is a RAM decision only |
| What is the floor for one forward pass? | ~15.5 ms of weight streaming at any batch size |
| Where is the cache cliff? | 32 MiB, sharp |
| What must the loader do? | prefault the whole mapping before serving |
| Is there a `wmma` header? | no — use clang's amdgcn builtins |
## Tokenizer

The checkpoint carries its own vocabulary and merges inside the GGUF, so no
`tokenizer.json` is parsed. Three facts about it are not obvious and each of them
cost a debugging session:

**The three control tokens have empty text.** Qwen3 stores ids 151643, 151644 and
151645 with `content: ""`, so `<|im_end|>` cannot be found in the vocabulary by
text. They are addressed by id, and the id comes from the GGUF metadata
(`tokenizer.ggml.eot_token_id`), not from a name.

**`eos_token_id` and the end-of-turn token are different ids, one apart.**
151643 is the EOS llama.cpp appends when `add_special` is on; 151645 is
`<|im_end|>`, the end-of-turn marker. Appending the wrong one drops the cosine
against llama.cpp from **0.99986** to **0.70591**, which is the single most
expensive thing to get wrong in this file.

**The byte-level map has 324 entries, not 256 or 288.** 188 byte values are
printable and map to themselves; the remaining 68 - the space among them - map to
256..323. GPT-2's printable run starts at `!` (0x21) and *not* at the space
(0x20), which is exactly why a space becomes `Ġ` rather than looking like
itself. Starting that run at 0x20 silently shifts most of the vocabulary: every
space becomes the wrong symbol, and no single-token test notices.

Verified against llama.cpp's own `/tokenize` on the same file: **90 of 90
tokenizations identical**, over English, Italian, digits, punctuation, CJK,
Hangul, Hebrew, Arabic, emoji, code, contractions and whitespace edge cases
(fixture in `tests/fixtures/tokenizer.txt`).

## Forward pass parity

`tests/fixtures/parity.txt` holds llama.cpp's pooled vectors for seven inputs.
The CPU reference forward agrees with it at:

| input | tokens | cosine |
|---|---|---|
| `!` | 2 | 0.999742 |
| `hello world` | 3 | 0.999857 |
| `The quick brown fox jumps over the lazy dog.` | 11 | 0.999722 |
| `Instruct: ...\nQuery: what is a Strix Halo` | 26 | 0.999669 |
| `Il gatto dorme sul divano.` | 10 | 0.999595 |
| `def fibonacci(n): ...` (multi-line code) | 27 | 0.999780 |
| `東京は日本の首都です、人口は約1400万人です。` | 18 | 0.999764 |

The residual ~1.5e-4 is llama.cpp's own accumulation order and its fp16 KV plane,
not an error here. The CPU reference runs at about **2.5 s per token** on the
32 cores of this machine, which is the number F4 through F6 are measured against.

Note what llama.cpp evaluates: for a one-token input it evaluates **two**,
because it appends the end token. A single-token probe that forgets this is
comparing different computations, and it is why a one-token parity check is
worth having: it removes attention from the picture entirely, since attention
over a single position is exactly `V[0]` and q_norm, k_norm and RoPE cannot
affect the result.

## What the reference does not do

llama.cpp's `/embedding` applies **no instruction template**. The vectors in the
current pipeline were therefore produced from raw text without the
`Instruct: <task>\nQuery:` wrapper these models are trained to be instruction-aware
with. Matching llama.cpp is the right gate for F3 - it is the thing being
replaced and it is the only independent oracle available - but it is not the same
as matching the model card. F7 measures what the instruction is worth, on the
same checkpoint, with both prompts.

## The baseline to beat

Reproduced by `scripts/verify.sh` against a freshly started llama.cpp serving the
same GGUF with `-ngl 0`:

| input | llama.cpp on the CPU |
|---|---|
| ~2.2k tokens, steady state | **4.97 s** |
| first request after model load | 10.4 s |

The second figure is a cold page cache and should not be quoted as the
throughput baseline. The steady-state 5 s is the honest one, and it is what the
speed targets are measured against: at 47 TFLOPS a 2.2k-token forward is
0.33 s of compute, so the GPU target is **7-10x**, not the 15-20x an early cold
measurement suggested.

## Run-to-run variation

Numbers taken more than once on this machine, from `tools/bench.hip`:

| | first run | second run |
|---|---|---|
| streaming read, 1 GiB | 240.0 GB/s | 240.7 GB/s |
| WMMA int8 ceiling | 51.1 TOPS | 49.5 TOPS |
| WMMA bf16 ceiling | 47.4 TFLOPS | 48.1 TFLOPS |
| **int8 over bf16** | **1.08x** | **1.03x** |

Bandwidth is stable to about 0.3%. The matrix ceilings move by 3%, which is the
size of the effect being claimed: int8 is not faster than bf16 on this silicon,
and the two measurements bracket the ratio around 1.05. The conclusion does not
depend on which run is quoted.

## The comparison that matters

Measured against llama.cpp serving **the same two checkpoints**, both on the
GPU with `-ngl 99`. Reproduce with `scripts/verify.sh` plus the two probe
containers below.

| | llama.cpp | Orderit, projected |
|---|---|---|
| Qwen3-Embedding-4B, 2002 tokens | **934 ms** (2143 tok/s, 15.4 TFLOPS, 32% of the ceiling) | 320-450 ms (2.1-2.9x) |
| Qwen3-Reranker-0.6B, 1 query x 20 docs, 3270 tokens | **366 ms** (8934 tok/s, 7.9 TFLOPS, 16%) | 64-90 ms (4.1-5.7x) |

The reranker is the stronger case and the reason is measured, not guessed: with
hidden 1024 and ffn 3072 the matrices are small, and llama.cpp's generic path
runs them at 16% of the ceiling. The projection uses gufo's documented custom
kernel efficiency on comparable shapes (32 TFLOPS) and an optimistic 45.

## Memory, measured as MemAvailable

| | cost |
|---|---|
| llama.cpp, default context (4 slots x 40960) | **16.8 GB** |
| llama.cpp, `-c 512 -np 1` on both models | ~6.4 GB |
| Orderit, mapped weights + batch-sized scratch | ~5.1 GB |

5.4 GB of llama.cpp's 16.8 GB is a KV cache pool that an encoder-decoder with no
decode never reads. It is real waste, and it is also reclaimable with two flags,
so the honest advantage is **~1.3 GB**, not the 2-3 GB an earlier reading of
`VmSize` suggested. `VmSize` is address space: the 4B server reserves 17.9 GB of
it and has 1.3 GB resident.

## Three numbers that were wrong the first time

Recorded because the first three claims this project made about itself were
each off in its favour, and all three were caught by measuring before promising.

| claimed | measured | what the mistake was |
|---|---|---|
| 10.4 s per document | 4.97 s steady state | the first request after a model load, cold page cache |
| int8 is 1.08x bf16 | 1.03x on a second run | a 3% effect measured twice as if it were 8% |
| saves 2-3 GB of RAM | ~1.3 GB | read `VmSize` as if it were resident memory |

The generalisable version: a projection built from a hardware ceiling and
someone else's efficiency is arithmetic, not evidence. F4 exists to replace the
2.1-2.9x above with a measurement, and the projection should be deleted the
moment the measurement disagrees with it.

## The NPU is not an alternative to the GPU for these two models

Measured with halogen-flash-server 0.18.1 serving `qwen3-embedding-4b` and
`qwen3-reranker-0.6b` on the XDNA (`/dev/accel/accel0`, driver `amdxdna`, XRT
2.21.75), against llama.cpp with `-ngl 99` serving the same checkpoints.

| | NPU | GPU | NPU is |
|---|---|---|---|
| Qwen3-Embedding-4B, 2002 tokens | 1040 ms | 891 ms | 1.17x slower |
| Qwen3-Reranker-0.6B, 1 query x 20 docs | 1126 ms | 366 ms | 3.08x slower |

The NPU's memory bandwidth is the reason, and the gap widens with batch size, so
the batched jobs a RAG pipeline sends are the worst case for it. The NPU engine
also cannot be reached through the unified proxy: `/v1/embeddings` and
`/v1/rerank` there always route to the llama.cpp containers and the `model`
field is ignored. Reaching it means port 8730 directly.

The prerequisite the documentation names is real and already satisfied on this
host: `halogen-fabric-clock.service` is installed and enabled and reports
`held (manual)`, with `pp_dpm_fclk` at 2000 MHz, the top of its range. Without
it, GPU and NPU work together can hang a Strix Halo machine.

**A trap worth recording:** `rocm-smi --showuse` reports 100% on this APU while
idle, so it cannot be used to prove that a kernel ran on the GPU. The signal
that does work is `npu.passes` and `npu.loaded` on `/health`. The first
attempt at this measurement went through the proxy, hit the llama.cpp
containers, and produced two "different" engines agreeing on 2560 of 2560
components bit for bit - which was the clue that nothing had run on the NPU at
all.

NPU models are now off in `halogen-stack.sh`; re-enable with
`STACK_NPU_MODELS=qwen3-asr-1.7b`.
