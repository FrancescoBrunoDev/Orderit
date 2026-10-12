// Model geometry, read from the checkpoint rather than compiled in.
//
// There are two models in this project and they are the same family: Qwen3
// dense, 8 KV heads, head_dim 128, rope_theta 1e6, rms_norm_eps 1e-6, grouped
// query attention, RMSNorm on q and k, SwiGLU. What differs is only the size:
//
//                 layers  hidden  ffn   q heads
//   Embedding 4B     36     2560  9728     32
//   Reranker 0.6B    28     1024  3072     16
//   Reranker 4B      36     2560  9728     32
//
// So the dimensions are data and the family invariants are assertions. A file
// that satisfies the invariants and has internally consistent shapes is accepted;
// one that does not is refused at load rather than producing a plausible and
// wrong embedding.
#pragma once

#include <cstdint>
#include <string>

namespace orderit {

struct Config {
  int layers = 0;
  int hidden = 0;
  int intermediate = 0;
  int q_heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int vocab = 0;
  int max_position = 0;
  float rope_theta = 0.0f;
  float rms_norm_eps = 0.0f;

  int gqa() const { return kv_heads > 0 ? q_heads / kv_heads : 0; }
  int q_width() const { return q_heads * head_dim; }
  int kv_width() const { return kv_heads * head_dim; }

  // Family invariants. These are the values both checkpoints agree on, and the
  // only reason one set of seven kernels serves every model here.
  bool IsQwen3Family() const {
    return kv_heads == 8 && head_dim == 128 && rope_theta == 1000000.0f &&
           rms_norm_eps == 1e-6f && q_heads % kv_heads == 0 &&
           hidden % 128 == 0 && intermediate % 128 == 0;
  }
  std::string Describe() const;

  // Q8_0 is not a model property: one fp16 scale per 32 int8 weights, 34 bytes
  // per block, in every checkpoint this project reads.
  static constexpr int kQuantBlock = 32;
  static constexpr int kQuantBytes = 34;
  static int RowBytes(int k) { return (k / kQuantBlock) * kQuantBytes; }
};

}  // namespace orderit