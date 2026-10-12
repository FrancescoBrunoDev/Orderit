// CPU references for every operation the engine performs, in double
// precision, written to be checked by hand and read like a formula.
//
// These are the definition of correct. A kernel on the GPU agrees with these,
// or the kernel is wrong. Two GPU kernels agreeing with each other proves
// nothing: that rule is inherited from gufo and it is the one that matters here,
// because the only other runtime available for comparison is llama.cpp serving
// the same checkpoint, and it is a different implementation rather than an
// independent statement of the model.
//
// Nothing in this header is compiled into the production binary's request path
// and none of it is HIP; it is host-only and exists so the GPU kernels have
// something to be measured against.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "config.hpp"
#include "half.hpp"

namespace orderit::reference {

using Vector = std::vector<float>;
using Matrix = std::vector<float>;  // row-major

// fp16/bf16 conversions are shared with the production path and live in
// half.hpp; the reference header only declares what is an oracle.
using orderit::Bf16ToFp32;
using orderit::Fp16ToFp32;
using orderit::Fp32ToBf16;
using orderit::Fp32ToFp16;

// ------------------------------------------------------------------- norms

// RMSNorm over one row. Weights are applied after normalisation, which is the
// order Qwen3 uses (norm, then scale), and the sum of squares is taken on the
// unweighted vector.
inline Vector RmsNorm(const float* x, const float* weight, int n, float eps) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += static_cast<double>(x[i]) * x[i];
  const float inv = static_cast<float>(1.0 / std::sqrt(sum / n + eps));
  Vector out(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) out[i] = x[i] * inv * weight[i];
  return out;
}

// --------------------------------------------------------------------- rope
//
// Half-split (GPT-NeoX) layout: element i is paired with i + rotary/2, not with
// an interleaved neighbour. Every Qwen path in gufo uses this layout; an
// interleaved implementation looks right on small inputs and is subtly wrong.
inline void ApplyRope(float* vec, int rotary, int pos, float theta) {
  for (int i = 0; i < rotary / 2; ++i) {
    const float exponent = 2.0f * static_cast<float>(i) / static_cast<float>(rotary);
    const float inv_freq = 1.0f / std::pow(theta, exponent);
    const float angle = static_cast<float>(pos) * inv_freq;
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const float a = vec[i];
    const float b = vec[i + rotary / 2];
    vec[i] = a * c - b * s;
    vec[i + rotary / 2] = a * s + b * c;
  }
  // Elements past the rotary span are untouched.
}

inline std::vector<float> RopeFrequencies(int rotary, int positions, float theta) {
  std::vector<float> out(static_cast<std::size_t>(rotary / 2) * positions);
  for (int p = 0; p < positions; ++p) {
    for (int i = 0; i < rotary / 2; ++i) {
      const float exponent = 2.0f * static_cast<float>(i) / static_cast<float>(rotary);
      out[static_cast<std::size_t>(p) * (rotary / 2) + i] =
          static_cast<float>(p) / std::pow(theta, exponent);
    }
  }
  return out;
}

// --------------------------------------------------------------------- ffn

inline float Silu(float x) { return x / (1.0f + std::expf(-x)); }

inline Vector Swiglu(const float* gate, const float* up, int n) {
  Vector out(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) out[i] = Silu(gate[i]) * up[i];
  return out;
}

// ----------------------------------------------------------------- softmax

inline Vector Softmax(const float* logits, int n) {
  const float max = *std::max_element(logits, logits + n);
  Vector out(static_cast<std::size_t>(n));
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    out[i] = std::exp(static_cast<double>(logits[i]) - max);
    sum += out[i];
  }
  for (int i = 0; i < n; ++i) out[i] = static_cast<float>(out[i] / sum);
  return out;
}

// ------------------------------------------------------------- dequantising

// One Q8_0 block: fp16 scale then 32 signed bytes. The scale is the only
// per-block state, so a 16-deep matrix fragment always sits inside two blocks
// at most and never straddles a scale change mid-fragment.
struct Q8Block {
  float scale;
  signed char q[Config::kQuantBlock];
};

inline Q8Block ReadQ8Block(const unsigned char* raw) {
  Q8Block block;
  std::uint16_t bits;
  std::memcpy(&bits, raw, 2);
  block.scale = Fp16ToFp32(bits);
  std::memcpy(block.q, raw + 2, Config::kQuantBlock);
  return block;
}

inline float DequantQ8(const unsigned char* data, int k, int index) {
  const Q8Block block = ReadQ8Block(data + (index / Config::kQuantBlock) * Config::kQuantBytes);
  return block.scale * static_cast<float>(block.q[index % Config::kQuantBlock]);
}

// Row `row` of a quantised weight matrix, dequantised. Weights are stored
// row-major as [out_features, in_features], which is the layout every GGUF
// producer writes and the layout the kernel reads with no repacking.
inline Vector DequantRow(const unsigned char* data, int row, int k) {
  const unsigned char* src = data + static_cast<std::size_t>(row) * Config::RowBytes(k);
  Vector out(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) out[i] = DequantQ8(src, k, i);
  return out;
}

// -------------------------------------------------------------------- gemm
//
// out[m, n] = sum_k x[m, k] * w[n, k], accumulated left to right in double.
// The accumulation order is part of the contract: a kernel that reduces in a
// different order will differ in the last bits, and the tolerance has to know
// that rather than discover it.
inline void Gemm(const float* x, const unsigned char* w, float* out, int tokens, int n,
                 int k, float* w_scratch) {
  for (int t = 0; t < tokens; ++t) {
    const float* row = x + static_cast<std::size_t>(t) * k;
    for (int j = 0; j < n; ++j) {
      Vector wr = DequantRow(w, j, k);
      if (w_scratch) std::copy(wr.begin(), wr.end(), w_scratch);
      double acc = 0.0;
      for (int i = 0; i < k; ++i) acc += static_cast<double>(row[i]) * wr[i];
      out[static_cast<std::size_t>(t) * n + j] = static_cast<float>(acc);
    }
  }
}

// -------------------------------------------------------------- attention

// Causal attention for one sequence, grouped-query: `head` is a query head and
// reads KV head `head / Gqa`. Scores scale by 1/sqrt(head_dim).
inline void Attention(const float* q, const float* k, const float* v, float* out,
                      int q_heads, int kv_heads, int head_dim, int length) {
  const int gqa = q_heads / kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  Vector logits(static_cast<std::size_t>(length));
  for (int h = 0; h < q_heads; ++h) {
    const int kvh = h / gqa;
    for (int i = 0; i < length; ++i) {
      const float* qh = q + static_cast<std::size_t>(h) * head_dim + static_cast<std::size_t>(i) * q_heads * head_dim;
      for (int j = 0; j <= i; ++j) {  // causal: position j sees 0..j
        const float* kh = k + static_cast<std::size_t>(kvh) * head_dim + static_cast<std::size_t>(j) * kv_heads * head_dim;
        double acc = 0.0;
        for (int d = 0; d < head_dim; ++d) acc += static_cast<double>(qh[d]) * kh[d];
        logits[j] = static_cast<float>(acc) * scale;
      }
      const Vector p = Softmax(logits.data(), i + 1);
      float* oh = out + static_cast<std::size_t>(h) * head_dim + static_cast<std::size_t>(i) * q_heads * head_dim;
      std::fill(oh, oh + head_dim, 0.0f);
      for (int j = 0; j <= i; ++j) {
        const float* vh = v + static_cast<std::size_t>(kvh) * head_dim + static_cast<std::size_t>(j) * kv_heads * head_dim;
        for (int d = 0; d < head_dim; ++d) oh[d] += p[j] * vh[d];
      }
    }
  }
}

// ----------------------------------------------------------------- metrics

// Cosine similarity of two vectors. This is the number every quality gate in
// this project is expressed in, because a retrieval model's output is compared
// against another implementation of the same model, not against ground truth.
inline double Cosine(const float* a, const float* b, int n) {
  double ab = 0.0, aa = 0.0, bb = 0.0;
  for (int i = 0; i < n; ++i) {
    ab += static_cast<double>(a[i]) * b[i];
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
  }
  if (aa <= 0.0 || bb <= 0.0) return 0.0;
  return ab / (std::sqrt(aa) * std::sqrt(bb));
}

inline double MaxAbsDiff(const float* a, const float* b, int n) {
  double worst = 0.0;
  for (int i = 0; i < n; ++i) worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
  return worst;
}

}  // namespace orderit::reference