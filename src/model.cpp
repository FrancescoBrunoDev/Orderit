#include "model.hpp"

#include <algorithm>
#include <cstring>

#include "reference.hpp"

namespace orderit {

bool Model::Load(const std::string& path, std::string* error) {
  auto fail = [&](const std::string& what) {
    if (error) *error = what;
    return false;
  };

  weights_ = std::make_unique<Weights>();
  if (!weights_->Open(path, error)) return false;

  // The geometry comes from the file. What is asserted is the family: the
  // invariants both checkpoints share, plus internal consistency between the
  // declared dimensions and the tensors actually present. A checkpoint that
  // disagrees is refused here rather than producing a plausible and wrong
  // embedding three hours later.
  config_.layers = weights_->layers();
  config_.hidden = weights_->hidden();
  config_.intermediate = weights_->intermediate();
  config_.q_heads = weights_->q_heads();
  config_.kv_heads = weights_->kv_heads();
  config_.head_dim = weights_->head_dim();
  config_.vocab = weights_->vocab();
  config_.max_position = weights_->max_position();
  config_.rope_theta = weights_->rope_theta();
  config_.rms_norm_eps = weights_->rms_eps();
  if (!config_.IsQwen3Family()) {
    return fail("not a Qwen3 dense checkpoint: " + config_.Describe() + " in " + path);
  }

  const auto shape_ok = [&](const TensorInfo& t, int inner, int outer, const char* what) {
    if (static_cast<int>(t.Inner()) == inner && static_cast<int>(t.Outer()) == outer) return true;
    return fail(std::string(what) + " in block 0 is [" + std::to_string(t.Inner()) + ", " +
                std::to_string(t.Outer()) + "], expected [" + std::to_string(inner) + ", " +
                std::to_string(outer) + "]: " + path);
  };
  if (static_cast<int>(weights_->TokenEmbedding().Inner()) != config_.hidden) {
    return fail("token_embd width does not match embedding_length");
  }
  const LayerWeights& first = weights_->Layer(0);
  if (!shape_ok(*first.attn_q, config_.hidden, config_.q_width(), "q_proj")) return false;
  if (!shape_ok(*first.attn_k, config_.hidden, config_.kv_width(), "k_proj")) return false;
  if (!shape_ok(*first.attn_v, config_.hidden, config_.kv_width(), "v_proj")) return false;
  if (!shape_ok(*first.attn_output, config_.q_width(), config_.hidden, "o_proj")) return false;
  if (!shape_ok(*first.ffn_gate, config_.hidden, config_.intermediate, "ffn_gate")) return false;
  if (!shape_ok(*first.ffn_up, config_.hidden, config_.intermediate, "ffn_up")) return false;
  if (!shape_ok(*first.ffn_down, config_.intermediate, config_.hidden, "ffn_down")) return false;

  // pooling_type is a hint to llama.cpp about how to reduce a sequence, not a
  // statement about the architecture: the embedding checkpoint declares LAST (3)
  // and the reranker declares RANK (4). Both read the last position of the final
  // hidden state; one normalises it and the other scores it.
  const int pooling = weights_->pooling_type();
  if (pooling != 0 && pooling != 3 && pooling != 4) {
    return fail("pooling_type " + std::to_string(pooling) +
                " is neither NONE, LAST nor RANK, so this is not one of the "
                "checkpoints this engine drives");
  }

  tokenizer_ = std::make_unique<Tokenizer>();
  if (!tokenizer_->Init(*weights_, error)) return false;

  name_ = weights_->Name().empty() ? path : weights_->Name();
  return true;
}

void Model::GemmCpu(const float* x, const TensorInfo& weight, float* y, int tokens,
                    int out_features, int in_features) {
  // Dequantise the whole matrix once, then every token in the batch reads it.
  // Doing it inside the dot product instead would multiply the dequantisation
  // cost by the token count and turn a 25-second check into a 25-minute one.
  w_.resize(static_cast<std::size_t>(out_features) * in_features);
  const unsigned char* src = weights_->Bytes(weight);
  const int row_bytes = Config::RowBytes(in_features);
  const int blocks = in_features / Config::kQuantBlock;
  for (int row = 0; row < out_features; ++row) {
    const unsigned char* p = src + static_cast<std::size_t>(row) * row_bytes;
    float* out = w_.data() + static_cast<std::size_t>(row) * in_features;
    for (int b = 0; b < blocks; ++b) {
      const reference::Q8Block block =
          reference::ReadQ8Block(p + static_cast<std::size_t>(b) * Config::kQuantBytes);
      float* dst = out + static_cast<std::size_t>(b) * Config::kQuantBlock;
      for (int i = 0; i < Config::kQuantBlock; ++i) dst[i] = block.scale * block.q[i];
    }
  }

  for (int t = 0; t < tokens; ++t) {
    const float* row = x + static_cast<std::size_t>(t) * in_features;
    float* out = y + static_cast<std::size_t>(t) * out_features;
    for (int r = 0; r < out_features; ++r) {
      const float* w_row = w_.data() + static_cast<std::size_t>(r) * in_features;
      float acc = 0.0f;
      for (int i = 0; i < in_features; ++i) acc += row[i] * w_row[i];
      out[r] = acc;
    }
  }
}

void Model::ForwardCpu(const std::vector<int>& ids, std::vector<float>* out) {
  const int tokens = static_cast<int>(ids.size());
  const int hidden = config_.hidden;
  const int width = config_.q_width();
  const int kv_width = config_.kv_width();
  const int ffn = config_.intermediate;

  hidden_.assign(static_cast<std::size_t>(tokens) * hidden, 0.0f);
  normed_.resize(static_cast<std::size_t>(tokens) * hidden);
  q_.resize(static_cast<std::size_t>(tokens) * width);
  k_.resize(static_cast<std::size_t>(tokens) * kv_width);
  v_.resize(static_cast<std::size_t>(tokens) * kv_width);
  attn_.resize(static_cast<std::size_t>(tokens) * width);
  gate_.resize(static_cast<std::size_t>(tokens) * ffn);
  up_.resize(static_cast<std::size_t>(tokens) * ffn);
  act_.resize(static_cast<std::size_t>(tokens) * ffn);

  // Token embedding: a gather of T rows out of a 2560 x vocab matrix. The row is
  // dequantised here and never again.
  const TensorInfo& emb = weights_->TokenEmbedding();
  std::vector<float> row;  // one dequantised token row, reused for every position
  for (int t = 0; t < tokens; ++t) {
    Row(emb, ids[t], &row);
    std::copy(row.begin(), row.end(), hidden_.data() + static_cast<std::size_t>(t) * hidden);
  }

  for (int layer = 0; layer < config_.layers; ++layer) {
    const LayerWeights& lw = weights_->Layer(layer);

    for (int t = 0; t < tokens; ++t) {
      const float* src = hidden_.data() + static_cast<std::size_t>(t) * hidden;
      const reference::Vector n =
          reference::RmsNorm(src, weights_->AsFloat(*lw.attn_norm), hidden, config_.rms_norm_eps);
      std::copy(n.begin(), n.end(), normed_.data() + static_cast<std::size_t>(t) * hidden);
    }

    GemmCpu(normed_.data(), *lw.attn_q, q_.data(), tokens, width, hidden);
    GemmCpu(normed_.data(), *lw.attn_k, k_.data(), tokens, kv_width, hidden);
    GemmCpu(normed_.data(), *lw.attn_v, v_.data(), tokens, kv_width, hidden);

    // Per-head RMSNorm on q and k, then RoPE at this position, in place over the
    // token-major layout the GPU kernels will use.
    for (int t = 0; t < tokens; ++t) {
      for (int h = 0; h < config_.q_heads; ++h) {
        float* head = q_.data() + static_cast<std::size_t>(t) * width +
                      static_cast<std::size_t>(h) * config_.head_dim;
        const reference::Vector n = reference::RmsNorm(
            head, weights_->AsFloat(*lw.attn_q_norm), config_.head_dim, config_.rms_norm_eps);
        std::copy(n.begin(), n.end(), head);
        reference::ApplyRope(head, config_.head_dim, t, config_.rope_theta);
      }
      for (int h = 0; h < config_.kv_heads; ++h) {
        float* head = k_.data() + static_cast<std::size_t>(t) * kv_width +
                      static_cast<std::size_t>(h) * config_.head_dim;
        const reference::Vector n = reference::RmsNorm(
            head, weights_->AsFloat(*lw.attn_k_norm), config_.head_dim, config_.rms_norm_eps);
        std::copy(n.begin(), n.end(), head);
        reference::ApplyRope(head, config_.head_dim, t, config_.rope_theta);
      }
    }

    reference::Attention(q_.data(), k_.data(), v_.data(), attn_.data(), config_.q_heads,
                         config_.kv_heads, config_.head_dim, tokens);
    GemmCpu(attn_.data(), *lw.attn_output, gate_.data(), tokens, hidden, width);
    for (int i = 0; i < tokens * hidden; ++i) hidden_[i] += gate_[i];

    for (int t = 0; t < tokens; ++t) {
      const float* src = hidden_.data() + static_cast<std::size_t>(t) * hidden;
      const reference::Vector n =
          reference::RmsNorm(src, weights_->AsFloat(*lw.ffn_norm), hidden, config_.rms_norm_eps);
      std::copy(n.begin(), n.end(), normed_.data() + static_cast<std::size_t>(t) * hidden);
    }
    GemmCpu(normed_.data(), *lw.ffn_gate, gate_.data(), tokens, ffn, hidden);
    GemmCpu(normed_.data(), *lw.ffn_up, up_.data(), tokens, ffn, hidden);
    for (int t = 0; t < tokens; ++t) {
      const std::size_t base = static_cast<std::size_t>(t) * ffn;
      const reference::Vector a = reference::Swiglu(gate_.data() + base, up_.data() + base, ffn);
      std::copy(a.begin(), a.end(), act_.data() + base);
    }
    GemmCpu(act_.data(), *lw.ffn_down, up_.data(), tokens, hidden, ffn);
    for (int i = 0; i < tokens * hidden; ++i) hidden_[i] += up_[i];
  }

  out->resize(static_cast<std::size_t>(tokens) * hidden);
  for (int t = 0; t < tokens; ++t) {
    const float* src = hidden_.data() + static_cast<std::size_t>(t) * hidden;
    const reference::Vector n =
        reference::RmsNorm(src, weights_->AsFloat(weights_->OutputNorm()), hidden, config_.rms_norm_eps);
    std::copy(n.begin(), n.end(), out->data() + static_cast<std::size_t>(t) * hidden);
  }
}

void Model::Row(const TensorInfo& tensor, int row, std::vector<float>* out) const {
  const std::size_t width = static_cast<std::size_t>(tensor.Inner());
  out->resize(width);
  if (tensor.type == GgmlType::kF32) {
    const float* src = weights_->AsFloat(tensor) + static_cast<std::size_t>(row) * width;
    std::copy(src, src + width, out->begin());
    return;
  }
  const unsigned char* src =
      weights_->Bytes(tensor) + static_cast<std::size_t>(row) * Config::RowBytes(tensor.Inner());
  const int blocks = static_cast<int>(width) / Config::kQuantBlock;
  for (int b = 0; b < blocks; ++b) {
    const reference::Q8Block block =
        reference::ReadQ8Block(src + static_cast<std::size_t>(b) * Config::kQuantBytes);
    for (int i = 0; i < Config::kQuantBlock; ++i) {
      (*out)[static_cast<std::size_t>(b) * Config::kQuantBlock + i] = block.scale * block.q[i];
    }
  }
}

}  // namespace orderit