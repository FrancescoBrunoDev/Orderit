// Reading the weights.
//
// The engine reads GGUF v3, because the checkpoint it is given already is: an
// int8 Q8_0 Qwen3-Embedding-4B exists on this machine and llama.cpp serves it
// today, which makes it both the input format and the independent reference
// the CPU forward pass is graded against.
//
// The file is mapped, never read into a buffer. Two reasons, both measured in
// docs/BASELINE.md: mapping the file and registering it with the GPU aliases the
// same physical pages on both sides, so there is no 3.7 GiB copy at startup; and
// the alternative, a device-side buffer, costs 3-8% of throughput because the
// weights then stream from a copy rather than from the file's own pages.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "config.hpp"

namespace orderit {

// GGML tensor types this reader recognises. Q8_0 is the only quantised one that
// exists in these two checkpoints; the f32 ones are the 145 one-dimensional
// norm vectors.
enum class GgmlType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ8_0 = 8,
};

struct TensorInfo {
  std::string name;
  GgmlType type;
  std::uint64_t n_dims;
  std::uint64_t dims[4];
  std::uint64_t offset;

  std::uint64_t Elements() const {
    std::uint64_t n = 1;
    for (std::uint64_t i = 0; i < n_dims; ++i) n *= dims[i];
    return n;
  }
  // GGUF always writes dims[0] as the contiguous dimension. For a projection
  // that is in_features; for token_embd it is hidden_size and dims[1] is the
  // vocabulary. Getting this backwards reads the wrong elements and produces
  // vectors that are plausible and wrong.
  std::uint64_t Inner() const { return n_dims >= 1 ? dims[0] : 0; }
  std::uint64_t Outer() const { return n_dims >= 2 ? dims[1] : 0; }
};

// The 36 weight matrices of one transformer block, in GGUF's llama.cpp naming.
struct LayerWeights {
  const TensorInfo* attn_norm = nullptr;
  const TensorInfo* attn_q = nullptr;
  const TensorInfo* attn_k = nullptr;
  const TensorInfo* attn_v = nullptr;
  const TensorInfo* attn_output = nullptr;
  const TensorInfo* attn_q_norm = nullptr;
  const TensorInfo* attn_k_norm = nullptr;
  const TensorInfo* ffn_norm = nullptr;
  const TensorInfo* ffn_gate = nullptr;
  const TensorInfo* ffn_up = nullptr;
  const TensorInfo* ffn_down = nullptr;
};

class Weights {
 public:
  ~Weights();
  Weights() = default;
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;

  // Maps the file and parses the header. Returns false and fills `error` rather
  // than throwing: a malformed model file is an expected input, not an
  // exceptional one.
  bool Open(const std::string& path, std::string* error);

  // Faults every page of the mapping in from the host. Without this the first
  // kernel that touches the weights runs at 32 GB/s instead of 227 GB/s, which
  // is a 7x penalty paid by whichever request arrives first.
  void Prefault();

  const TensorInfo* Find(std::string_view name) const;
  const TensorInfo& Required(std::string_view name, const char* what) const;

  // Raw pointer to a tensor's data. Returns nullptr for a quantised tensor asked
  // for as float: the caller must ask for bytes and dequantise, because the
  // bytes are not floats and silently reinterpreting them is how a model ends up
  // producing plausible nonsense.
  const unsigned char* Bytes(const TensorInfo& t) const;
  const float* AsFloat(const TensorInfo& t) const;

  const LayerWeights& Layer(int index) const { return layers_[index]; }
  int layers() const { return static_cast<int>(layers_.size()); }
  const TensorInfo& TokenEmbedding() const { return *token_embd_; }
  const TensorInfo& OutputNorm() const { return *output_norm_; }

  // Metadata this engine actually reads. Everything else in the header is
  // skipped: a Qwen3 block count of something other than 36 means the wrong
  // checkpoint, and every dimension is asserted against config.hpp at load.
  int vocab() const { return vocab_; }
  int context_length() const { return context_length_; }
  int pooling_type() const { return pooling_type_; }
  int eos_token() const { return eos_; }
  int eot_token() const { return eot_; }
  bool add_eos() const { return add_eos_; }
  int bos_token() const { return bos_; }
  const std::vector<std::string>& Tokens() const { return tokens_; }
  // GGUF token types: 1 normal, 3 control, 4 user-defined. Control tokens are
  // the ones plain text may never produce.
  const std::vector<int>& TokenTypes() const { return token_types_; }
  bool IsControl(int id) const {
    return id >= 0 && id < static_cast<int>(token_types_.size()) && token_types_[id] == 3;
  }
  const std::vector<std::pair<std::string, std::string>>& Merges() const { return merges_; }
  float rope_theta() const { return rope_theta_; }
  float rms_eps() const { return rms_eps_; }
  int hidden() const { return hidden_; }
  int intermediate() const { return intermediate_; }
  int q_heads() const { return q_heads_; }
  int kv_heads() const { return kv_heads_; }
  int head_dim() const { return head_dim_; }
  int max_position() const { return max_position_; }
  const std::string& Name() const { return name_; }

 private:
  bool ParseHeader(std::string* error);

  const unsigned char* base_ = nullptr;
  std::size_t size_ = 0;
  int fd_ = -1;
  std::vector<TensorInfo> tensors_;
  std::vector<LayerWeights> layers_;
  const TensorInfo* token_embd_ = nullptr;
  const TensorInfo* output_norm_ = nullptr;
  std::vector<std::string> tokens_;
  std::vector<int> token_types_;
  std::vector<std::pair<std::string, std::string>> merges_;
  std::string name_;
  int vocab_ = 0;
  int hidden_ = 0, intermediate_ = 0;
  int q_heads_ = 0, kv_heads_ = 0, head_dim_ = 0;
  int max_position_ = 0;
  int context_length_ = 0;
  int pooling_type_ = 0;
  // -1 means the checkpoint did not say. Token 0 is a valid id, so 0 cannot
  // double as "absent": the embedding checkpoint declares an eot token and
  // the reranker does not, and both are accepted.
  int eos_ = -1, eot_ = -1, bos_ = -1;
  bool add_eos_ = false;
  float rope_theta_ = 1000000.0f;
  float rms_eps_ = 1e-6f;
};

}  // namespace orderit