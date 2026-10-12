#include "weights.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace orderit {
namespace {

// GGUF value types. Only the ones these checkpoints actually use are decoded;
// anything else is skipped by its declared width, which keeps the parser honest
// about what it has seen rather than guessing.
enum : std::uint32_t {
  kU8 = 0, kI8, kU16, kI16, kU32, kI32, kF32, kBool, kString, kArray,
  kU64, kI64, kF64,
};

class Cursor {
 public:
  Cursor(const unsigned char* data, std::size_t size) : data_(data), size_(size) {}

  bool Ok() const { return ok_; }
  std::size_t Tell() const { return pos_; }

  template <typename T>
  T Read() {
    T value{};
    if (pos_ + sizeof(T) > size_) {
      ok_ = false;
      return value;
    }
    std::memcpy(&value, data_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return value;
  }

  std::string ReadString() {
    const std::uint64_t len = Read<std::uint64_t>();
    if (!ok_ || pos_ + len > size_) {
      ok_ = false;
      return {};
    }
    std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
    pos_ += len;
    return s;
  }

  // Skips one value of the given type without materialising it.
  void Skip(std::uint32_t type, int depth = 0) {
    switch (type) {
      case kU8: case kI8: case kBool: pos_ += 1; break;
      case kU16: case kI16: pos_ += 2; break;
      case kU32: case kI32: case kF32: pos_ += 4; break;
      case kU64: case kI64: case kF64: pos_ += 8; break;
      case kString: ReadString(); break;
      case kArray: {
        const std::uint32_t elem = Read<std::uint32_t>();
        const std::uint64_t count = Read<std::uint64_t>();
        for (std::uint64_t i = 0; i < count && ok_; ++i) Skip(elem, depth + 1);
        break;
      }
      default: ok_ = false; break;
    }
    if (pos_ > size_) ok_ = false;
  }

 private:
  const unsigned char* data_;
  std::size_t size_;
  std::size_t pos_ = 0;
  bool ok_ = true;
};

std::string Errno(const char* what, const std::string& path) {
  return std::string(what) + " " + path + ": " + std::strerror(errno);
}

}  // namespace

Weights::~Weights() {
  if (base_) ::munmap(const_cast<unsigned char*>(base_), size_);
  if (fd_ >= 0) ::close(fd_);
}

bool Weights::Open(const std::string& path, std::string* error) {
  auto fail = [&](const std::string& what) {
    if (error) *error = what;
    return false;
  };

  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) return fail(Errno("cannot open", path));

  struct stat st {};
  if (::fstat(fd_, &st) != 0) return fail(Errno("cannot stat", path));
  size_ = static_cast<std::size_t>(st.st_size);
  if (size_ < 24) return fail("file is too small to be a GGUF: " + path);

  void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapped == MAP_FAILED) return fail(Errno("cannot mmap", path));
  base_ = static_cast<const unsigned char*>(mapped);
  // The kernel will read this mapping; faulting it in in 2 MiB steps keeps the
  // fault cost off the critical path of a 3.7 GiB model.
  ::madvise(const_cast<void*>(mapped), size_, MADV_SEQUENTIAL);

  return ParseHeader(error);
}

bool Weights::ParseHeader(std::string* error) {
  auto fail = [&](const std::string& what) {
    if (error) *error = what;
    return false;
  };

  Cursor cur(base_, size_);
  if (cur.Read<std::uint32_t>() != 0x46554747u) return fail("not a GGUF file (bad magic)");
  const std::uint32_t version = cur.Read<std::uint32_t>();
  if (version != 3) return fail("unsupported GGUF version " + std::to_string(version));
  const std::uint64_t n_tensors = cur.Read<std::uint64_t>();
  const std::uint64_t n_kv = cur.Read<std::uint64_t>();
  if (!cur.Ok()) return fail("truncated GGUF header");

  for (std::uint64_t i = 0; i < n_kv && cur.Ok(); ++i) {
    const std::string key = cur.ReadString();
    const std::uint32_t type = cur.Read<std::uint32_t>();
    // Only the handful of keys this engine acts on are materialised. Everything
    // else, including the 11 MiB of tokenizer token strings read later, is
    // skipped here by width so the header walk stays cheap.
    const bool wanted = key == "general.name" ||
                        key == "qwen3.block_count" ||
                        key == "qwen3.embedding_length" ||
                        key == "qwen3.feed_forward_length" ||
                        key == "qwen3.attention.head_count" ||
                        key == "qwen3.attention.head_count_kv" ||
                        key == "qwen3.attention.key_length" ||
                        key == "qwen3.attention.value_length" ||
                        key == "qwen3.context_length" ||
                        key == "qwen3.rope.freq_base" ||
                        key == "qwen3.attention.layer_norm_rms_epsilon" ||
                        key == "qwen3.pooling_type" ||
                        key == "tokenizer.ggml.tokens" ||
                        key == "tokenizer.ggml.token_type" ||
                        key == "tokenizer.ggml.merges" ||
                        key == "tokenizer.ggml.eos_token_id" ||
                        key == "tokenizer.ggml.eot_token_id" ||
                        key == "tokenizer.ggml.bos_token_id" ||
                        key == "tokenizer.ggml.add_eos_token";
    if (!wanted) {
      cur.Skip(type);
      continue;
    }
    switch (key[0]) {
      case 'g': name_ = cur.ReadString(); break;
      case 'q': {
        if (key == "qwen3.block_count") layers_.resize(cur.Read<std::uint32_t>());
        else if (key == "qwen3.context_length") { context_length_ = static_cast<int>(cur.Read<std::uint32_t>()); max_position_ = context_length_; }
        else if (key == "qwen3.pooling_type") pooling_type_ = static_cast<int>(cur.Read<std::uint32_t>());
        else if (key == "qwen3.embedding_length") hidden_ = static_cast<int>(cur.Read<std::uint32_t>());
        else if (key == "qwen3.feed_forward_length") intermediate_ = static_cast<int>(cur.Read<std::uint32_t>());
        else if (key == "qwen3.attention.head_count") q_heads_ = static_cast<int>(cur.Read<std::uint32_t>());
        else if (key == "qwen3.attention.head_count_kv") kv_heads_ = static_cast<int>(cur.Read<std::uint32_t>());
        else if (key == "qwen3.attention.key_length") head_dim_ = static_cast<int>(cur.Read<std::uint32_t>());
        else if (key == "qwen3.attention.value_length") { cur.Skip(kI32); }
        else if (key == "qwen3.rope.freq_base") rope_theta_ = cur.Read<float>();
        else rms_eps_ = cur.Read<float>();
        break;
      }
      case 't': {
        if (key == "tokenizer.ggml.tokens") {
          const std::uint32_t elem = cur.Read<std::uint32_t>();
          const std::uint64_t count = cur.Read<std::uint64_t>();
          if (elem != kString) return fail("tokenizer.ggml.tokens is not an array of strings");
          tokens_.resize(count);
          for (std::uint64_t t = 0; t < count && cur.Ok(); ++t) tokens_[t] = cur.ReadString();
        } else if (key == "tokenizer.ggml.token_type") {
          const std::uint32_t elem = cur.Read<std::uint32_t>();
          const std::uint64_t count = cur.Read<std::uint64_t>();
          if (elem != kI32) return fail("tokenizer.ggml.token_type is not an array of int32");
          token_types_.resize(count);
          for (std::uint64_t t = 0; t < count && cur.Ok(); ++t) {
            token_types_[t] = static_cast<int>(cur.Read<std::int32_t>());
          }
        } else if (key == "tokenizer.ggml.merges") {
          const std::uint32_t elem = cur.Read<std::uint32_t>();
          const std::uint64_t count = cur.Read<std::uint64_t>();
          if (elem != kString) return fail("tokenizer.ggml.merges is not an array of strings");
          merges_.reserve(count);
          for (std::uint64_t t = 0; t < count && cur.Ok(); ++t) {
            // Each merge is "left right" separated by one space. Token text can
            // itself contain spaces, so the split is on the first space only.
            const std::string line = cur.ReadString();
            const std::size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            merges_.emplace_back(line.substr(0, sp), line.substr(sp + 1));
          }
        } else if (key == "tokenizer.ggml.eos_token_id") eos_ = static_cast<int>(cur.Read<std::uint32_t>());
        else if (key == "tokenizer.ggml.eot_token_id") eot_ = static_cast<int>(cur.Read<std::uint32_t>());
        else if (key == "tokenizer.ggml.bos_token_id") bos_ = static_cast<int>(cur.Read<std::uint32_t>());
        else add_eos_ = cur.Read<std::uint8_t>() != 0;
        break;
      }
      default: cur.Skip(type); break;
    }
  }
  if (!cur.Ok()) return fail("truncated GGUF metadata");

  tensors_.reserve(n_tensors);
  for (std::uint64_t i = 0; i < n_tensors && cur.Ok(); ++i) {
    TensorInfo t;
    t.name = cur.ReadString();
    t.n_dims = cur.Read<std::uint32_t>();
    if (t.n_dims > 4) return fail("tensor " + t.name + " has more than 4 dimensions");
    for (std::uint64_t d = 0; d < t.n_dims; ++d) t.dims[d] = cur.Read<std::uint64_t>();
    t.type = static_cast<GgmlType>(cur.Read<std::uint32_t>());
    t.offset = cur.Read<std::uint64_t>();
    if (!cur.Ok()) return fail("truncated GGUF tensor table");
    tensors_.push_back(std::move(t));
  }
  if (!cur.Ok()) return fail("truncated GGUF tensor table");

  // The tensor data block starts at the next 32-byte boundary after the table.
  std::size_t data_start = (cur.Tell() + 31u) & ~std::size_t(31);
  if (data_start >= size_) return fail("GGUF has no tensor data");

  // Index by name once, then wire up the per-layer view.
  for (const TensorInfo& t : tensors_) {
    if (t.offset + t.Elements() > size_) {
      return fail("tensor " + t.name + " points past the end of the file");
    }
  }
  token_embd_ = Find("token_embd.weight");
  output_norm_ = Find("output_norm.weight");
  if (!token_embd_) return fail("missing token_embd.weight");
  if (!output_norm_) return fail("missing output_norm.weight");
  vocab_ = static_cast<int>(token_embd_->Outer());

  if (layers_.empty()) {
    // No block_count metadata: count blocks by name instead.
    for (const TensorInfo& t : tensors_) {
      if (t.name.rfind("blk.", 0) == 0) {
        const int index = std::atoi(t.name.c_str() + 4);
        if (index + 1 > static_cast<int>(layers_.size())) layers_.resize(index + 1);
      }
    }
  }
  for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
    LayerWeights& w = layers_[i];
    const std::string p = "blk." + std::to_string(i) + ".";
    w.attn_norm = Find(p + "attn_norm.weight");
    w.attn_q = Find(p + "attn_q.weight");
    w.attn_k = Find(p + "attn_k.weight");
    w.attn_v = Find(p + "attn_v.weight");
    w.attn_output = Find(p + "attn_output.weight");
    w.attn_q_norm = Find(p + "attn_q_norm.weight");
    w.attn_k_norm = Find(p + "attn_k_norm.weight");
    w.ffn_norm = Find(p + "ffn_norm.weight");
    w.ffn_gate = Find(p + "ffn_gate.weight");
    w.ffn_up = Find(p + "ffn_up.weight");
    w.ffn_down = Find(p + "ffn_down.weight");
    if (!w.attn_q || !w.attn_k || !w.attn_v || !w.attn_output ||
        !w.attn_norm || !w.ffn_norm || !w.ffn_gate || !w.ffn_up || !w.ffn_down) {
      return fail("block " + std::to_string(i) + " is missing a projection");
    }
  }

  // Store the data offset in the reserved first tensor's slot rather than in a
  // member, by rebasing every offset once.
  const std::uint64_t base = data_start;
  for (TensorInfo& t : tensors_) t.offset += base;
  return true;
}

void Weights::Prefault() {
  // One read per 4 KiB is the OS minimum fault granularity; reading a float4
  // every 2 MiB is enough to make the pages resident without paying 2 GiB of
  // pointless memory traffic for it.
  volatile float sink = 0.0f;
  const auto* as_float = reinterpret_cast<const float*>(base_);
  const std::size_t floats = size_ / sizeof(float);
  for (std::size_t i = 0; i < floats; i += 512) sink += as_float[i];
  (void)sink;
}

const TensorInfo* Weights::Find(std::string_view name) const {
  for (const TensorInfo& t : tensors_) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

const TensorInfo& Weights::Required(std::string_view name, const char* what) const {
  const TensorInfo* t = Find(name);
  if (!t) throw std::runtime_error(std::string("missing tensor ") + std::string(name) + " for " + what);
  return *t;
}

const unsigned char* Weights::Bytes(const TensorInfo& t) const {
  return base_ + t.offset;
}

const float* Weights::AsFloat(const TensorInfo& t) const {
  if (t.type != GgmlType::kF32) return nullptr;
  return reinterpret_cast<const float*>(base_ + t.offset);
}

}  // namespace orderit