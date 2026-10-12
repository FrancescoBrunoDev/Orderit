// The model: one Qwen3 dense block stack, 36 layers, run on the CPU.
//
// This is not the product. It is the definition of the forward pass, written in
// the least clever way that can be checked by reading it, and every GPU kernel
// is graded against it. It exists because the only other implementation this
// project can compare against is llama.cpp serving the same file, and that is a
// different program on a different codebase, not a statement of the model.
//
// It dequantises each weight matrix once per layer into a reusable scratch
// buffer and then runs every token of the batch against it. The alternative,
// dequantising inside the dot product, multiplies the dequantisation cost by the
// token count and turns a 25-second check into a 25-minute one.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "tokenizer.hpp"
#include "weights.hpp"

namespace orderit {

class Model {
 public:
  bool Load(const std::string& path, std::string* error);

  const Tokenizer& Tokens() const { return *tokenizer_; }
  const Weights& WeightsRef() const { return *weights_; }
  const std::string& Name() const { return name_; }
  const Config& Geometry() const { return config_; }
  int Layers() const { return weights_->layers(); }
  int Vocab() const { return weights_->vocab(); }
  // 3 = last-token pooling (the embedding model), 4 = classifier/rank mode
  // (the reranker). The two have the same backbone and different heads.
  bool IsClassifier() const { return weights_->pooling_type() == 4; }

  // Runs the full stack and writes output_norm(hidden) for every position,
  // row-major [tokens, hidden]. This is what both heads read: the embedding head
  // takes the last row, the reranker takes the last row and scores it.
  void ForwardCpu(const std::vector<int>& ids, std::vector<float>* out);

  // Dequantised weight rows, for heads that need a single row of the tied output
  // matrix rather than a projection.
  void Row(const TensorInfo& tensor, int row, std::vector<float>* out) const;

 private:
  void GemmCpu(const float* x, const TensorInfo& weight, float* y, int tokens,
               int out_features, int in_features);

  std::unique_ptr<Weights> weights_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::string name_;
  Config config_;

  // Scratch, sized once for the largest batch this process has seen.
  std::vector<float> hidden_, normed_, q_, k_, v_, attn_, gate_, up_, act_, w_;
};

}  // namespace orderit