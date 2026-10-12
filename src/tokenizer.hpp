// Byte-level BPE over the vocabulary the checkpoint carries in its GGUF.
//
// The pre-tokenizer is Qwen2's, hand-coded rather than run through a regex
// engine: std::regex has no Unicode property classes, and a retrieval engine
// that tokenises differently from the reference on CJK text is not a retrieval
// engine. Seven alternation branches and a compact classifier is the whole of it.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "weights.hpp"

namespace orderit {

class Tokenizer {
 public:
  // Builds from the tokenizer metadata inside the GGUF. Returns false and fills
  // `error` if the vocabulary is not the byte-level BPE these models use.
  bool Init(const Weights& w, std::string* error);

  // Plain text to ids. Does not add BOS or EOS; the caller owns the template
  // because where the end token goes is a property of the task, not the
  // tokenizer.
  std::vector<int> Encode(std::string_view text) const;

  int VocabSize() const { return vocab_; }
  int Special(const char* name) const;
  bool HasSpecial(const char* name) const { return Special(name) >= 0; }

  // The four control tokens this model family is driven with. Qwen3-Embedding
  // and Qwen3-Reranker both wrap their payload in im_start/im_end and read the
  // hidden state at the final position, so the eot token is not decoration: it
  // is the position the vector comes from.
  int ImStart() const { return im_start_; }
  int ImEnd() const { return im_end_; }
  int EndOfText() const { return eot_; }
  // The checkpoint's declared eos_token_id, 151643. Distinct from the
  // end-of-turn marker at 151645 and one below it; llama.cpp appends this one
  // when add_special is on, and the pooled vector is read wherever the prompt
  // actually ends.
  int Eos() const { return eos_; }

 private:
  std::vector<std::vector<std::string>> bpe_cache_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> token_to_id_;
  std::unordered_map<std::string, int> ranks_;
  std::vector<std::string> special_tokens_;  // text -> id
  std::vector<int> specials_by_id_;
  int vocab_ = 0;
  int im_start_ = -1, im_end_ = -1, eot_ = -1, eos_ = -1;
};

}  // namespace orderit