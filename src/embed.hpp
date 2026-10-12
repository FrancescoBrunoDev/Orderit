// The two heads. Everything above them is a transformer block; these are the
// only two things either model does differently, and the whole difference is in
// this file.
//
// Embedding: last-token pooling of the final hidden state, then L2 normalise.
// Reranking: the same pooled row, scored against two rows of the tied output
// matrix.
//
// The reranker has no classification head of its own. Its 398 tensors are
// embed_tokens, output_norm and 36 x 11 block weights, and tie_word_embeddings
// is true, so the vocabulary matrix is the output projection - and only the two
// rows that spell "yes" and "no" are ever read from it.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "model.hpp"

namespace orderit {

// Wraps a payload in the chat template both models were trained with and
// appends the end token. `instruction` may be empty, in which case no Instruct
// line is emitted.
std::vector<int> WrapPrompt(const Tokenizer& tok, std::string_view instruction,
                            std::string_view payload, bool append_eos = true);

// The last position's row, L2 normalised. Optional `dimensions` truncates first,
// which is the Matryoshka representation these models are trained to support:
// the prefix of a normalised unit vector is a valid lower-dimensional vector, so
// truncation needs no projection and no retraining.
std::vector<float> PoolLast(const std::vector<float>& hidden, int tokens,
                            int hidden_size, int dimensions = 0);

void L2Normalize(std::vector<float>* vector);

// The two vocabulary rows the reranker scores against.
struct RerankerScore {
  float yes = 0.0f;
  float no = 0.0f;
};

RerankerScore ScorePair(const Model& model, const std::vector<float>& pooled,
                        int yes_token, int no_token);

// Softmax over the pair, returned as the probability that the document is
// relevant. A pair softmax is not a softmax over a vocabulary: it is the model
// answering a yes/no question, and the two logits are all that exists.
float RelevanceFromScore(RerankerScore score);

}  // namespace orderit