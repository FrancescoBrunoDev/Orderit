#include "embed.hpp"

#include <cmath>

namespace orderit {

std::vector<int> WrapPrompt(const Tokenizer& tok, std::string_view instruction,
                            std::string_view payload, bool append_eos) {
  std::vector<int> ids;
  ids.push_back(tok.ImStart());
  ids.push_back(tok.Encode("user").front());
  ids.push_back(tok.Encode("\n").front());
  if (!instruction.empty()) {
    const std::vector<int> head = tok.Encode("Instruct: " + std::string(instruction));
    ids.insert(ids.end(), head.begin(), head.end());
    ids.push_back(tok.Encode("\n").front());
  }
  const std::vector<int> body = tok.Encode(payload);
  ids.insert(ids.end(), body.begin(), body.end());
  ids.push_back(tok.ImEnd());
  if (append_eos) ids.push_back(tok.EndOfText());
  return ids;
}

void L2Normalize(std::vector<float>* vector) {
  double sum = 0.0;
  for (float v : *vector) sum += static_cast<double>(v) * v;
  if (sum <= 0.0) return;
  const float inv = static_cast<float>(1.0 / std::sqrt(sum));
  for (float& v : *vector) v *= inv;
}

std::vector<float> PoolLast(const std::vector<float>& hidden, int tokens, int hidden_size,
                            int dimensions) {
  const int last = (tokens - 1) * hidden_size;
  const int take = (dimensions > 0 && dimensions < hidden_size) ? dimensions : hidden_size;
  std::vector<float> out(hidden.begin() + last, hidden.begin() + last + take);
  // Normalise after truncation, not before: the truncated vector is the one
  // that gets compared, and normalising the full vector then truncating leaves
  // a vector with norm < 1.
  L2Normalize(&out);
  return out;
}

RerankerScore ScorePair(const Model& model, const std::vector<float>& pooled, int yes_token,
                        int no_token) {
  std::vector<float> yes_row, no_row;
  model.Row(model.WeightsRef().TokenEmbedding(), yes_token, &yes_row);
  model.Row(model.WeightsRef().TokenEmbedding(), no_token, &no_row);
  double yes = 0.0, no = 0.0;
  for (std::size_t i = 0; i < pooled.size(); ++i) {
    yes += static_cast<double>(pooled[i]) * yes_row[i];
    no += static_cast<double>(pooled[i]) * no_row[i];
  }
  return {static_cast<float>(yes), static_cast<float>(no)};
}

float RelevanceFromScore(RerankerScore score) {
  // The two logits differ by tens of units on a confident pair, so the naive
  // exp() overflows before the subtraction on at least one side.
  const float max = score.yes > score.no ? score.yes : score.no;
  const double yes = std::exp(static_cast<double>(score.yes - max));
  const double no = std::exp(static_cast<double>(score.no - max));
  return static_cast<float>(yes / (yes + no));
}

}  // namespace orderit