#include "config.hpp"

namespace orderit {

std::string Config::Describe() const {
  return "layers=" + std::to_string(layers) + " hidden=" + std::to_string(hidden) +
         " ffn=" + std::to_string(intermediate) + " heads=" + std::to_string(q_heads) +
         "/" + std::to_string(kv_heads) + " head_dim=" + std::to_string(head_dim) +
         " vocab=" + std::to_string(vocab);
}

}  // namespace orderit
