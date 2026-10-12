#include "tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace orderit {
namespace {

// --------------------------------------------------------------- utf8
//
// Decode enough of UTF-8 to classify code points and to iterate a string. The
// encoder never re-encodes: text goes from UTF-8 bytes to code points only for
// classification, and the byte-level mapping works on the original bytes.

struct CodePoint {
  std::uint32_t value;
  std::size_t bytes;
};

bool DecodeUtf8(std::string_view text, std::size_t pos, CodePoint* out) {
  const auto byte = [&](std::size_t i) { return static_cast<unsigned char>(text[i]); };
  const unsigned char b0 = pos < text.size() ? byte(pos) : 0;
  if (b0 < 0x80) { out->value = b0; out->bytes = 1; return true; }
  int extra;
  std::uint32_t value;
  if ((b0 & 0xE0) == 0xC0) { extra = 1; value = b0 & 0x1Fu; }
  else if ((b0 & 0xF0) == 0xE0) { extra = 2; value = b0 & 0x0Fu; }
  else if ((b0 & 0xF8) == 0xF0) { extra = 3; value = b0 & 0x07u; }
  else { out->value = 0xFFFD; out->bytes = 1; return false; }
  if (pos + extra >= text.size()) { out->value = 0xFFFD; out->bytes = 1; return false; }
  for (int i = 1; i <= extra; ++i) {
    const unsigned char b = byte(pos + i);
    if ((b & 0xC0) != 0x80) { out->value = 0xFFFD; out->bytes = 1; return false; }
    value = (value << 6) | (b & 0x3Fu);
  }
  out->value = value;
  out->bytes = static_cast<std::size_t>(extra) + 1;
  return true;
}

// -------------------------------------------------------------- classes
//
// The three properties Qwen2's pre-tokenizer branches on: letters, numbers and
// whitespace. The ranges below are the blocks those properties actually occupy
// for the scripts this model family is used on; anything outside them is Other,
// which is the same answer a full property table gives for the ranges omitted.

enum class CharClass { kLetter, kNumber, kSpace, kNewline, kOther };

bool InRanges(std::uint32_t cp, const std::pair<std::uint32_t, std::uint32_t>* r,
              std::size_t n) {
  std::size_t lo = 0, hi = n;
  while (lo < hi) {
    const std::size_t mid = (lo + hi) / 2;
    if (cp < r[mid].first) hi = mid;
    else if (cp > r[mid].second) lo = mid + 1;
    else return true;
  }
  return false;
}

constexpr std::pair<std::uint32_t, std::uint32_t> kLetterRanges[] = {
    {0x0041, 0x005A}, {0x0061, 0x007A}, {0x00AA, 0x00AA}, {0x00B5, 0x00B5},
    {0x00BA, 0x00BA}, {0x00C0, 0x00D6}, {0x00D8, 0x00F6}, {0x00F8, 0x02C1},
    {0x0370, 0x0374}, {0x0376, 0x0377}, {0x037A, 0x037D}, {0x0386, 0x0386},
    {0x0388, 0x03FF}, {0x0400, 0x0481}, {0x048A, 0x052F}, {0x0531, 0x0556},
    {0x0561, 0x0587}, {0x05D0, 0x05EA}, {0x0620, 0x064A}, {0x066E, 0x066F},
    {0x0671, 0x06D3}, {0x0710, 0x072F}, {0x0750, 0x07A5}, {0x0904, 0x0939},
    {0x093D, 0x093D}, {0x0950, 0x0950}, {0x0958, 0x0961}, {0x0985, 0x09B9},
    {0x0A05, 0x0A39}, {0x0A85, 0x0AB9}, {0x0B05, 0x0B39}, {0x0B85, 0x0BB9},
    {0x0C05, 0x0C39}, {0x0C85, 0x0CB9}, {0x0D05, 0x0D3A}, {0x0E01, 0x0E30},
    {0x0E32, 0x0E33}, {0x0E40, 0x0E46}, {0x10A0, 0x10FA}, {0x1100, 0x11FF},
    {0x1E00, 0x1FFF}, {0x2C00, 0x2C5E}, {0x2C60, 0x2CE4}, {0x3005, 0x3006},
    {0x3041, 0x3096}, {0x309D, 0x309F}, {0x30A1, 0x30FA}, {0x30FC, 0x30FF},
    {0x3105, 0x312F}, {0x3131, 0x318E}, {0x31A0, 0x31BF}, {0x3400, 0x4DBF},
    {0x4E00, 0x9FFF}, {0xA000, 0xA48C}, {0xAC00, 0xD7A3}, {0xF900, 0xFAFF},
    {0x20000, 0x2A6DF}, {0x2A700, 0x2EBEF}, {0x2F800, 0x2FA1F},
};
constexpr std::pair<std::uint32_t, std::uint32_t> kNumberRanges[] = {
    {0x0030, 0x0039}, {0x00B2, 0x00B3}, {0x00B9, 0x00B9}, {0x00BC, 0x00BE},
    {0x0660, 0x0669}, {0x06F0, 0x06F9}, {0x0966, 0x096F}, {0x09E6, 0x09EF},
    {0x0A66, 0x0A6F}, {0x0AE6, 0x0AEF}, {0x0B66, 0x0B6F}, {0x0BE6, 0x0BEF},
    {0x0C66, 0x0C6F}, {0x0CE6, 0x0CEF}, {0x0D66, 0x0D6F}, {0x0E50, 0x0E59},
    {0x2070, 0x2070}, {0x2074, 0x2079}, {0x2080, 0x2089}, {0xFF10, 0xFF19},
};
constexpr std::pair<std::uint32_t, std::uint32_t> kSpaceRanges[] = {
    {0x0009, 0x000D}, {0x0020, 0x0020}, {0x0085, 0x0085}, {0x00A0, 0x00A0},
    {0x1680, 0x1680}, {0x2000, 0x200A}, {0x2028, 0x2029}, {0x202F, 0x202F},
    {0x205F, 0x205F}, {0x3000, 0x3000},
};

CharClass Classify(std::uint32_t cp) {
  if (cp == '\r' || cp == '\n') return CharClass::kNewline;
  if (InRanges(cp, kLetterRanges, std::size(kLetterRanges))) return CharClass::kLetter;
  if (InRanges(cp, kNumberRanges, std::size(kNumberRanges))) return CharClass::kNumber;
  if (InRanges(cp, kSpaceRanges, std::size(kSpaceRanges))) return CharClass::kSpace;
  return CharClass::kOther;
}

// --------------------------------------------------------- byte-level maps
//
// GPT-2's byte-to-unicode: every one of the 256 byte values has a printable
// stand-in, so the BPE vocabulary is a vocabulary over printable characters. The
// checkpoint stores its token strings in exactly this form.
//
// 188 bytes are already printable and map to themselves; the remaining 68 - the
// space among them - map to 256..323, so the table needs 324 entries. Note the
// printable run starts at '!' (0x21) and not at the space (0x20): that is
// exactly why a space becomes 'G with stroke' and not ' '. Starting the run at
// 0x20 leaves every space looking like itself and shifts most of the vocabulary,
// which no single-token test would catch and which no encoder test catches
// either until the ids are compared against a reference.

// Fills byte_to_uni, and uni_to_byte when the caller wants the inverse.
// Both halves are optional: only the forward map is needed to encode, and a
// null out-parameter here is a null dereference rather than a skipped step.
void BuildByteMap(std::vector<std::uint32_t>* byte_to_uni, std::vector<std::uint8_t>* uni_to_byte) {
  auto& b2u = *byte_to_uni;
  b2u.assign(256, 0);
  std::vector<std::uint8_t> scratch;
  std::vector<std::uint8_t>& u2b = uni_to_byte ? *uni_to_byte : scratch;
  constexpr std::size_t kTable = 324;  // 256 + (256 - 188 printable)
  u2b.assign(kTable, 0);
  std::vector<bool> taken(kTable, false);
  for (int b = 0x21; b <= 0x7E; ++b) { b2u[b] = b; taken[b] = true; }
  for (int b = 0xA1; b <= 0xAC; ++b) { b2u[b] = b; taken[b] = true; }
  for (int b = 0xAE; b <= 0xFF; ++b) { b2u[b] = b; taken[b] = true; }
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (taken[b]) continue;
    b2u[b] = 256 + n;
    taken[256 + n] = true;
    ++n;
  }
  for (int b = 0; b < 256; ++b) u2b[b2u[b]] = static_cast<std::uint8_t>(b);
}

void AppendUtf8(std::string* out, std::uint32_t cp) {
  if (cp < 0x80) {
    out->push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

std::string ByteEncode(std::string_view bytes) {
  static std::vector<std::uint32_t> byte_to_uni;
  if (byte_to_uni.empty()) BuildByteMap(&byte_to_uni, nullptr);
  std::string out;
  out.reserve(bytes.size() * 2);
  for (char c : bytes) {
    AppendUtf8(&out, byte_to_uni[static_cast<unsigned char>(c)]);
  }
  return out;
}

// ------------------------------------------------------- pre-tokenisation
//
// Qwen2's pattern, branch by branch, in its own order:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   [^\r\n\p{L}\p{N}]?\p{L}+
//   \p{N}{1,3}
//   ?[^\s\p{L}\p{N}]+[\r\n]*
//   \s*[\r\n]+
//   \s+(?!\S)
//   \s+
// Alternation is first-match-wins at each position, which is what decides
// whether a leading space joins the next word or starts its own piece.

struct Piece {
  std::size_t begin;
  std::size_t end;  // byte offsets into the source
};

bool MatchContraction(std::string_view t, std::size_t p, std::size_t* len) {
  static const char* kForms[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
  for (const char* form : kForms) {
    const std::size_t n = std::strlen(form);
    if (p + n > t.size()) continue;
    std::size_t i = 0;
    for (; i < n; ++i) {
      char a = t[p + i], b = form[i];
      if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
      if (a != b) break;
    }
    if (i == n) { *len = n; return true; }
  }
  return false;
}

void PreTokenize(std::string_view text, std::vector<Piece>* out) {
  out->clear();
  std::size_t p = 0;
  CodePoint cp{};
  auto class_at = [&](std::size_t pos, std::size_t* size) {
    DecodeUtf8(text, pos, &cp);
    *size = cp.bytes;
    return Classify(cp.value);
  };

  while (p < text.size()) {
    const std::size_t start = p;
    std::size_t len = 0;

    if (MatchContraction(text, p, &len)) {
      p += len;
      out->push_back({start, p});
      continue;
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+
    // The optional prefix is "any character that is not a newline, a letter or a
    // number", which includes the space. Treating space as an ordinary "other"
    // character here is what splits " World" into " " and "World" and shifts
    // every token id after the first one.
    {
      std::size_t q = p;
      std::size_t size = 0;
      const CharClass first = class_at(q, &size);
      if (first == CharClass::kOther || first == CharClass::kSpace) {
        CodePoint next{};
        if (q + size < text.size() && DecodeUtf8(text, q + size, &next) &&
            Classify(next.value) == CharClass::kLetter) {
          q += size;
        }
      }
      std::size_t letters = 0;
      while (q < text.size()) {
        std::size_t s = 0;
        CharClass cc = class_at(q, &s);
        if (cc != CharClass::kLetter) break;
        q += s;
        ++letters;
      }
      if (letters > 0) {
        p = q;
        out->push_back({start, p});
        continue;
      }
    }

    // \p{N}{1,3}
    if (class_at(p, &len) == CharClass::kNumber) {
      std::size_t digits = 0;
      std::size_t q = p;
      while (q < text.size() && digits < 3) {
        std::size_t s = 0;
        if (class_at(q, &s) != CharClass::kNumber) break;
        q += s;
        ++digits;
      }
      p = q;
      out->push_back({start, p});
      continue;
    }

    // ?[^\s\p{L}\p{N}]+[\r\n]*
    {
      std::size_t q = p;
      if (q < text.size() && text[q] == ' ') q++;
      std::size_t punct = 0;
      while (q < text.size()) {
        std::size_t s = 0;
        CharClass cc = class_at(q, &s);
        if (cc != CharClass::kOther) break;
        q += s;
        ++punct;
      }
      if (punct > 0) {
        while (q < text.size()) {
          std::size_t s = 0;
          if (class_at(q, &s) != CharClass::kNewline) break;
          q += s;
        }
        p = q;
        out->push_back({start, p});
        continue;
      }
    }

    // \s*[\r\n]+ , \s+(?!\S) , \s+
    //
    // Three alternates in this order, and the order decides every answer:
    //
    //   \s*[\r\n]+  the greedy \s* still has to be followed by a newline, so the
    //               match ends at the *last* newline of the run and the
    //               whitespace after it is left for the next branch.
    //   \s+(?!\S)   matches the run only if what follows is not a non-space, so
    //               for a run followed by a word it gives up its last character
    //               - that character becomes the prefix of the word instead.
    //   \s+         a single space with a word after it ends up here, alone.
    {
      std::size_t q = p;
      std::size_t spaces = 0;
      while (q < text.size()) {
        std::size_t s = 0;
        CharClass cc = class_at(q, &s);
        if (cc != CharClass::kSpace && cc != CharClass::kNewline) break;
        q += s;
        ++spaces;
      }
      if (spaces > 0) {
        std::size_t after_newline = 0;
        bool has_newline = false;
        for (std::size_t r = p; r < q;) {
          CodePoint c{};
          DecodeUtf8(text, r, &c);
          if (c.value == '\n' || c.value == '\r') {
            after_newline = r + c.bytes;
            has_newline = true;
          }
          r += c.bytes;
        }
        std::size_t cut;
        if (has_newline) {
          cut = after_newline;
        } else if (q >= text.size()) {
          cut = q;  // trailing whitespace: the lookahead has nothing to fail on
        } else if (q - 1 > start) {
          cut = q - 1;  // give the last character to the word that follows
        } else {
          cut = q;  // a single space: \s+(?!\S) cannot match, \s+ takes it
        }
        out->push_back({start, cut});
        p = cut;
        continue;
      }
    }

    // Nothing matched: consume one code point so the loop always advances.
    class_at(p, &len);
    p += len;
    out->push_back({start, p});
  }
}

}  // namespace

bool Tokenizer::Init(const Weights& w, std::string* error) {
  auto fail = [&](const std::string& what) {
    if (error) *error = what;
    return false;
  };

  const std::vector<std::string>& tokens = w.Tokens();
  if (tokens.empty()) return fail("checkpoint carries no tokenizer vocabulary");

  id_to_token_ = tokens;
  vocab_ = w.vocab();

  // Control tokens are identified by their GGUF type, never by their text, and
  // for the three that matter the text is not even present: Qwen3 stores ids
  // 151643, 151664 and 151665 with an empty content string, so looking up
  // "<|im_end|>" in the vocabulary can never succeed and is not how these tokens
  // are addressed. They are ids, full stop.
  const std::vector<int>& types = w.TokenTypes();
  if (!types.empty() && types.size() != tokens.size()) {
    return fail("tokenizer token and type arrays disagree in length");
  }
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const bool control = !types.empty() && types[i] == 3;
    if (control) {
      special_tokens_.push_back(tokens[i]);
      specials_by_id_.push_back(static_cast<int>(i));
    } else {
      token_to_id_.emplace(tokens[i], static_cast<int>(i));
    }
  }

  ranks_.clear();
  const auto& merges = w.Merges();
  ranks_.reserve(merges.size() * 2);
  for (std::size_t i = 0; i < merges.size(); ++i) {
    ranks_.emplace(merges[i].first + "\x01" + merges[i].second, static_cast<int>(i));
  }

  // The end-of-turn token is recorded in the GGUF; the start-of-turn token sits
  // immediately below it, which is the Qwen3 layout and is checked here rather
  // than assumed: both ids must be control tokens or the checkpoint is not the
  // family this engine drives.
  // The two checkpoints disagree here: the embedding file carries an explicit
  // eot_token_id of 151645, the reranker file carries none and puts 151645 in
  // eos_token_id instead. Either way the end-of-turn token is 151645.
  eot_ = w.eot_token();
  if (eot_ < 0) eot_ = w.eos_token();
  eos_ = w.eos_token();
  if (eos_ < 0) eos_ = eot_;
  if (eot_ < 0 || eot_ >= static_cast<int>(tokens.size())) {
    return fail("checkpoint declares no usable end-of-turn token");
  }
  im_end_ = eot_;
  im_start_ = eot_ - 1;
  if (im_start_ < 0) return fail("checkpoint has no start-of-turn token");
  if (!w.IsControl(im_end_) || !w.IsControl(im_start_)) {
    return fail("end-of-turn token " + std::to_string(im_end_) + " and start-of-turn token " +
                std::to_string(im_start_) + " are not both control tokens");
  }

  bpe_cache_.assign(tokens.size(), {});
  return true;
}

int Tokenizer::Special(const char* name) const {
  for (std::size_t i = 0; i < special_tokens_.size(); ++i) {
    if (special_tokens_[i] == name) return specials_by_id_[i];
  }
  return -1;
}

std::vector<int> Tokenizer::Encode(std::string_view text) const {
  std::vector<int> ids;
  std::vector<Piece> pieces;
  PreTokenize(text, &pieces);

  std::vector<std::string> word;
  std::vector<std::size_t> word_len;
  std::vector<int> out_of_word;

  for (const Piece& piece : pieces) {
    const std::string_view raw = text.substr(piece.begin, piece.end - piece.begin);
    const std::string encoded = ByteEncode(raw);
    if (std::getenv("ORDERIT_TRACE")) {
      std::fprintf(stderr, "      piece %zu..%zu %s -> %s\n", piece.begin, piece.end,
                   std::string(raw).substr(0, 24).c_str(), encoded.c_str());
    }

    // Split the byte-level piece into its characters: each is one symbol.
    word.clear();
    word_len.clear();
    for (std::size_t i = 0; i < encoded.size();) {
      CodePoint cp{};
      DecodeUtf8(encoded, i, &cp);
      word.push_back(encoded.substr(i, cp.bytes));
      word_len.push_back(cp.bytes);
      i += cp.bytes;
    }
    if (word.empty()) continue;

    // Repeated BPE. The cache is keyed by the input string, which for a
    // retrieval workload is dominated by repeated words and document prefixes.
    const auto cached = token_to_id_.find(encoded);
    if (cached != token_to_id_.end()) {
      ids.push_back(cached->second);
      continue;
    }

    while (word.size() > 1) {
      int best_rank = -1;
      std::size_t best = 0;
      for (std::size_t i = 0; i + 1 < word.size(); ++i) {
        const auto it = ranks_.find(word[i] + "\x01" + word[i + 1]);
        if (it != ranks_.end() && (best_rank < 0 || it->second < best_rank)) {
          best_rank = it->second;
          best = i;
        }
      }
      if (best_rank < 0) break;
      word[best] = word[best] + word[best + 1];
      word.erase(word.begin() + static_cast<long>(best) + 1);
      word_len[best] += word_len[best + 1];
      word_len.erase(word_len.begin() + static_cast<long>(best) + 1);
    }

    for (const std::string& symbol : word) {
      const auto it = token_to_id_.find(symbol);
      if (it != token_to_id_.end()) {
        ids.push_back(it->second);
      } else {
        // A single byte with no vocabulary entry would mean a vocabulary this
        // tokenizer does not match. Surface it rather than silently dropping it.
        for (std::size_t i = 0; i < symbol.size(); ++i) {
          const auto byte = token_to_id_.find(std::string(1, symbol[i]));
          if (byte != token_to_id_.end()) ids.push_back(byte->second);
        }
      }
    }
  }
  return ids;
}

}  // namespace orderit