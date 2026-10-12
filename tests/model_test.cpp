// F2 and F3: the data path, checked against the only independent statement of
// what this checkpoint should produce - llama.cpp reading the same file.
//
//   orderit-test model <model.gguf>            run everything
//   orderit-test model <model.gguf> tokenizer  only the tokenizer fixture
//   orderit-test model <model.gguf> prompt     report cosine for each candidate
//                                              prompt shape, then stop
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "embed.hpp"
#include "harness.hpp"
#include "model.hpp"
#include "reference.hpp"

using orderit::Model;

namespace {

Model* g_model = nullptr;
std::string g_fixture_dir = "tests/fixtures";

std::string Unescape(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '\\' || i + 1 >= text.size()) {
      out.push_back(text[i]);
      continue;
    }
    switch (text[++i]) {
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      case '\\': out.push_back('\\'); break;
      default: out.push_back(text[i]);
    }
  }
  return out;
}

// --------------------------------------------------------------- tokenizer

TEST(TokenizerMatchesTheCheckpointVocabulary) {
  std::ifstream file(g_fixture_dir + "/tokenizer.txt");
  CHECK_MSG(file.good(), "cannot open " + g_fixture_dir + "/tokenizer.txt");
  if (!file.good()) return;

  const orderit::Tokenizer& tok = g_model->Tokens();
  std::string line;
  int cases = 0;
  int mismatches = 0;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    const std::size_t tab = line.rfind('\t');
    if (tab == std::string::npos) continue;
    const std::string input = Unescape(line.substr(0, tab));
    std::vector<int> want;
    {
      std::stringstream ids(line.substr(tab + 1));
      std::string item;
      while (std::getline(ids, item, ',')) {
        if (!item.empty()) want.push_back(std::stoi(item));
      }
    }
    if (std::getenv("ORDERIT_TRACE")) {
      std::printf("    [%d] %.48s\n", cases, input.c_str());
      std::fflush(stdout);
    }
    const std::vector<int> got = tok.Encode(input);
    ++cases;
    if (got != want) {
      ++mismatches;
      if (mismatches <= 8) {
        std::string got_text, want_text;
        for (int id : got) got_text += " " + std::to_string(id);
        for (int id : want) want_text += " " + std::to_string(id);
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "\"%s\"\n      want%s\n      got %s",
                      input.substr(0, 60).c_str(), want_text.c_str(), got_text.c_str());
        std::printf("    case %d: %s\n", cases, buf);
      }
    }
  }
  CHECK_MSG(cases >= 80, "fixture is too small to mean anything: " + std::to_string(cases));
  CHECK_MSG(mismatches == 0, std::to_string(mismatches) + " of " + std::to_string(cases) +
                                 " tokenizations differ from llama.cpp");
}

TEST(TokenizerNeverEmitsControlTokens) {
  const orderit::Tokenizer& tok = g_model->Tokens();
  for (const char* text : {"<|im_start|>", "<|im_end|>", "<|endoftext|>",
                           "a <|im_start|> b <|im_end|> c"}) {
    for (int id : tok.Encode(text)) {
      CHECK_MSG(id < g_model->Vocab(),
                "control token leaked into plain text for: " + std::string(text));
    }
  }
}

// ------------------------------------------------------------------ forward

std::vector<float> ForwardAndPool(const std::vector<int>& ids) {
  std::vector<float> hidden;
  g_model->ForwardCpu(ids, &hidden);
  return orderit::PoolLast(hidden, static_cast<int>(ids.size()), g_model->Geometry().hidden);
}

std::string UnescapeKey(const std::string& text) { return Unescape(text); }

double CosineToLlama(const std::vector<float>& ours, const char* content) {
  // The reference vector is read from a fixture file so the test needs no
  // running server; tools/make_parity_fixture.py writes it from llama.cpp.
  //
  // On a classifier checkpoint there is no pooled embedding to compare: the
  // reranker emits two logits and its parity fixture is a set of scores, which
  // F8 writes. Reporting a low cosine here would be a test failure caused by
  // asking the wrong question, not by the model.
  if (g_model->IsClassifier()) return -4.0;
  std::ifstream file(g_fixture_dir + "/parity.txt");
  if (!file.good()) return -2.0;  // "not measured"
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    const std::size_t tab = line.rfind('\t');
    if (tab == std::string::npos) continue;
    if (UnescapeKey(line.substr(0, tab)) != content) continue;
    // Comma separated, and operator>> does not skip commas: read the fields.
    std::vector<float> theirs;
    {
      std::stringstream rest(line.substr(tab + 1));
      std::string field;
      while (std::getline(rest, field, ',')) {
        if (!field.empty()) theirs.push_back(std::stof(field));
      }
    }
    if (theirs.size() != ours.size()) {
      std::printf("    fixture for \"%s\" has %zu dims, we produced %zu\n", content,
                  theirs.size(), ours.size());
      return -3.0;
    }
    return orderit::reference::Cosine(ours.data(), theirs.data(),
                                      static_cast<int>(ours.size()));
  }
  return -2.0;
}

// One token is the smallest input that still runs the whole stack, and it
// isolates the backbone from the prompt: attention degenerates to a single
// weighted average of one value, so a low cosine here is the arithmetic and a
// high cosine with a longer prompt failing is the tokenisation.
TEST(ForwardMatchesLlamaCppOneToken) {
  const orderit::Tokenizer& tok = g_model->Tokens();
  const std::vector<float> ours = ForwardAndPool({0, tok.Eos()});
  const double cosine = CosineToLlama(ours, "!");
  std::printf("    one token: cosine %s\n",
              cosine < -1.5 ? "no fixture" : std::to_string(cosine).c_str());
  if (cosine > -1.5) CHECK_MSG(cosine > 0.99, "one-token cosine " + std::to_string(cosine));
}

TEST(ForwardMatchesLlamaCpp) {
  // The reference prompt is tokens(text) followed by eos_token_id, which is what
  // llama.cpp builds with add_special on, and the vector is read at that last
  // position. Both halves matter: appending the end-of-turn marker at 151645
  // instead drops the cosine from 0.9999 to 0.71.
  const orderit::Tokenizer& tok = g_model->Tokens();
  std::vector<int> ids = tok.Encode("hello world");
  ids.push_back(tok.Eos());
  const auto start = std::chrono::steady_clock::now();
  const std::vector<float> ours = ForwardAndPool(ids);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  CHECK(ours.size() == static_cast<std::size_t>(g_model->Geometry().hidden));
  std::printf("    forward %d tokens in %.1f s\n", static_cast<int>(ids.size()), seconds);

  const double cosine = CosineToLlama(ours, "hello world");
  if (cosine < -3.5) {
    std::printf("    %s\n", g_model->IsClassifier()
        ? "classifier checkpoint: no pooled embedding to compare (F8 scores)"
        : "no parity fixture; run tools/make_parity_fixture.py");
    return;  // not a failure: the right fixture for this model does not exist yet
  }
  std::printf("    cosine vs llama.cpp: %.6f\n", cosine);
  CHECK_MSG(cosine > 0.999, "cosine " + std::to_string(cosine) + " is below the 0.999 gate");
}

TEST(PromptShapesReportTheirCosine) {
  // Establishes empirically which prompt shape the reference implementation
  // actually runs, rather than assuming one from a paper.
  struct Candidate {
    const char* name;
    std::vector<int> ids;
  };
  const orderit::Tokenizer& tok = g_model->Tokens();
  const std::vector<int> body = tok.Encode("hello world");
  std::vector<Candidate> candidates;
  candidates.push_back({"raw", body});
  // llama.cpp's add_special appends eos_token_id, which is 151643 and is NOT the
  // same token as the end-of-turn marker at 151645. The two are one apart and
  // the pooled vector is read at whichever one the prompt ends with.
  candidates.push_back({"raw+eos", [&] {
    std::vector<int> v = body; v.push_back(tok.Eos()); return v; }()});
  candidates.push_back({"raw+eot", [&] {
    std::vector<int> v = body; v.push_back(tok.EndOfText()); return v; }()});
  candidates.push_back({"chat (instruction template)", [&] {
    return orderit::WrapPrompt(tok, "Given a web search query, retrieve relevant passages "
                                    "that answer the query", "hello world", true); }()});

  for (const Candidate& candidate : candidates) {
    const std::vector<float> ours = ForwardAndPool(candidate.ids);
    const double cosine = CosineToLlama(ours, "hello world");
    char buf[256];
    std::snprintf(buf, sizeof(buf), "    %-10s %2d tokens  cosine %s", candidate.name,
                  static_cast<int>(candidate.ids.size()),
                  cosine < -1.5 ? "no fixture" : std::to_string(cosine).c_str());
    std::printf("%s\n", buf);
    if (cosine > 0.999) {
      // A near-exact match is a contract, not a coincidence: record it.
      orderit::test::g_checks++;
      std::printf("    -> MATCH, this is the reference prompt shape\n");
    }
  }
}

TEST_REGISTER(TokenizerMatchesTheCheckpointVocabulary)
TEST_REGISTER(TokenizerNeverEmitsControlTokens)
// Every case in the parity fixture, not just the short ones. The long ones are
// where an attention bug shows up: a pooling bug hides in a single token and a
// causal-mask bug hides in a three-token prompt.
TEST(EveryParityCaseMatchesLlamaCpp) {
  std::ifstream file(g_fixture_dir + "/parity.txt");
  if (!file.good()) {
    std::printf("    no parity fixture\n");
    return;
  }
  const orderit::Tokenizer& tok = g_model->Tokens();
  std::string line;
  int cases = 0;
  double worst = 1.0;
  std::string worst_case;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    const std::size_t tab = line.rfind('\t');
    if (tab == std::string::npos) continue;
    const std::string text = Unescape(line.substr(0, tab));
    std::vector<float> theirs;
    {
      std::stringstream rest(line.substr(tab + 1));
      std::string field;
      while (std::getline(rest, field, ',')) {
        if (!field.empty()) theirs.push_back(std::stof(field));
      }
    }
    std::vector<int> ids = tok.Encode(text);
    ids.push_back(tok.Eos());
    const std::vector<float> ours = ForwardAndPool(ids);
    const double cosine = orderit::reference::Cosine(ours.data(), theirs.data(),
                                             static_cast<int>(ours.size()));
    ++cases;
    std::printf("    %4d tokens  cosine %.6f  %s\n", static_cast<int>(ids.size()), cosine,
                text.substr(0, 44).c_str());
    if (cosine < worst) { worst = cosine; worst_case = text.substr(0, 44); }
  }
  std::printf("    %d cases, worst cosine %.6f on \"%s\"\n", cases, worst, worst_case.c_str());
  CHECK_MSG(cases >= 5, "fixture has too few cases: " + std::to_string(cases));
  CHECK_MSG(worst > 0.999, "worst cosine " + std::to_string(worst) + " below the 0.999 gate");
}

TEST_REGISTER(EveryParityCaseMatchesLlamaCpp)
TEST_REGISTER(ForwardMatchesLlamaCppOneToken)
TEST_REGISTER(ForwardMatchesLlamaCpp)
TEST_REGISTER(PromptShapesReportTheirCosine)

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s <model.gguf> [tokenizer|prompt|all]\n", argv[0]);
    return 2;
  }
  if (argc >= 3 && std::string(argv[2]) == "tokenizer") g_fixture_dir = "tests/fixtures";
  if (const char* dir = std::getenv("ORDERIT_FIXTURES")) g_fixture_dir = dir;

  Model model;
  std::string error;
  if (!model.Load(argv[1], &error)) {
    std::printf("cannot load %s: %s\n", argv[1], error.c_str());
    return 2;
  }
  g_model = &model;
  std::printf("loaded %s: %d layers, %d vocab, %d tokens in vocabulary\n",
              model.Name().c_str(), model.Layers(), model.Vocab(),
              static_cast<int>(model.Tokens().VocabSize()));

  if (model.IsClassifier()) {
    std::printf("note: classifier checkpoint, pooling_type=4. The backbone and "
                "tokenizer are checked; the head is not, until F8 writes its "
                "score fixture.\n");
  }
  const std::string mode = argc >= 3 ? argv[2] : "all";
  const int rc = orderit::test::RunAll(mode == "all" ? "model (F2/F3)" : mode.c_str());
  return rc;
}