// Minimal test harness. No gtest, no dependencies: a project this size does not
// need a framework, and one that has to compile inside a throwaway container is
// better off with a header.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace orderit::test {

inline int g_failures = 0;
inline int g_checks = 0;
inline const char* g_current = "";

inline void Fail(const char* file, int line, const std::string& what) {
  ++g_failures;
  std::printf("  FAIL %s\n    %s:%d: %s\n", g_current, file, line, what.c_str());
}

inline bool Near(double a, double b, double tol) {
  const double diff = std::fabs(a - b);
  const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
  return diff <= tol * scale;
}

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++::orderit::test::g_checks;                                             \
    if (!(cond)) ::orderit::test::Fail(__FILE__, __LINE__, #cond);           \
  } while (0)

#define CHECK_MSG(cond, msg)                                                 \
  do {                                                                       \
    ++::orderit::test::g_checks;                                             \
    if (!(cond))                                                             \
      ::orderit::test::Fail(__FILE__, __LINE__, std::string(#cond) + " -- " + (msg)); \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                \
  do {                                                                       \
    ++::orderit::test::g_checks;                                             \
    if (!::orderit::test::Near((a), (b), (tol))) {                          \
      char buf_[256];                                                        \
      std::snprintf(buf_, sizeof(buf_), "%s == %s (got %.9g, want %.9g)", #a, \
                    #b, (double)(a), (double)(b));                          \
      ::orderit::test::Fail(__FILE__, __LINE__, buf_);                       \
    }                                                                        \
  } while (0)

#define TEST(name)                                                           \
  static void name();                                                        \
  static void Run_##name() {                                                 \
    ::orderit::test::g_current = #name;                                                       \
    const int before = ::orderit::test::g_failures;                                           \
    name();                                                                  \
    std::printf("  %-44s %s\n", #name,                            \
                ::orderit::test::g_failures == before ? "ok" : "FAILED"); \
  }                                                                          \
  static void name()

struct Registrar {
  using Fn = void (*)();
  Registrar(const char* name, Fn fn) {
    entries().push_back({name, fn});
  }
  struct Entry {
    const char* name;
    Fn fn;
  };
  static std::vector<Entry>& entries() {
    static std::vector<Entry> v;
    return v;
  }
};

#define TEST_REGISTER(name)                                                  \
  static ::orderit::test::Registrar reg_##name(#name, &Run_##name);

inline int RunAll(const char* suite) {
  std::printf("== %s ==\n", suite);
  for (auto& e : Registrar::entries()) e.fn();
  std::printf("-- %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

}  // namespace orderit::test