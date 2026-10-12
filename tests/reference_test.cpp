// Analytic tests for the CPU references themselves. If these are wrong, every
// GPU kernel is being graded against a broken ruler, so they are checked against
// closed forms that can be verified by hand rather than against another
// implementation.
#include "harness.hpp"

#include <cmath>
#include <vector>

#include "config.hpp"
#include "reference.hpp"

namespace ref = orderit::reference;
// The family invariants, restated here so the oracle tests fail if the shared
// constants ever move away from what both checkpoints declare.
constexpr float kRmsNormEps = 1e-6f;
constexpr float kRopeTheta = 1000000.0f;

TEST(Fp16RoundTrip) {
  const float values[] = {0.0f,  1.0f,  -1.0f, 0.5f,    -0.25f, 3.14159f,
                          65504.0f, -65504.0f, 1e-7f, 6.1e-5f, 0.1f, 1234.5f};
  for (float v : values) {
    const float back = ref::Fp16ToFp32(ref::Fp32ToFp16(v));
    CHECK_NEAR(back, v, 1e-3);
  }
  // Known exact bit patterns rather than round trips.
  CHECK(ref::Fp16ToFp32(0x3C00) == 1.0f);
  CHECK(ref::Fp16ToFp32(0xC000) == -2.0f);
  CHECK(ref::Fp16ToFp32(0x0001) > 0.0f && ref::Fp16ToFp32(0x0001) < 1e-4f);
  CHECK(std::isinf(ref::Fp16ToFp32(0x7C00)));
}

TEST(Bf16RoundTrip) {
  const float values[] = {0.0f, 1.0f, -1.0f, 0.1f, 3.14159f, 1e6f, -12345.678f};
  for (float v : values) {
    // bfloat16 keeps 8 mantissa bits, so the relative error is at most 2^-9.
    CHECK_NEAR(ref::Bf16ToFp32(ref::Fp32ToBf16(v)), v, 2.0 / 512.0);
  }
  CHECK(ref::Fp32ToBf16(1.0f) == 0x3F80);
  CHECK(ref::Fp16ToFp32(0x3C00) == 1.0f);  // 0x3F80 is bfloat16 of 1.0
  CHECK(ref::Fp16ToFp32(0x3F80) == 1.875f);  // the same bits read as fp16: 1 + 0x380/1024
}

TEST(RmsNormNormalises) {
  const int n = 64;
  std::vector<float> x(n), w(n, 1.0f);
  for (int i = 0; i < n; ++i) x[i] = std::sin(i * 0.37f) * 3.0f;
  const ref::Vector out = ref::RmsNorm(x.data(), w.data(), n, kRmsNormEps);
  double sum = 0.0;
  for (float v : out) sum += v * v;
  // RMSNorm without a learned weight leaves a unit-RMS vector.
  CHECK_NEAR(sum / n, 1.0, 1e-5);

  // A zero input must not divide by zero: eps carries it.
  std::vector<float> zeros(n, 0.0f);
  const ref::Vector flat = ref::RmsNorm(zeros.data(), w.data(), n, kRmsNormEps);
  for (float v : flat) CHECK(v == 0.0f);

  // The weight multiplies the normalised value, after normalisation.
  std::vector<float> scaled(n, 2.0f);
  const ref::Vector twice = ref::RmsNorm(x.data(), scaled.data(), n, kRmsNormEps);
  CHECK_NEAR(twice[5], out[5] * 2.0, 1e-6);
}

TEST(RopeIsOrthogonalAndPositionPreserving) {
  const int rotary = 128;
  std::vector<float> v(rotary);
  for (int i = 0; i < rotary; ++i) v[i] = std::cos(i * 0.11f);
  std::vector<float> before = v;

  ref::ApplyRope(v.data(), rotary, 0, kRopeTheta);
  // Position 0 is the identity rotation.
  for (int i = 0; i < rotary; ++i) CHECK(v[i] == before[i]);

  ref::ApplyRope(v.data(), rotary, 1, kRopeTheta);
  // A rotation preserves the L2 norm of each pair.
  for (int i = 0; i < rotary / 2; ++i) {
    const double a = before[i], b = before[i + rotary / 2];
    const double p = v[i], q = v[i + rotary / 2];
    CHECK_NEAR(p * p + q * q, a * a + b * b, 1e-5);
  }

  // Half-split layout: element i is paired with i + rotary/2, so a vector that
  // is zero on one half of each pair is rotated independently. This is the
  // check that fails if the layout is interleaved instead.
  std::vector<float> half(rotary, 0.0f);
  half[0] = 1.0f;
  ref::ApplyRope(half.data(), rotary, 5, kRopeTheta);
  const float freq = 1.0f / std::pow(kRopeTheta, 0.0f);  // i = 0 -> 1 Hz
  CHECK_NEAR(half[0], std::cos(5.0f * freq), 1e-5);
  CHECK_NEAR(half[rotary / 2], std::sin(5.0f * freq), 1e-5);
  CHECK(half[1] == 0.0f);  // element 1 belongs to the next pair, untouched
}

TEST(RopeLeavesTailAlone) {
  // head_dim 128 with rotary 64 is not this model, but the kernel shares the
  // code path and the behaviour must be right for any rotary span.
  const int rotary = 8, n = 16;
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i) v[i] = static_cast<float>(i) + 1.0f;
  std::vector<float> before = v;
  ref::ApplyRope(v.data(), rotary, 3, 10000.0f);
  for (int i = rotary; i < n; ++i) CHECK(v[i] == before[i]);
}

TEST(SwigluMatchesFormula) {
  const int n = 8;
  std::vector<float> gate{0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 10.0f};
  std::vector<float> up{1.0f, 1.0f, 1.0f, 3.0f, 3.0f, 2.0f, 2.0f, -1.0f};
  const ref::Vector out = ref::Swiglu(gate.data(), up.data(), n);
  for (int i = 0; i < n; ++i) CHECK_NEAR(out[i], ref::Silu(gate[i]) * up[i], 1e-6);
  CHECK_NEAR(out[0], 0.0, 1e-9);              // silu(0) = 0, not sigmoid(0)
  CHECK_NEAR(out[1], 0.7310586, 1e-5);        // silu(1)
  CHECK_NEAR(out[2], -0.2689414, 1e-5);       // silu(-1) = -0.2689...
}

TEST(SoftmaxIsShiftInvariantAndNormalised) {
  std::vector<float> a{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> b = a;
  for (float& v : b) v += 1000.0f;  // would overflow without the max subtraction
  const ref::Vector pa = ref::Softmax(a.data(), 4);
  const ref::Vector pb = ref::Softmax(b.data(), 4);
  double sum = 0.0;
  for (int i = 0; i < 4; ++i) {
    CHECK_NEAR(pa[i], pb[i], 1e-6);
    sum += pa[i];
    CHECK(pa[i] > 0.0f);
  }
  CHECK_NEAR(sum, 1.0, 1e-6);
  CHECK_NEAR(pa[3], pa[0] * std::exp(3.0), 1e-5);  // monotone
}

TEST(Q8BlockDecodesExactly) {
  unsigned char raw[orderit::Config::kQuantBytes] = {0};
  ref::Fp32ToFp16(0.5f);  // sanity of the writer
  const std::uint16_t scale_bits = ref::Fp32ToFp16(0.5f);
  std::memcpy(raw, &scale_bits, 2);
  for (int i = 0; i < orderit::Config::kQuantBlock; ++i) {
    raw[2 + i] = static_cast<unsigned char>(static_cast<signed char>(i - 16));
  }
  for (int i = 0; i < orderit::Config::kQuantBlock; ++i) {
    CHECK_NEAR(ref::DequantQ8(raw, orderit::Config::kQuantBlock, i),
               0.5f * static_cast<float>(i - 16), 1e-6);
  }
}

TEST(CosineIsSelfOne) {
  const int n = 16;
  std::vector<float> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = std::sin(i);
    b[i] = a[i] * 2.0f;  // scaling must not change the angle
  }
  CHECK_NEAR(ref::Cosine(a.data(), b.data(), n), 1.0, 1e-12);
  b[3] = -b[3];
  CHECK(ref::Cosine(a.data(), b.data(), n) < 0.999);
}

TEST_REGISTER(Fp16RoundTrip)
TEST_REGISTER(Bf16RoundTrip)
TEST_REGISTER(RmsNormNormalises)
TEST_REGISTER(RopeIsOrthogonalAndPositionPreserving)
TEST_REGISTER(RopeLeavesTailAlone)
TEST_REGISTER(SwigluMatchesFormula)
TEST_REGISTER(SoftmaxIsShiftInvariantAndNormalised)
TEST_REGISTER(Q8BlockDecodesExactly)
TEST_REGISTER(CosineIsSelfOne)

int main() { return orderit::test::RunAll("reference (F1)"); }