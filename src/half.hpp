// Half and bfloat16 conversions. These live outside reference.hpp because the
// production dequantiser needs them too: the Q8_0 block scale is an fp16, and
// every 32 weights depend on it.
//
// Both conversions are implemented by hand rather than with compiler
// intrinsics because the device compiler's fp16 round-trip mode is a build
// flag, and a conversion whose rounding depends on a flag is not one this
// project can state a quality tolerance against.
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

namespace orderit {

inline float Fp16ToFp32(std::uint16_t bits) {
  const std::uint32_t sign = (bits >> 15) & 1u;
  const std::uint32_t exp = (bits >> 10) & 0x1fu;
  const std::uint32_t mant = bits & 0x3ffu;
  float value;
  if (exp == 0) {
    // Subnormal: value = mant * 2^-24.
    value = std::ldexp(static_cast<float>(mant), -24);
  } else if (exp == 31) {
    value = mant ? NAN : INFINITY;
  } else {
    // Unbiased exponent with an implicit leading one.
    value = std::ldexp(static_cast<float>(mant + 1024u), static_cast<int>(exp) - 25);
  }
  return sign ? -value : value;
}

inline std::uint16_t Fp32ToFp16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::int32_t exp =
      static_cast<std::int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  const std::uint32_t mant = bits & 0x7fffffu;
  if (((bits >> 23) & 0xffu) == 0xffu) return static_cast<std::uint16_t>(sign | 0x7c00u);  // inf
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  if (exp <= 0) {
    // Subnormal in half, or simply too small to represent.
    if (exp < -10) return static_cast<std::uint16_t>(sign);
    const std::uint32_t full = (mant | 0x800000u) >> (1 - exp + 13);
    return static_cast<std::uint16_t>(sign | (full >> 13));
  }
  // Round to nearest, ties to even.
  const std::uint32_t rounded = mant + 0xfffu + ((mant >> 13) & 1u);
  return static_cast<std::uint16_t>(sign |
                                    (static_cast<std::uint32_t>(exp) << 10) |
                                    (rounded >> 13));
}

inline float Bf16ToFp32(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

inline std::uint16_t Fp32ToBf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if (((bits >> 23) & 0xffu) == 0xffu) {
    return static_cast<std::uint16_t>(bits >> 16);
  }
  // Round to nearest, ties to even, on bit 16.
  const std::uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>(rounded >> 16);
}

}  // namespace orderit