// Orderit F0 - the roofline of this exact machine, measured with this exact
// ROCm. Every later kernel is judged against these numbers, so they are recorded
// here rather than quoted from gufo, which measured on ROCm 7.2.3.
//
//   ./orderit-bench info     device, memory pools, wave/lane geometry
//   ./orderit-bench mem      streaming read bandwidth vs working set size
//   ./orderit-bench commit   cost of hipHostRegister over an mmap'd file
//   ./orderit-bench wmma     pure int8 and bf16 matrix throughput ceiling
//   ./orderit-bench all
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define HIP_CHECK(expr)                                                      \
  do {                                                                       \
    hipError_t err_ = (expr);                                                \
    if (err_ != hipSuccess) {                                                \
      std::printf("HIP error %s at %s:%d\n", hipGetErrorString(err_),         \
                  __FILE__, __LINE__);                                       \
      return 1;                                                              \
    }                                                                        \
  } while (0)

namespace {

// Fragment types for the three matrix paths this engine may use. Verified
// against clang 23 on gfx1151; this ROCm ships no hip_wmma.h.
using v16h = __attribute__((__vector_size__(32))) _Float16;    // 16 x fp16
using v8f = __attribute__((__vector_size__(32))) float;        // 8 x fp32
using v16i8 = __attribute__((__vector_size__(16))) signed char; // 16 x int8
using v8i = __attribute__((__vector_size__(32))) int;          // 8 x int32

using Clock = std::chrono::steady_clock;

// 16-byte accesses are the widest the hardware takes and are what a
// weight-streaming GEMM actually issues.
constexpr std::size_t kVec = sizeof(float4);

double Seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// ---------------------------------------------------------------- bandwidth
// A streaming read of the whole buffer with a cheap reduction, so the compiler
// cannot drop the loads. float4 gives 16-byte accesses, which is the widest the
// hardware takes and matches what a weight-streaming GEMM actually issues.
__global__ void StreamReadKernel(const float4* __restrict__ data,
                                 std::size_t count, float* __restrict__ out) {
  std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
  const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
  float acc = 0.0f;
  for (; i < count; i += stride) {
    const float4 v = data[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1234.5678f) out[0] = acc;  // never true; defeats dead-code removal
}

int BenchMem() {
  std::printf("\n== streaming read bandwidth ==\n");
  std::printf("%12s %10s %12s %10s\n", "size", "seconds", "GB/s", "vs L3");
  const std::size_t sizes[] = {4u << 20, 16u << 20, 32u << 20, 64u << 20,
                               256u << 20, 1ull << 30, 4ull << 30,
                               8ull << 30};
  const std::size_t available = 96ull << 30;  // stay well inside the 124 GiB pool
  float* d_data = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, sizeof(float)));

  for (std::size_t bytes : sizes) {
    if (bytes > available) continue;
    if (hipMalloc(&d_data, bytes) != hipSuccess) {
      std::printf("%12s skipped (allocation refused)\n", "too large");
      break;
    }
    hipMemset(d_data, 1, bytes);
    const std::size_t count = bytes / kVec;
    const int block = 256;
    const int grid = 20 * 16;  // enough waves to saturate, not enough to thrash

    StreamReadKernel<<<grid, block>>>(reinterpret_cast<const float4*>(d_data),
                                      count, d_out);
    (void)hipDeviceSynchronize();

    const auto t0 = Clock::now();
    for (int rep = 0; rep < 4; ++rep) {
      StreamReadKernel<<<grid, block>>>(reinterpret_cast<const float4*>(d_data),
                                        count, d_out);
    }
    (void)hipDeviceSynchronize();
    const auto t1 = Clock::now();

    const double secs = Seconds(t0, t1) / 4.0;
    const double gbps = (double)bytes / secs / 1e9;
    std::printf("%9zu MiB %10.4f %12.1f %10s\n", bytes >> 20, secs, gbps,
                bytes <= (32u << 20) ? "in L3" : "from RAM");
    fflush(stdout);
    hipFree(d_data);
  }
  hipFree(d_out);
  return 0;
}

// ------------------------------------------------------------------ commit
// The engine reads weights straight out of a file mapping through
// hipHostRegister, so the startup cost is a page commit. This measures it.
int BenchCommit() {
  std::printf("\n== hipHostRegister page-commit cost ==\n");
  const std::size_t bytes = 4ull << 30;  // one int8 Q8_0 model
  const char* path = "/tmp/orderit-bench-weights.bin";
  const int fd = ::open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) {
    std::printf("cannot create %s\n", path);
    return 1;
  }
  if (::ftruncate(fd, (off_t)bytes) != 0) {
    std::printf("cannot size %s\n", path);
    ::close(fd);
    return 1;
  }
  // Touch one page per 2 MiB so the file has content and the commit cost is real.
  std::vector<std::size_t> touched;
  for (std::size_t off = 0; off < bytes; off += (2u << 20)) {
    if (::pwrite(fd, "\0", 1, (off_t)off) != 1) break;
    touched.push_back(off);
  }

  void* mapped = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    std::printf("mmap failed\n");
    ::close(fd);
    return 1;
  }
  std::printf("mapped %.2f GiB, %zu pages prefaulted\n", bytes / 1073741824.0,
              touched.size());

  const auto t0 = Clock::now();
  const hipError_t err = hipHostRegister(mapped, bytes,
                                         hipHostRegisterMapped |
                                             hipHostRegisterReadOnly);
  const auto t1 = Clock::now();
  if (err != hipSuccess) {
    std::printf("hipHostRegister failed: %s\n", hipGetErrorString(err));
    ::munmap(mapped, bytes);
    ::close(fd);
    return 1;
  }
  const double secs = Seconds(t0, t1);
  std::printf("hipHostRegister  %.3f s  ->  %.1f GB/s commit rate\n", secs,
              (double)bytes / secs / 1e9);

  // The device pointer must alias the host pointer: that is the whole point.
  void* device_ptr = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&device_ptr, mapped, 0));
  std::printf("device alias      %s\n", device_ptr == mapped ? "yes" : "NO");

  // Three states of the same buffer, because the difference between them is the
  // whole cold-start story: pages never touched by the host, pages faulted by
  // the host, and pages already read once by the GPU.
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
  const std::size_t probe = 1ull << 30;  // 1 GiB window
  auto read_probe = [&](const char* label) {
    auto t2 = Clock::now();
    StreamReadKernel<<<320, 256>>>(
        reinterpret_cast<const float4*>(mapped), probe / kVec, d_out);
    (void)hipDeviceSynchronize();
    auto t3 = Clock::now();
    std::printf("  %-34s %6.3f s -> %6.1f GB/s\n", label, Seconds(t2, t3),
                (double)probe / Seconds(t2, t3) / 1e9);
  };
  std::printf("GPU reads of a 1 GiB window:\n");
  read_probe("cold, host never touched");

  // Fault the whole mapping in from the host, the way a loader must.
  volatile float sink = 0.0f;
  const float* as_floats = static_cast<const float*>(mapped);
  for (std::size_t i = 0; i < bytes / sizeof(float); i += 512) sink += as_floats[i];
  (void)sink;
  read_probe("after host prefault of the whole map");

  hipHostUnregister(mapped);
  hipFree(d_out);
  ::munmap(mapped, bytes);
  ::close(fd);
  ::unlink(path);
  return 0;
}

// -------------------------------------------------------------------- wmma
// Pure register-resident matrix throughput: no memory traffic at all, so this
// is the hard ceiling a GEMM kernel of any tiling can approach.
__global__ __launch_bounds__(256) void WmmaI8PeakKernel(int iters, int seed) {
  v16i8 a{}, b{};
  v8i c{};
  for (int i = 0; i < iters; ++i) {
    // The operands change every iteration, otherwise the loop is hoistable and
    // the measurement is of nothing.
    a[0] = (signed char)(i + seed);
    b[0] = (signed char)(i - seed);
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, b, true, a, c, true);
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, a, c, true);
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, b, true, b, c, true);
  }
  if (c[0] == 0x7f3f91ab) printf("%d", c[0]);
}

__global__ __launch_bounds__(256) void WmmaBf16PeakKernel(int iters, int seed) {
  v16h a{}, b{};
  v8f c{};
  for (int i = 0; i < iters; ++i) {
    a[0] = (_Float16)((float)i * 0.5f + (float)seed);
    b[0] = (_Float16)((float)i * -0.5f + (float)seed);
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b, a, c);
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, a, c);
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b, b, c);
  }
  if (c[0] == 12345.6789f) printf("%f", c[0]);
}

int BenchWmma() {
  std::printf("\n== WMMA register ceiling (no memory traffic) ==\n");
  hipDeviceProp_t prop{};
  (void)hipGetDeviceProperties(&prop, 0);
  const int block = 256;
  const int warps_per_block = block / 32;
  const int blocks = prop.multiProcessorCount * 8;
  const int iters = 4096;
  // One 16x16x16 matrix instruction is 16*16*16 multiply-accumulates.
  const double ops_per_instruction = 2.0 * 16 * 16 * 16;

  auto run = [&](auto kernel, const char* label, const char* unit) {
    kernel<<<blocks, block>>>(8, 1);  // warm up
    (void)hipDeviceSynchronize();
    const auto t0 = Clock::now();
    kernel<<<blocks, block>>>(iters, 1);
    (void)hipDeviceSynchronize();
    const auto t1 = Clock::now();
    const double secs = Seconds(t0, t1);
    // Four matrix instructions per loop iteration.
    const double ops =
        (double)blocks * warps_per_block * iters * 4.0 * ops_per_instruction;
    std::printf("%-28s %8.4f s  %8.1f %s\n", label, secs, ops / secs / 1e12,
                unit);
    fflush(stdout);
    return ops / secs / 1e12;
  };

  const double tops = run(WmmaI8PeakKernel, "int8 (iu8 x iu8 -> i32)", "TOPS");
  const double tflops =
      run(WmmaBf16PeakKernel, "bf16 (bf16 x bf16 -> f32)", "TFLOPS");
  std::printf("int8 speed-up over bf16: %.2fx\n", tops / tflops);
  return 0;
}

int BenchInfo() {
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("== device ==\n");
  std::printf("%-24s %s\n", "name", prop.name);
  std::printf("%-24s %s\n", "gcnArchName", prop.gcnArchName);
  std::printf("%-24s %d\n", "multiProcessorCount", prop.multiProcessorCount);
  std::printf("%-24s %d\n", "maxThreadsPerMultiProcessor",
              prop.maxThreadsPerMultiProcessor);
  std::printf("%-24s %d\n", "warpSize", prop.warpSize);
  std::printf("%-24s %d\n", "regsPerBlock", prop.regsPerBlock);
  std::printf("%-24s %zu KiB\n", "sharedMemPerBlock",
              prop.sharedMemPerBlock / 1024);
  std::printf("%-24s %zu KiB\n", "l2CacheSize", prop.l2CacheSize / 1024);
  std::printf("%-24s %d MHz\n", "clockRate", prop.clockRate / 1000);
  std::printf("%-24s %d\n", "memoryClockRate", prop.memoryClockRate);
  std::printf("%-24s %d\n", "memoryBusWidth", prop.memoryBusWidth);
  std::printf("%-24s %.2f GiB\n", "totalGlobalMem",
              prop.totalGlobalMem / 1073741824.0);
  std::printf("%-24s %d\n", "integrated", prop.integrated);
  std::printf("%-24s %d\n", "canMapHostMemory", prop.canMapHostMemory);
  std::printf("%-24s %d\n", "asyncEngineCount", prop.asyncEngineCount);
  std::printf("%-24s %d\n", "maxBlocksPerMultiProcessor", prop.maxBlocksPerMultiProcessor);
  std::printf("%-24s %zu KiB\n", "sharedMemPerBlockOptin", prop.sharedMemPerBlockOptin / 1024);
  std::printf("%-24s %zu KiB\n", "maxSharedMemoryPerMultiProcessor", prop.maxSharedMemoryPerMultiProcessor / 1024);
  std::printf("%-24s %d\n", "regsPerMultiprocessor", prop.regsPerMultiprocessor);
  std::printf("%-24s %d\n", "hostRegisterReadOnlySupported", prop.hostRegisterReadOnlySupported);
  std::printf("%-24s %d\n", "unifiedAddressing", prop.unifiedAddressing);
  std::printf("%-24s %d\n", "memoryBusWidth", prop.memoryBusWidth);
  std::printf("%-24s %d kHz\n", "memoryClockRate", prop.memoryClockRate);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string what = argc > 1 ? argv[1] : "all";
  if (what == "info") return BenchInfo();
  if (what == "mem") return BenchMem();
  if (what == "commit") return BenchCommit();
  if (what == "wmma") return BenchWmma();
  if (what == "all") {
    int rc = BenchInfo();
    if (rc) return rc;
    if ((rc = BenchMem())) return rc;
    if ((rc = BenchCommit())) return rc;
    return BenchWmma();
  }
  std::printf("usage: %s [info|mem|commit|wmma|all]\n", argv[0]);
  return 2;
}