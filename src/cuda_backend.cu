#include "engine.h"
#include "cuda_device.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace harvest_cuda {
namespace {
constexpr double pi = 3.14159265358979323846264338327950288;
void check(cudaError_t code) {
  check_cuda(code);
}
void check_fft(cufftResult code) {
  if (code == CUFFT_SUCCESS) return;
  const char* detail = "unknown error";
  switch (code) {
    case CUFFT_INVALID_PLAN: detail = "invalid FFT plan"; break;
    case CUFFT_ALLOC_FAILED: detail = "allocation failed; insufficient GPU memory"; break;
    case CUFFT_INVALID_TYPE: detail = "invalid transform type"; break;
    case CUFFT_INVALID_VALUE: detail = "invalid parameter"; break;
    case CUFFT_INTERNAL_ERROR: detail = "internal cuFFT error"; break;
    case CUFFT_EXEC_FAILED: detail = "FFT execution failed"; break;
    case CUFFT_SETUP_FAILED: detail = "cuFFT initialization failed"; break;
    case CUFFT_INVALID_SIZE: detail = "unsupported transform size"; break;
    case CUFFT_UNALIGNED_DATA: detail = "unaligned data"; break;
    case CUFFT_INVALID_DEVICE: detail = "FFT plan used on a different device"; break;
    case CUFFT_NO_WORKSPACE: detail = "missing FFT workspace"; break;
    case CUFFT_NOT_IMPLEMENTED: detail = "transform not implemented"; break;
    case CUFFT_NOT_SUPPORTED: detail = "operation not supported on this GPU"; break;
    case CUFFT_MISSING_DEPENDENCY: detail = "missing cuFFT dependency; check the installed CUDA runtime packages"; break;
    case CUFFT_NVRTC_FAILURE: detail = "runtime compilation failed; check CUDA runtime and driver compatibility"; break;
    case CUFFT_NVJITLINK_FAILURE: detail = "JIT linking failed; check nvJitLink and driver compatibility"; break;
    default: break;
  }
  throw std::runtime_error("cuFFT error " + std::to_string(int(code)) + ": " + detail);
}
template<class T> struct Buffer {
  T* p = nullptr;
  size_t capacity = 0;
  ~Buffer() { if (p) cudaFree(p); }
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void resize(size_t n) {
    if (n <= capacity) return;
    T* next = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&next), n * sizeof(T)));
    if (p) cudaFree(p);
    p = next; capacity = n;
  }
  void upload(const T* src, size_t n) {
    resize(n);
    check(cudaMemcpy(p, src, n * sizeof(T), cudaMemcpyHostToDevice));
  }
};
struct Plan {
  cufftHandle h = 0;
  size_t work_size = 0;
  size_t last_use = 0;
  Plan(int n, int batch, cufftType type) {
    check_fft(cufftCreate(&h));
    try {
      check_fft(cufftSetAutoAllocation(h, 0));
      check_fft(cufftMakePlan1d(h, n, type, batch, &work_size));
    } catch (...) {
      cufftDestroy(h);
      throw;
    }
  }
  ~Plan() { if (h) cufftDestroy(h); }
};
struct Job { double time; int half, base, row, col; double phase_s, phase_c; };
struct BinSelection { int n, count, bins[6]; };
struct AggregateState { double sum; int run, count; };
// Real/imaginary parts at N/2-1 and the real Nyquist bin.
struct alignas(16) FilterBins { double x, y, z; };
} // namespace

struct GpuWorkspace {
  // Declared first so it outlives every buffer and FFT plan below.
  CudaPrimaryContext context;
  explicit GpuWorkspace(int device) : context(device) {}
  Buffer<unsigned char> fft_work;
  Buffer<double> waveform, real, boundaries;
  Buffer<double> raw;
  Buffer<double> full_f0, full_scores, candidate_f0;
  Buffer<int> counts, edge_offsets, job_counts, owners, candidate_offsets;
  Buffer<AggregateState> aggregate_state;
  Buffer<cufftDoubleComplex> spectrum, signal_spectrum, filter_spectra;
  // Filter block spectra and refinement harmonics have disjoint lifetimes.
  Buffer<cufftDoubleComplex> twiddles, window_basis, aligned_windows, aux_spectrum;
  Buffer<cufftDoubleComplex> convolution_phases;
  Buffer<FilterBins> filter_ends, corrections;
  int filter_global_n = 0;
  int phase_length = 0;
  int filter_n = 0;
  double filter_fs = 0;
  std::vector<double> filter_boundaries;
  Buffer<Job> jobs;
  Buffer<BinSelection> selections;
  std::map<std::tuple<int, int, int>, std::unique_ptr<Plan>> plans;
  size_t plan_clock = 0;
  cufftHandle plan(int n, int batch, cufftType type) {
    auto key = std::make_tuple(n, batch, int(type));
    auto it = plans.find(key);
    if (it == plans.end()) {
      if (plans.size() >= 16) {
        auto oldest = std::min_element(plans.begin(), plans.end(),
            [](const auto& a, const auto& b) { return a.second->last_use < b.second->last_use; });
        plans.erase(oldest);
      }
      it = plans.emplace(key, std::make_unique<Plan>(n, batch, type)).first;
    }
    auto& p = *it->second;
    p.last_use = ++plan_clock;
    // FFTs execute sequentially on the default stream, so their scratch can
    // be shared. Rebind on every use because growing the buffer moves it.
    fft_work.resize(p.work_size);
    if (p.work_size) check_fft(cufftSetWorkArea(p.h, fft_work.p));
    return p.h;
  }
};
namespace {
// Coefficients from the pinned WORLD FilterForDecimate: a[0..2], b[0..1].
__constant__ double decimation_coefficients[13][5] = {
  {0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0},
  {0.041156734567757189, -0.42599112459189636, 0.041037215479961225, 0.16797464681802227, 0.50392394045406674},
  {0.95039378983237421, -0.67429146741526791, 0.15412211621346475, 0.071221945171178636, 0.21366583551353591},
  {1.4499664446880227, -0.98943497080950582, 0.24578252340690215, 0.036710750339322612, 0.11013225101796784},
  {1.7610939654280557, -1.2554914843859768, 0.3237186507788215, 0.021334858522387423, 0.06400457556716227},
  {1.9715352749512141, -1.4686795689225347, 0.3893908434965701, 0.013469181309343825, 0.040407543928031475},
  {2.1225239019534703, -1.6395144861046302, 0.44469707800587366, 0.0090366882681608418, 0.027110064804482525},
  {2.2357462340187593, -1.7780899984041358, 0.49152555365968692, 0.0063522763407111993, 0.019056829022133598},
  {2.3236003491759578, -1.8921545617463598, 0.53148928133729068, 0.0046331164041389372, 0.013899349212416812},
  {2.3936475118069387, -1.9873904075111861, 0.5658879979027055, 0.0034818622251927556, 0.010445586675578267},
  {2.450743295230728, -2.06794904601978, 0.59574774438332101, 0.0026822508007163792, 0.0080467524021491377},
  {2.4981398605924205, -2.1368928194784025, 0.62187513816221485, 0.0021097275904709001, 0.0063291827714127002}
};
// Overlap each independent IIR segment far beyond its settling time.
// 64*r samples attenuate the slowest pole below 1e-23 for all supported ratios.
template<bool reverse>
__global__ void decimate_pass(const double* input, int length, int total, int lag,
                              int ratio, int begin, int out_length, double* output) {
  size_t offset = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 32;
  if (offset >= size_t(total)) return;
  int start = int(offset), end = start + min(32, total - start);
  const double* c = decimation_coefficients[ratio];
  double w0 = 0, w1 = 0, w2 = 0;
  for (int i = max(0, start - 64 * ratio); i < end; ++i) {
    double x = reverse ? input[total - 1 - i] : input[max(0, min(length - 1, i - lag - 9))];
    double wt = x + c[0] * w0 + c[1] * w1 + c[2] * w2;
    if (i >= start) {
      int index = i;
      if constexpr (reverse) index = total - 1 - begin - i;
      if (!reverse || (index >= 0 && index % ratio == 0 && index / ratio < out_length))
        output[reverse ? index / ratio : index] = c[3] * wt + c[4] * w0 + c[4] * w1 + c[3] * w2;
    }
    w2 = w1; w1 = w0; w0 = wt;
  }
}
__global__ void make_filters(double* dst, int n, double fs, const double* boundaries) {
  int band = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double f = boundaries[band];
  int half = int(fs / f * 2.0 + 0.5);
  double v = 0;
  if (i <= 2 * half) {
    double t = i / (2.0 * half);
    v = (0.355768 - 0.487396 * cos(2.0 * pi * t) +
         0.144232 * cos(4.0 * pi * t) - 0.012604 * cos(6.0 * pi * t)) *
         cos(2.0 * pi * f * (i - half) / fs);
  }
  dst[size_t(band) * n + i] = v;
}
__device__ cufftDoubleComplex multiply(cufftDoubleComplex a, cufftDoubleComplex b) {
  return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
__global__ void convolve(const cufftDoubleComplex* filters, const cufftDoubleComplex* signal,
                         cufftDoubleComplex* dst, int n) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int band = blockIdx.y;
  int bins = n / 2 + 1;
  if (k >= bins) return;
  // Reproduce WORLD's assignments at the final two bins (including its
  // overwrite of the Nyquist bin) rather than silently changing the filter.
  int source = k >= n / 2 - 1 ? n / 2 - 1 : k;
  auto value = multiply(filters[size_t(band) * bins + source], signal[source]);
  if (k >= n / 2 - 1) value = multiply(value, signal[n / 2]);
  dst[size_t(band) * bins + k] = value;
}

// Overlap-save computes linear convolution with short transforms. WORLD's
// unusual final two global FFT bins are corrected separately below.
__global__ void gather_blocks(const double* y, int length, int n, int step,
                              int blocks_count, double* blocks) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int block = int(index / n), i = int(index % n);
  if (block >= blocks_count) return;
  int sample = block * step - (n - step) + i;
  blocks[index] = sample >= 0 && sample < length ? y[sample] : 0;
}
__global__ void filter_ends(const double* filters, int n, int full_n, double fs,
                            const double* boundaries, FilterBins* ends) {
  int band = blockIdx.x;
  double re = 0, im = 0, nyquist = 0;
  int size = min(n, 2 * int(fs / boundaries[band] * 2.0 + 0.5) + 1);
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    double v = filters[size_t(band) * n + i] * (i % 2 ? -1.0 : 1.0);
    double c, s; sincos(2.0 * pi * i / full_n, &s, &c);
    re += v * c; im += v * s; nyquist += v;
  }
  __shared__ double values[3][128];
  values[0][threadIdx.x] = re; values[1][threadIdx.x] = im; values[2][threadIdx.x] = nyquist;
  __syncthreads();
  for (int offset = 64; offset; offset /= 2) {
    if (threadIdx.x < offset) for (int j = 0; j < 3; ++j)
      values[j][threadIdx.x] += values[j][threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) ends[band] = {values[0][0], values[1][0], values[2][0]};
}
__global__ void convolution_phases(cufftDoubleComplex* phases, int n, int length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= length) return;
  double c, s; sincos(2.0 * pi * i / n, &s, &c);
  phases[i] = {c, s};
}
__global__ void correct_filter_ends(const FilterBins* ends, const cufftDoubleComplex* signal,
                                   int n, int bands, FilterBins* corrections) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= bands) return;
  auto h = ends[b];
  auto p = multiply({h.x, h.y}, signal[n / 2 - 1]);
  auto changed = multiply(p, signal[n / 2]);
  // Difference from ordinary convolution at N/2-1 (complex) and N/2 (real).
  // Fold the conjugate bin's factor of two into the complex correction.
  corrections[b] = {2.0 * (changed.x - p.x), 2.0 * (changed.y - p.y),
                    changed.x - signal[n / 2].x * h.z};
}
__global__ void convolve_blocks(const cufftDoubleComplex* filters, const cufftDoubleComplex* signal,
                                int n, int blocks, int bands, cufftDoubleComplex* output) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int bins = n / 2 + 1;
  int transform = int(index / bins), k = int(index % bins);
  int band = transform / blocks, block = transform % blocks;
  if (band >= bands) return;
  output[index] = multiply(filters[size_t(band) * bins + k],
                           signal[size_t(block) * bins + k]);
}
// Each crossing type occurs at most 128 times in a 256-sample tile.
// Pack the four counts into bytes so one prefix scan handles all four types.
__global__ void extract_edges(const double* input, int length, int tiles,
                              int n, int blocks, int step, double scale, double fs,
                              const double* boundaries, bool segmented,
                              const FilterBins* corrections, const cufftDoubleComplex* phases,
                              double* edges, int* offsets) {
  int band = blockIdx.y, tile = blockIdx.x;
  __shared__ double samples[258];
  __shared__ int lag;
  if (threadIdx.x == 0) lag = int(fs / boundaries[band] * 2.0 + 0.5) + 1;
  __syncthreads();
  // Recover only this tile and its two-sample derivative halo from the FFT.
  // No full filtered-waveform buffer is needed between convolution and edges.
  for (int j = threadIdx.x; j < 258; j += 256) {
    int i = tile * 256 + j;
    double value = 0;
    if (i < length) {
      int q = i + lag;
      if (segmented) {
        int block = q / step, sample = q % step + n - step;
        auto d = corrections[band]; auto p = phases[q];
        double correction = fma(d.x, p.x, fma(d.y, p.y, d.z));
        value = input[(size_t(band) * blocks + block) * n + sample] * scale;
        value += q % 2 ? -correction : correction;
      } else value = input[size_t(band) * n + q];
    }
    samples[j] = value;
  }
  __syncthreads();
  int i = tile * 256 + threadIdx.x;
  double a = samples[threadIdx.x], b = samples[threadIdx.x + 1], c = samples[threadIdx.x + 2];
  double da = b - a, db = c - b;
  unsigned flags = 0;
  if (i < length - 1) {
    if (a > 0 && b <= 0) flags |= 1;
    if (a < 0 && b >= 0) flags |= 1u << 8;
  }
  if (i < length - 2) {
    if (da > 0 && db <= 0) flags |= 1u << 16;
    if (da < 0 && db >= 0) flags |= 1u << 24;
  }
  using Scan = cub::BlockScan<unsigned, 256>;
  __shared__ Scan::TempStorage scratch;
  unsigned prefix, total;
  Scan(scratch).ExclusiveSum(flags, prefix, total);
  #pragma unroll
  for (int pair = 0; pair < 2; ++pair) {
    unsigned mask = (flags >> (pair * 16)) & 0xffff;
    if (mask) {
      int kind = pair * 2 + (mask & 1 ? 0 : 1);
      int event = band * 4 + kind;
      double left = pair ? da : a, right = pair ? db : b;
      edges[(size_t(event) * tiles + tile) * 128 + ((prefix >> (8 * kind)) & 255)] =
          i + 1 - left / (right - left);
    }
  }
  if (threadIdx.x == 0) for (int kind = 0; kind < 4; ++kind)
    offsets[size_t(band * 4 + kind) * (tiles + 1) + tile] = (total >> (8 * kind)) & 255;
}
__global__ void scan_edge_counts(int* offsets, int tiles, int* counts) {
  int event = blockIdx.x;
  int* row = offsets + size_t(event) * (tiles + 1);
  using Scan = cub::BlockScan<int, 256>;
  __shared__ Scan::TempStorage scratch;
  __shared__ int carry;
  if (threadIdx.x == 0) carry = 0;
  __syncthreads();
  for (int start = 0; start < tiles; start += 256) {
    int i = start + threadIdx.x;
    int prefix, total;
    Scan(scratch).ExclusiveSum(i < tiles ? row[i] : 0, prefix, total);
    if (i < tiles) row[i] = carry + prefix;
    __syncthreads();
    if (threadIdx.x == 0) carry += total;
    __syncthreads();
  }
  if (threadIdx.x == 0) counts[event] = row[tiles] = carry;
}
__global__ void pack_edges(const double* tiles_data, const int* offsets,
                           int tiles, int stride, double* edges) {
  int event = blockIdx.y;
  int tile = blockIdx.x;
  const int* row = offsets + size_t(event) * (tiles + 1);
  int start = row[tile];
  if (threadIdx.x < row[tile + 1] - start)
    edges[size_t(event) * stride + start + threadIdx.x] =
      tiles_data[(size_t(event) * tiles + tile) * 128 + threadIdx.x];
}
__global__ void interpolate_events(const double* edges, const int* counts,
                                   const int* offsets, int tiles,
                                   int stride, int frames, double fs,
                                   const double* boundaries, double floor,
                                   double ceil, double* raw) {
  int frame = blockIdx.x * blockDim.x + threadIdx.x;
  int band = blockIdx.y;
  if (frame >= frames) return;
  double sample = (frame / 1000.0) * fs;
  double sum = 0;
  for (int kind = 0; kind < 4; ++kind) {
    int event = band * 4 + kind;
    int count = counts[event];
    if (count < 4) { raw[size_t(band) * frames + frame] = 0; return; }
    const double* e = edges + size_t(event) * stride;
    // Tile prefix counts bracket nearby crossings. Include neighbors because
    // interpolation uses midpoints of two crossings, including across gaps.
    int tile = max(0, min(tiles - 1, int(sample) / 256));
    const int* prefix = offsets + size_t(event) * (tiles + 1);
    int lo = max(0, prefix[tile] - 2);
    int hi = min(count - 1, prefix[tile + 1] + 1);
    // Search in sample coordinates to avoid division at every bisection.
    // Preserve WORLD's endpoint extrapolation after locating the interval.
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (e[mid] + e[mid + 1] <= sample * 2.0) lo = mid + 1;
      else hi = mid;
    }
    int k = max(0, min(count - 3, lo - 1));
    // Linear interpolation is invariant under the conversion to seconds.
    double left = (e[k] + e[k + 1]) / 2.0;
    double right = (e[k + 1] + e[k + 2]) / 2.0;
    // Combine the two reciprocal periods and interpolation into one quotient.
    double d0 = e[k + 1] - e[k], d1 = e[k + 2] - e[k + 1];
    double span = right - left;
    sum += fs * (span * d1 + (sample - left) * (d0 - d1)) / (span * d0 * d1);
  }
  double value = sum / 4.0;
  double boundary = boundaries[band];
  if (value > boundary * 1.1 || value < boundary * 0.9 || value > ceil || value < floor) value = 0;
  raw[size_t(band) * frames + frame] = value;
}

__device__ double window(int i, const Job& job, double fs) {
  double tmp = (job.base + i - 1.0) / fs - job.time;
  double duration = (2.0 * job.half + 1.0) / fs;
  return 0.42 + 0.5 * cos(2.0 * pi * tmp / duration) +
         0.08 * cos(4.0 * pi * tmp / duration);
}

__global__ void aggregate_candidates(const double* raw, int first_band, int batch,
                                     int bands, int frames, int capacity,
                                     AggregateState* states, double* candidates, int* maximum) {
  int frame = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame >= frames) return;
  AggregateState state = first_band ? states[frame] : AggregateState{};
  // WORLD forces the first and last channels unvoiced and averages runs of
  // at least ten voiced channels, in their original summation order.
  for (int band = 0; band < batch; ++band) {
    int index = first_band + band;
    double value = index == 0 || index == bands - 1 ? 0 : raw[size_t(band) * frames + frame];
    if (value > 0) { ++state.run; state.sum += value; }
    else {
      if (state.run >= 10)
        candidates[size_t(frame) * capacity + state.count++] = state.sum / state.run;
      state.run = 0;
      state.sum = 0;
    }
  }
  if (first_band + batch == bands) atomicMax(maximum, state.count);
  else states[frame] = state;
}

// Expand the original candidates only for the refinement chunk being processed.
__global__ void overlap_candidates(const double* original, int total_frames, int stride,
                                   int first_frame, int frames, int candidates, double* output) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(frames) * candidates) return;
  int count = candidates / 7;
  int frame = first_frame + int(index / candidates), col = int(index % candidates);
  int group = col / count;
  int source = frame + (group <= 3 ? -group : group - 3);
  output[index] = source >= 0 && source < total_frames ?
      original[size_t(source) * stride + col % count] : 0;
}
__global__ void make_windows(const double* waveform, int length, const Job* jobs,
                             int count, double fs, int n, double* dst) {
  int j = blockIdx.x;
  __shared__ double sin_step, sin_two_steps, phase, step;
  if (threadIdx.x == 0 && j < count) {
    Job job = jobs[j];
    step = 2.0 * pi / (2.0 * job.half + 1.0);
    phase = 2.0 * pi * ((job.base - 1.0) / fs - job.time) /
            ((2.0 * job.half + 1.0) / fs);
    sin_step = sin(step);
    sin_two_steps = sin(2.0 * step);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    double main = 0, diff = 0;
    if (j < count) {
      Job job = jobs[j];
      int size = 2 * job.half + 1;
      if (i < size) {
        int index = max(0, min(length - 1, job.base + i - 1));
        double x = waveform[index];
        double s, c;
        sincos(phase + i * step, &s, &c);
        // Trigonometric identities for the same Blackman window and its
        // centered finite difference; one sincos instead of six cos calls.
        main = x * (0.42 + 0.5 * c + 0.08 * (2.0 * c * c - 1.0));
        double d = i == 0 ? -window(1, job, fs) / 2.0 :
                   i == size - 1 ? window(i - 1, job, fs) / 2.0 :
                   0.5 * s * sin_step + 0.08 * (2.0 * s * c) * sin_two_steps;
        diff = x * d;
      }
    }
    dst[size_t(2 * j) * n + i] = main;
    dst[size_t(2 * j + 1) * n + i] = diff;
  }
}
// Equal integer window lengths at the same frame produce identical spectra.
// Classify once per frame, comparing integer lengths rather than recomputing
// divisions for every pair of candidates.
__global__ void count_jobs(const double* f0, int candidates,
                           double fs, int* counts, int* owners, BinSelection* selections) {
  int row = blockIdx.x;
  extern __shared__ int halves[];
  for (int col = threadIdx.x; col < candidates; col += blockDim.x) {
    double value = f0[size_t(row) * candidates + col];
    int half = value > 0 ? int(1.5 * fs / value + 1.0) : 0;
    halves[col] = half;
    if (half) {
      BinSelection selection{};
      selection.n = 1 << (33 - __clz(unsigned(2 * half + 1)));
      selection.count = min(int(fs / 2.0 / value), 6);
      for (int h = 1; h <= selection.count; ++h)
        selection.bins[h - 1] = int(value * selection.n / fs * h + 0.5);
      selections[size_t(row) * candidates + col] = selection;
    }
  }
  __syncthreads();
  for (int col = threadIdx.x; col < candidates; col += blockDim.x) {
    int half = halves[col], owner = col;
    if (half) {
      for (int j = 0; j < col; ++j) if (halves[j] == half) { owner = j; break; }
      if (owner == col) atomicAdd(counts + 33 - __clz(unsigned(2 * half + 1)), 1);
    } else owner = -1;
    owners[size_t(row) * candidates + col] = owner;
  }
}
__global__ void make_jobs(const double* f0, int frames, int first_frame, int candidates,
                          double fs, const int* owners, int* offsets, Job* jobs) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(frames) * candidates) return;
  int row = int(index / candidates), col = int(index % candidates);
  if (owners[index] != col) return;
  double value = f0[index];
  int half = int(1.5 * fs / value + 1.0);
  int exponent = 33 - __clz(unsigned(2 * half + 1));
  double time = (row + first_frame) / 1000.0;
  double base = (time + (-half) / fs) * fs + 0.001;
  int rounded = base > 0 ? int(base + 0.5) : int(base - 0.5);
  double phase_s, phase_c;
  double delta = (rounded + half) / fs - time;
  sincos(2.0 * pi * delta / ((2.0 * half + 1.0) / fs), &phase_s, &phase_c);
  jobs[atomicAdd(offsets + exponent, 1)] = {time, half, rounded, row, col, phase_s, phase_c};
}
// Large windows use cuFFT, then retain only the bins used by the scorer.
__global__ void collect_bins(const cufftDoubleComplex* spectra, const Job* jobs,
                             int count, int n, int candidates, const int* owners,
                             const BinSelection* selections, cufftDoubleComplex* output) {
  int j = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (j >= count) return;
  Job job = jobs[j];
  int bins = n / 2 + 1;
  const auto* main = spectra + size_t(2 * j) * bins;
  const auto* diff = main + bins;
  for (int col = threadIdx.x % 32; col < candidates; col += 32) {
    size_t index = size_t(job.row) * candidates + col;
    if (owners[index] != job.col) continue;
    auto selection = selections[index];
    for (int h = 0; h < selection.count; ++h) {
      int k = selection.bins[h];
      output[index * 6 + h] = {
        main[k].x * main[k].x + main[k].y * main[k].y,
        main[k].x * diff[k].y - main[k].y * diff[k].x};
    }
  }
}

// Small refinements fit in shared memory. Packing the main and derivative
// windows into one complex FFT avoids materializing two global FFT inputs.
constexpr int local_fft_limit = 2048;
__global__ void make_twiddles(cufftDoubleComplex* values) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= local_fft_limit / 2) return;
  double s, c;
  sincos(-2.0 * pi * i / local_fft_limit, &s, &c);
  values[i] = {c, s};
}
__global__ void make_local_windows(cufftDoubleComplex* basis) {
  int half = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > 2 * half) return;
  double s, c;
  sincos(2.0 * pi * (i - half - 1.0) / (2.0 * half + 1.0), &s, &c);
  // Sum of the preceding odd window lengths is half squared.
  basis[size_t(half) * half + i] = {c, s};
}
// Sample-aligned frames have exactly zero phase offset. Cache their complete
// window and derivative without changing the arithmetic of the general path.
__global__ void make_aligned_windows(const cufftDoubleComplex* basis, cufftDoubleComplex* weights) {
  int half = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (half < 3 || i > 2 * half) return;
  size_t offset = size_t(half) * half;
  auto v = basis[offset + i];
  double c = v.x, s = v.y;
  double main = 0.42 + 0.5 * c + 0.08 * (2.0 * c * c - 1.0);
  double d = 0.5 * s * basis[offset + half + 2].y +
      0.08 * (2.0 * s * c) * basis[offset + half + 3].y;
  if (i == 0 || i == 2 * half) {
    double c = basis[offset + (i == 0 ? 1 : i - 1)].x;
    d = (i == 0 ? -0.5 : 0.5) * (0.42 + 0.5 * c + 0.08 * (2.0 * c * c - 1.0));
  }
  weights[offset + i] = {main, d};
}
__device__ cufftDoubleComplex refinement_sample(const double* waveform, int length,
    const Job& job, int i, const cufftDoubleComplex* basis, const cufftDoubleComplex* aligned) {
  double sin_step = basis[size_t(job.half) * job.half + job.half + 2].y;
  double sin_two_steps = basis[size_t(job.half) * job.half + job.half + 3].y;
  double phase_s = job.phase_s, phase_c = job.phase_c;
  int size = 2 * job.half + 1;
  double main = 0, diff = 0;
  if (i < size) {
    double x = waveform[max(0, min(length - 1, job.base + i - 1))];
    if (phase_s == 0.0 && phase_c == 1.0) {
      auto weights = aligned[size_t(job.half) * job.half + i];
      main = x * weights.x;
      diff = x * weights.y;
    } else {
      auto v = basis[size_t(job.half) * job.half + i];
      double s = v.y * phase_c + v.x * phase_s;
      double c = v.x * phase_c - v.y * phase_s;
      main = x * (0.42 + 0.5 * c + 0.08 * (2.0 * c * c - 1.0));
      double d = 0.5 * s * sin_step + 0.08 * (2.0 * s * c) * sin_two_steps;
      if (i == 0 || i == size - 1) {
        auto neighbor = basis[size_t(job.half) * job.half + (i == 0 ? 1 : i - 1)];
        double c = neighbor.x * phase_c - neighbor.y * phase_s;
        d = (i == 0 ? -0.5 : 0.5) * (0.42 + 0.5 * c + 0.08 * (2.0 * c * c - 1.0));
      }
      diff = x * d;
    }
  }
  return {main, diff};
}
__device__ cufftDoubleComplex multiply_fft(cufftDoubleComplex a, cufftDoubleComplex b) {
  // Explicit IEEE double FMA; global contraction and fast-math remain disabled.
  return {fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x)};
}
template<int n>
__global__ __launch_bounds__(n <= 256 ? 32 : n == 512 ? 64 : 128)
void refine_local(const double* waveform, int length, const Job* jobs, int candidates,
                             const int* owner, const BinSelection* selections,
                             const cufftDoubleComplex* twiddles, const cufftDoubleComplex* basis,
                             const cufftDoubleComplex* aligned,
                             cufftDoubleComplex* refined_bins) {
  constexpr int threads = n <= 256 ? 32 : n == 512 ? 64 : 128;
  extern __shared__ cufftDoubleComplex data[];
  Job job = jobs[blockIdx.x];
  int bits = 31 - __clz(unsigned(n));
  // The input window occupies less than half the FFT. Fuse window generation
  // with the first radix-4 stage, omitting its two identically zero inputs.
  for (int i = threadIdx.x; i < n / 4; i += threads) {
    auto a = refinement_sample(waveform, length, job, i, basis, aligned);
    auto c = refinement_sample(waveform, length, job, i + n / 4, basis, aligned);
    int start = __brev(unsigned(i)) >> (32 - bits);
    data[start] = {a.x + c.x, a.y + c.y};
    data[start + 1] = {a.x + c.y, a.y - c.x};
    data[start + 2] = {a.x - c.x, a.y - c.y};
    data[start + 3] = {a.x - c.y, a.y + c.x};
  }
  __syncthreads();
  // Combine two radix-2 stages: three complex products instead of four,
  // with one shared-memory round trip and barrier per pair of stages.
  #pragma unroll
  for (int half = 4; half < n / 2; half *= 4) {
    constexpr int period = local_fft_limit;
    #pragma unroll
    for (int i = threadIdx.x; i < n / 4; i += threads) {
      int k = i & (half - 1);
      int start = ((i & ~(half - 1)) << 2) + k;
      auto a = data[start], b = data[start + half];
      auto c = data[start + 2 * half], d = data[start + 3 * half];
      if (k) {
        int step = period / (4 * half);
        b = multiply_fft(b, twiddles[2 * k * step]);
        c = multiply_fft(c, twiddles[k * step]);
        int q = 3 * k * step;
        auto w = twiddles[q % (period / 2)];
        if (q >= period / 2) { w.x = -w.x; w.y = -w.y; }
        d = multiply_fft(d, w);
      }
      cufftDoubleComplex ab = {a.x + b.x, a.y + b.y};
      cufftDoubleComplex cd = {c.x + d.x, c.y + d.y};
      cufftDoubleComplex ba = {a.x - b.x, a.y - b.y};
      cufftDoubleComplex dc = {c.y - d.y, d.x - c.x};
      data[start] = {ab.x + cd.x, ab.y + cd.y};
      data[start + half] = {ba.x + dc.x, ba.y + dc.y};
      data[start + 2 * half] = {ab.x - cd.x, ab.y - cd.y};
      data[start + 3 * half] = {ba.x - dc.x, ba.y - dc.y};
    }
    __syncthreads();
  }
  if constexpr (n == 32 || n == 128 || n == 512 || n == 2048) {
    for (int i = threadIdx.x; i < n / 2; i += threads) {
      auto a = data[i];
      auto b = multiply_fft(data[i + n / 2], twiddles[i * (local_fft_limit / n)]);
      data[i] = {a.x + b.x, a.y + b.y};
      data[i + n / 2] = {a.x - b.x, a.y - b.y};
    }
    __syncthreads();
  }
  for (int col = threadIdx.x; col < candidates; col += threads) {
    size_t index = size_t(job.row) * candidates + col;
    if (owner[index] != job.col) continue;
    auto selection = selections[index];
    int harmonics = selection.count;
    for (int h = 1; h <= harmonics; ++h) {
      int k = selection.bins[h - 1];
      auto a = data[k], b = data[n - k];
      double real = (a.x + b.x) / 2.0, imag = (a.y - b.y) / 2.0;
      double diff_real = (a.y + b.y) / 2.0, diff_imag = (b.x - a.x) / 2.0;
      refined_bins[index * 6 + h - 1] = {
        real * real + imag * imag, real * diff_imag - imag * diff_real};
    }
  }
}

__global__ void score_bins(const cufftDoubleComplex* bins, const double* original,
                           const BinSelection* selections, int frames, int candidates,
                           double fs, double floor, double ceil, double* f0, double* scores) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(frames) * candidates) return;
  double value = original[index];
  if (value <= 0) { f0[index] = scores[index] = 0; return; }
  auto selection = selections[index];
  int n = selection.n;
  int harmonics = selection.count;
  double num = 0, den = 0, score = 0;
  double inverse_value = 1.0 / value;
  #pragma unroll
  for (int h = 1; h <= 6; ++h) {
    if (h > harmonics) break;
    int k = selection.bins[h - 1];
    auto b = bins[index * 6 + h - 1];
    double frequency = b.x == 0 ? 0 : double(k) * fs / n + b.y / b.x * fs / 2.0 / pi;
    double amplitude = sqrt(b.x);
    num += amplitude * frequency;
    den += amplitude * h;
    score += fabs((frequency * (1.0 / h) - value) * inverse_value);
  }
  double refined = num / (den + 1e-12);
  score = 1.0 / (score / harmonics + 1e-12);
  if (refined < floor || refined > ceil || score < 2.5) refined = score = 0;
  f0[index] = refined;
  scores[index] = score;
}

// Multiplication handles comparisons far from the threshold. Retain the
// original division near it so rounding cannot change a keep/drop decision.
__device__ bool nearby_candidate(double value, double current, double lower, double upper) {
  double difference = fabs(current - value);
  if (difference < lower) return true;
  if (difference > upper) return false;
  return difference / current <= 0.05;
}
__global__ void prune_candidates(const double* input, double* output, double* scores,
                                 int frames, int candidates, double* best, int* counts) {
  int frame = blockIdx.x;
  double best_score = 0, best_value = 0;
  int best_index = INT_MAX;
  int nonzero = 0;
  for (int col = threadIdx.x; col < candidates; col += 32) {
    size_t index = size_t(frame) * candidates + col;
    double f0 = input[index];
    double limit = f0 * 0.05;
    double margin = limit * 0x1p-49;
    double lower = limit - margin, upper = limit + margin;
    bool keep = frame == 0 || frame == frames - 1 || f0 == 0;
    if (!keep) for (int j = 0; j < candidates; ++j) {
      double prev = input[size_t(frame - 1) * candidates + j];
      double next = input[size_t(frame + 1) * candidates + j];
      if (nearby_candidate(prev, f0, lower, upper) || nearby_candidate(next, f0, lower, upper)) { keep = true; break; }
    }
    output[index] = keep ? f0 : 0;
    nonzero += output[index] != 0;
    double score = keep ? scores[index] : 0;
    if (!keep) scores[index] = 0;
    if (score > best_score) { best_score = score; best_value = f0; best_index = col; }
  }
  // WORLD selects the first candidate on a score tie.
  for (int offset = 16; offset; offset /= 2) {
    nonzero += __shfl_down_sync(0xffffffff, nonzero, offset);
    double score = __shfl_down_sync(0xffffffff, best_score, offset);
    double value = __shfl_down_sync(0xffffffff, best_value, offset);
    int index = __shfl_down_sync(0xffffffff, best_index, offset);
    if (score > best_score || (score == best_score && index < best_index)) {
      best_score = score; best_value = value; best_index = index;
    }
  }
  if (threadIdx.x == 0) {
    best[frame] = best_score > 0 ? best_value : 0;
    counts[frame] = nonzero;
    if (frame == 0) counts[frames] = 0;
  }
}

// Preserve candidate order, including duplicate pitches and score ties.
__global__ void pack_candidates(const double* f0, const double* scores,
                                int candidates, const int* offsets,
                                double* packed_f0, double* packed_scores) {
  int frame = blockIdx.x, lane = threadIdx.x;
  int out = offsets[frame];
  for (int first = 0; first < candidates; first += 32) {
    int col = first + lane;
    size_t index = size_t(frame) * candidates + col;
    double value = col < candidates ? f0[index] : 0;
    unsigned mask = __ballot_sync(0xffffffff, value != 0);
    if (value != 0) {
      int position = out + __popc(mask & ((1u << lane) - 1));
      packed_f0[position] = value;
      packed_scores[position] = scores[index];
    }
    out += __popc(mask);
  }
}
}

bool cuda_available(int device) noexcept {
  if (device < 0) return false;
  try {
    check_harvest_device(device);
    return true;
  } catch (...) {
    cudaGetLastError();
    return false;
  }
}
GpuWorkspace* create_gpu_workspace(int device) {
  check_harvest_device(device);
  return new GpuWorkspace(device);
}
void delete_gpu_workspace(GpuWorkspace* w) noexcept {
  if (!w) return;
  try {
    CudaContextGuard context(w->context.get());
    delete w;
  } catch (...) {
    // Best-effort cleanup if the driver has already shut down.
    delete w;
  }
}
void gpu_decimate(GpuWorkspace& w, const double* x, int length, int ratio, double* y, int y_length) {
  CudaContextGuard context(w.context.get());
  int lag = ((140 + ratio - 1) / ratio) * ratio;
  if (length > INT_MAX - 2 * lag - 18)
    throw std::invalid_argument("input is too long for decimation padding");
  int padded = length + 2 * lag, total = padded + 18;
  // Candidate storage is unused during decimation. The reverse pass writes
  // only retained samples, directly into the downsampled waveform buffer.
  w.candidate_f0.upload(x, length);
  w.real.resize(total); w.waveform.resize(y_length);
  unsigned blocks = unsigned((size_t(total) + 4095) / 4096);
  decimate_pass<false><<<blocks, 128>>>(w.candidate_f0.p, length, total, lag,
      ratio, 0, 0, w.real.p);
  check(cudaGetLastError());
  int nout = (padded - 1) / ratio + 1;
  int begin = ratio - ratio * nout + padded + lag + 8;
  decimate_pass<true><<<blocks, 128>>>(w.real.p, total, total, lag,
      ratio, begin, y_length, w.waveform.p);
  check(cudaGetLastError());
  check(cudaMemcpy(y, w.waveform.p, size_t(y_length) * sizeof(double), cudaMemcpyDeviceToHost));
}

static int gpu_candidates(GpuWorkspace& w, const double* y, int length, int n, double fs,
                   const double* boundaries, int bands, int frames,
                   double floor, double ceil, int capacity) {
  int half = int(fs / boundaries[0] * 2.0 + 0.5);
  int block_n = 1;
  // Longer blocks reduce overlap work; cap them without truncating a filter.
  while (block_n < std::min(8192, (n >= 32768 ? 32 : 8) * half)) block_n *= 2;
  bool segmented = block_n > 2 * half && n >= 4 * block_n;
  int transform_n = segmented ? block_n : n;
  int step = transform_n - 2 * half;
  int blocks = segmented ? (length + half + 1 + step - 1) / step : 1;
  int bins = transform_n / 2 + 1;
  // Cache as many complete band batches as fit, even when the whole filter
  // bank exceeds the limit. Remaining bands use scratch storage.
  // Bound filter batches; reuse FFT storage for crossings below.
  int batch = std::max(1, std::min(128, int((size_t(80) << 20) / (size_t(transform_n) * blocks * 40))));
  // Distribute bands without increasing the number of batches. This avoids
  // oversized full batches followed by a small tail and improves plan reuse.
  int batches = (bands + batch - 1) / batch;
  batch = (bands + batches - 1) / batches;
  // Nearby convolution batch sizes can share cuFFT plans. Keep extra work
  // small and leave recordings exceeding the normal batch budget unpadded.
  auto padded_batch = [](int count) { return count < 256 ? (count + 31) / 32 * 32 : count; };
  int fft_transforms = segmented ? padded_batch(batch * blocks) : batch * blocks;
  int cache_bands = int(std::min(size_t(bands), (size_t(256) << 20) /
      (size_t(bins) * sizeof(cufftDoubleComplex))));
  if (cache_bands < bands) cache_bands = cache_bands / batch * batch;
  bool cached = w.filter_n == transform_n && w.filter_fs == fs &&
    w.filter_boundaries.size() == size_t(bands) &&
    std::equal(w.filter_boundaries.begin(), w.filter_boundaries.end(), boundaries);
  // Block filter spectra depend on the block length, while WORLD's special
  // frequency-bin corrections also depend on the whole-recording FFT length.
  bool ends_cached = cached && w.filter_global_n == n;
  if (cache_bands && !cached) {
    w.filter_n = 0;
    w.filter_global_n = 0;
    w.filter_spectra.resize(size_t(bins) * cache_bands);
  }
  int stride = (length + 1) / 2 + 1;
  int tiles = (length + 255) / 256;
  w.waveform.resize(n);
  check(cudaMemcpy(w.waveform.p, y, size_t(length) * sizeof(double), cudaMemcpyHostToDevice));
  check(cudaMemset(w.waveform.p + length, 0, size_t(n - length) * sizeof(double)));
  w.boundaries.upload(boundaries, bands);
  w.signal_spectrum.resize(n / 2 + 1);
  check_fft(cufftExecD2Z(w.plan(n, 1, CUFFT_D2Z), w.waveform.p, w.signal_spectrum.p));
  w.real.resize(size_t(transform_n) * std::max(fft_transforms, blocks));
  w.spectrum.resize(size_t(bins) * (segmented ? fft_transforms : batch * std::max(2, blocks)));
  if (segmented) {
    w.aux_spectrum.resize(size_t(bins) * blocks);
    gather_blocks<<<unsigned((size_t(transform_n)*blocks+255)/256),256>>>(w.waveform.p, length, transform_n, step, blocks, w.real.p);
    check(cudaGetLastError());
    check_fft(cufftExecD2Z(w.plan(transform_n, blocks, CUFFT_D2Z), w.real.p, w.aux_spectrum.p));
    w.filter_ends.resize(bands); w.corrections.resize(bands);
    // Crossing extraction reads through (length - 1) + the largest filter lag,
    // not through the whole padded FFT. Track initialized length separately
    // from capacity: changing n can leave a stale tail in a larger allocation.
    int phase_length = length + half + 1;
    if (!ends_cached || phase_length > w.phase_length) {
      // Validate only after the synchronous copy at the end of this function.
      w.filter_global_n = 0;
      w.phase_length = 0;
      // Coarse allocation steps avoid reallocating for each slightly longer
      // recording; extra capacity is less than 2 MiB and never exceeds n.
      constexpr int phase_chunk = (2 << 20) / sizeof(cufftDoubleComplex);
      w.convolution_phases.resize(std::min(n,
          (phase_length + phase_chunk - 1) / phase_chunk * phase_chunk));
      convolution_phases<<<(phase_length+255)/256,256>>>(w.convolution_phases.p, n, phase_length);
      check(cudaGetLastError());
      w.phase_length = phase_length;
    }
  }
  // The inverse FFT is finished before compact crossings use this storage.
  w.real.resize(size_t(stride) * batch * 4);
  w.counts.resize(batch * 4);
  // Complex FFT products and tiled crossings have disjoint lifetimes.
  w.spectrum.resize(size_t(tiles) * 64 * batch * 4);
  double* tiled_edges = reinterpret_cast<double*>(w.spectrum.p);
  w.edge_offsets.resize(size_t(tiles + 1) * batch * 4);
  // Keep only one band batch plus aggregation state, not a recording-wide
  // matrix of every band. Refined F0 storage is unused until this phase ends.
  w.full_f0.resize(size_t(frames) * batch);
  w.raw.resize(size_t(frames) * (capacity / 7));
  check(cudaMemset(w.raw.p, 0, size_t(frames) * (capacity / 7) * sizeof(double)));
  w.aggregate_state.resize(frames);
  for (int offset = 0; offset < bands; offset += batch) {
    int count = std::min(batch, bands - offset);
    const double* band_boundaries = w.boundaries.p + offset;
    bool cache_band = offset < cache_bands;
    auto* filters = cache_band ? w.filter_spectra.p + size_t(offset) * bins : w.spectrum.p;
    bool need_filters = !cached || !cache_band;
    bool need_ends = segmented && (!ends_cached || !cache_band);
    if (need_filters || need_ends) {
      make_filters<<<dim3((transform_n + 255) / 256, count), 256>>>(w.real.p, transform_n, fs, band_boundaries);
      check(cudaGetLastError());
      if (need_ends) {
        filter_ends<<<count,128>>>(w.real.p, transform_n, n, fs, band_boundaries,
            w.filter_ends.p+offset);
        check(cudaGetLastError());
      }
      if (need_filters)
        check_fft(cufftExecD2Z(w.plan(transform_n, count, CUFFT_D2Z), w.real.p, filters));
    }
    if (segmented) {
      correct_filter_ends<<<(count+127)/128,128>>>(w.filter_ends.p+offset, w.signal_spectrum.p, n, count, w.corrections.p+offset);
      check(cudaGetLastError());
      convolve_blocks<<<unsigned((size_t(bins)*count*blocks+255)/256),256>>>(filters,w.aux_spectrum.p,transform_n,blocks,count,w.spectrum.p);
      check(cudaGetLastError());
      int active = count * blocks, padded = padded_batch(active);
      if (padded > active)
        check(cudaMemset(w.spectrum.p + size_t(bins) * active, 0,
            size_t(bins) * (padded - active) * sizeof(cufftDoubleComplex)));
      check_fft(cufftExecZ2D(w.plan(transform_n,padded,CUFFT_Z2D),w.spectrum.p,w.real.p));
    } else {
      auto* products = w.spectrum.p + size_t(bins) * count;
      convolve<<<dim3((bins + 255) / 256, count), 256>>>(filters, w.signal_spectrum.p, products, n);
      check(cudaGetLastError());
      check_fft(cufftExecZ2D(w.plan(n, count, CUFFT_Z2D), products, w.real.p));
    }
    extract_edges<<<dim3(tiles, count), 256>>>(w.real.p, length, tiles, transform_n,
        blocks, step, double(n) / transform_n, fs, band_boundaries, segmented,
        segmented ? w.corrections.p + offset : nullptr, w.convolution_phases.p,
        tiled_edges, w.edge_offsets.p);
    check(cudaGetLastError());
    scan_edge_counts<<<count * 4, 256>>>(w.edge_offsets.p, tiles, w.counts.p);
    check(cudaGetLastError());
    pack_edges<<<dim3(tiles, count * 4), 128>>>(tiled_edges, w.edge_offsets.p, tiles, stride, w.real.p);
    check(cudaGetLastError());
    interpolate_events<<<dim3((frames + 127) / 128, count), 128>>>(w.real.p, w.counts.p, w.edge_offsets.p, tiles, stride, frames, fs,
                                                               band_boundaries, floor, ceil, w.full_f0.p);
    check(cudaGetLastError());
    if (offset + count == bands) check(cudaMemset(w.counts.p, 0, sizeof(int)));
    aggregate_candidates<<<(frames + 127) / 128, 128>>>(w.full_f0.p, offset, count,
        bands, frames, capacity / 7, w.aggregate_state.p, w.raw.p, w.counts.p);
    check(cudaGetLastError());
  }
  int count = 0;
  check(cudaMemcpy(&count, w.counts.p, sizeof(int), cudaMemcpyDeviceToHost));
  if (cache_bands && !cached) {
    w.filter_boundaries.assign(boundaries, boundaries + bands);
    w.filter_fs = fs;
    w.filter_n = transform_n;
  }
  if (segmented) w.filter_global_n = n;
  return count * 7;
}

static void refine_chunk(GpuWorkspace& w, int length, double fs, int frames, int first_frame,
                         int candidates, double floor, double ceil) {
  size_t total = size_t(frames) * candidates;
  const double* input = w.candidate_f0.p;
  w.job_counts.resize(32);
  check(cudaMemset(w.job_counts.p, 0, 32 * sizeof(int)));
  w.owners.resize(total);
  w.selections.resize(total);
  count_jobs<<<frames, 128, size_t(candidates) * sizeof(int)>>>(input,
      candidates, fs, w.job_counts.p, w.owners.p, w.selections.p);
  check(cudaGetLastError());
  std::array<int, 32> counts{}, offsets{};
  check(cudaMemcpy(counts.data(), w.job_counts.p, sizeof(counts), cudaMemcpyDeviceToHost));
  int jobs = 0;
  for (int i = 0; i < 32; ++i) { offsets[i] = jobs; jobs += counts[i]; }
  w.jobs.resize(jobs);
  w.job_counts.upload(offsets.data(), offsets.size());
  make_jobs<<<unsigned((total + 255) / 256), 256>>>(input, frames, first_frame,
      candidates, fs, w.owners.p, w.job_counts.p, w.jobs.p);
  check(cudaGetLastError());
  w.aux_spectrum.resize(total * 6);
  constexpr int max_batch = 8192;
  for (int exponent = 0; exponent < 31; ++exponent) {
    if (!counts[exponent]) continue;
    int n = 1 << exponent;
    if (n <= local_fft_limit) {
      auto launch = [&](auto size) {
        constexpr int n = decltype(size)::value;
        refine_local<n><<<counts[exponent], (n <= 256 ? 32 : n == 512 ? 64 : 128), size_t(n) * sizeof(cufftDoubleComplex)>>>(
            w.waveform.p, length, w.jobs.p + offsets[exponent], candidates,
            w.owners.p, w.selections.p, w.twiddles.p, w.window_basis.p, w.aligned_windows.p,
            w.aux_spectrum.p);
      };
      switch (n) {
        case 32: launch(std::integral_constant<int, 32>{}); break;
        case 64: launch(std::integral_constant<int, 64>{}); break;
        case 128: launch(std::integral_constant<int, 128>{}); break;
        case 256: launch(std::integral_constant<int, 256>{}); break;
        case 512: launch(std::integral_constant<int, 512>{}); break;
        case 1024: launch(std::integral_constant<int, 1024>{}); break;
        case 2048: launch(std::integral_constant<int, 2048>{}); break;
      }
      check(cudaGetLastError());
      continue;
    }
    size_t memory_batch = std::max(size_t(1), (size_t(128) << 20) / (size_t(n) * 32));
    int batch = int(std::min({size_t(max_batch), (size_t(counts[exponent]) + 255) / 256 * 256, memory_batch}));
    w.real.resize(size_t(n) * 2 * batch);
    w.spectrum.resize(size_t(n / 2 + 1) * 2 * batch);
    for (size_t offset = 0; offset < size_t(counts[exponent]); offset += batch) {
      int count = int(std::min(size_t(batch), size_t(counts[exponent]) - offset));
      int active_batch = std::min(batch, (count + 255) / 256 * 256);
      auto plan = w.plan(n, 2 * active_batch, CUFFT_D2Z);
      auto* current = w.jobs.p + offsets[exponent] + offset;
      make_windows<<<active_batch, 256>>>(w.waveform.p, length, current, count, fs, n, w.real.p);
      check(cudaGetLastError());
      check_fft(cufftExecD2Z(plan, w.real.p, w.spectrum.p));
      collect_bins<<<(count + 3) / 4, 128>>>(w.spectrum.p, current, count, n, candidates,
          w.owners.p, w.selections.p, w.aux_spectrum.p);
      check(cudaGetLastError());
    }
  }
  score_bins<<<unsigned((total + 127) / 128), 128>>>(w.aux_spectrum.p,
      input, w.selections.p, frames, candidates, fs, floor, ceil,
      w.full_f0.p + size_t(first_frame) * candidates,
      w.full_scores.p + size_t(first_frame) * candidates);
  check(cudaGetLastError());
}

static void gpu_refine(GpuWorkspace& w, int length, double fs, int frames,
                       int candidates, int stride, double floor, double ceil,
                       Candidates& result, double* best) {
  result.offsets.allocate(size_t(frames) + 1);
  if (!candidates) {
    std::fill_n(best, frames, 0.0);
    std::fill(result.offsets.begin(), result.offsets.end(), 0);
    result.f0.allocate(0);
    result.scores.allocate(0);
    return;
  }
  size_t total = size_t(frames) * candidates;
  w.full_f0.resize(total);
  w.full_scores.resize(total);
  if (!w.twiddles.capacity) {
    // WORLD zero-pads to more than twice the window length: half < n / 4.
    // All small-window bases together occupy a fixed 4 MiB.
    constexpr int halves = local_fft_limit / 4;
    w.window_basis.resize(size_t(halves) * halves);
    make_local_windows<<<dim3((2 * halves + 255) / 256, halves), 256>>>(w.window_basis.p);
    check(cudaGetLastError());
    w.aligned_windows.resize(size_t(halves) * halves);
    make_aligned_windows<<<dim3((2 * halves + 255) / 256, halves), 256>>>(w.window_basis.p, w.aligned_windows.p);
    check(cudaGetLastError());
    w.twiddles.resize(local_fft_limit / 2);
    make_twiddles<<<(local_fft_limit / 2 + 255) / 256, 256>>>(w.twiddles.p);
    check(cudaGetLastError());
  }
  // Limit transient harmonic data to 32 MiB. Expand neighboring candidates
  // per chunk using absolute frames; prune only after all chunks finish.
  int chunk_frames = int(std::min(size_t(4096), (size_t(32) << 20) /
      (size_t(candidates) * 6 * sizeof(cufftDoubleComplex))));
  chunk_frames = std::max(1, chunk_frames);
  w.candidate_f0.resize(size_t(chunk_frames) * candidates);
  for (int first = 0; first < frames;) {
    int count = std::min(chunk_frames, frames - first);
    overlap_candidates<<<unsigned((size_t(count) * candidates + 255) / 256), 256>>>(
        w.raw.p, frames, stride / 7, first, count, candidates, w.candidate_f0.p);
    check(cudaGetLastError());
    refine_chunk(w, length, fs, count, first, candidates, floor, ceil);
    first += count;
  }
  // Chunk input storage now holds pruned candidates. Original candidates are
  // no longer needed, so reuse their storage for best F0 and packed scores;
  // refined F0 storage holds the packed F0 values.
  w.raw.resize(size_t(frames) + total);
  w.candidate_f0.resize(total);
  w.candidate_offsets.resize(size_t(frames) + 1);
  prune_candidates<<<frames, 32>>>(w.full_f0.p, w.candidate_f0.p, w.full_scores.p,
      frames, candidates, w.raw.p, w.candidate_offsets.p);
  check(cudaGetLastError());
  size_t scan_bytes = 0;
  check(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, w.candidate_offsets.p,
      w.candidate_offsets.p, frames + 1));
  // All FFTs are finished; scan can use their scratch area on the same stream.
  w.fft_work.resize(scan_bytes);
  check(cub::DeviceScan::ExclusiveSum(w.fft_work.p, scan_bytes, w.candidate_offsets.p,
      w.candidate_offsets.p, frames + 1));
  pack_candidates<<<frames, 32>>>(w.candidate_f0.p, w.full_scores.p, candidates,
      w.candidate_offsets.p, w.full_f0.p, w.raw.p + frames);
  check(cudaGetLastError());
  check(cudaMemcpy(result.offsets.data(), w.candidate_offsets.p,
      (size_t(frames) + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  size_t packed = result.offsets.back();
  result.f0.allocate(packed);
  result.scores.allocate(packed);
  if (packed) {
    check(cudaMemcpy(result.f0.data(), w.full_f0.p, packed * sizeof(double), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(result.scores.data(), w.raw.p + frames, packed * sizeof(double), cudaMemcpyDeviceToHost));
  }
  check(cudaMemcpy(best, w.raw.p, size_t(frames) * sizeof(double), cudaMemcpyDeviceToHost));
}

void gpu_analyze(GpuWorkspace& w, const double* y, int length, int n, double fs,
                const double* boundaries, int bands, int frames, double floor,
                double ceil, int capacity, Candidates& candidates, double* best, Timings* timings) {
  CudaContextGuard context(w.context.get());
  using Clock = std::chrono::steady_clock;
  Clock::time_point start{};
  if (timings) start = Clock::now();
  int count = gpu_candidates(w, y, length, n, fs, boundaries, bands, frames, floor, ceil, capacity);
  if (timings) {
    check(cudaDeviceSynchronize());
    auto now = Clock::now();
    (*timings)["filter_events"] = std::chrono::duration<double, std::milli>(now - start).count();
    start = now;
  }
  size_t total = size_t(frames) * count;
  if (total > INT_MAX) throw std::invalid_argument("too many refinement candidates");
  gpu_refine(w, length, fs, frames, count, capacity, floor, ceil, candidates, best);
  if (timings) (*timings)["refine"] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}
