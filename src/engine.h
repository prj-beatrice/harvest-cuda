#pragma once
#include "host_memory.h"
#include <map>
#include <string>
#include <vector>

namespace harvest_cuda {
struct Result {
  std::vector<double> f0, times;
};
// Nonzero candidates in original order, from offsets[i] to offsets[i+1] per frame.
struct Candidates {
  HostArray<double> f0, scores;
  HostArray<int> offsets;
  explicit Candidates(HostArena* arena = nullptr) : f0(arena), scores(arena), offsets(arena) {}
};
void initialize_host_cache();
void clear_host_cache();
// Borrowed exclusively from the module-owned cache for one analysis call.
struct GpuWorkspace;
GpuWorkspace* create_gpu_workspace(int device);
void delete_gpu_workspace(GpuWorkspace*) noexcept;
using Timings = std::map<std::string, double>;
Result analyze(const double* x, int length, int fs, double floor, double ceil,
               double period, GpuWorkspace* workspace, Timings* timings = nullptr);
Result reference(const double* x, int length, int fs, double floor, double ceil,
                 double period);
bool cuda_available(int device) noexcept;
void gpu_decimate(GpuWorkspace& workspace, const double* x, int length, int ratio, double* y, int y_length);
// Filter, aggregate, refine, prune and compact on device. best receives one
// F0 per frame; only nonzero candidates are copied to host storage.
void gpu_analyze(GpuWorkspace& workspace, const double* y, int length, int fft_size, double fs,
                const double* boundaries, int bands, int frames, double floor,
                double ceil, int capacity, Candidates& candidates, double* best, Timings* timings);
}
