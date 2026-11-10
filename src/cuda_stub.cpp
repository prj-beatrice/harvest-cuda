#include "engine.h"
#include <stdexcept>
namespace harvest_cuda {
bool cuda_available(int) noexcept { return false; }
GpuWorkspace* create_gpu_workspace(int) {
  throw std::runtime_error("This development build has no CUDA backend. Rebuild with HARVEST_CUDA_ENABLE_CUDA=ON.");
}
void delete_gpu_workspace(GpuWorkspace*) noexcept {}
void gpu_decimate(GpuWorkspace&, const double*, int, int, double*, int) { create_gpu_workspace(0); }
void gpu_analyze(GpuWorkspace&, const double*, int, int, double, const double*, int, int,
                double, double, int, Candidates&, double*, Timings*) { create_gpu_workspace(0); }
}
