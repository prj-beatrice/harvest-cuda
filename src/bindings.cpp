#include <nanobind/nanobind.h>
#include "engine.h"
#include "workspace_cache.h"
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <cmath>
#include <climits>
#include <memory>
#include <optional>
#include <stdexcept>

namespace nb = nanobind;
using Input = nb::ndarray<const double, nb::numpy, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Output = nb::ndarray<nb::numpy, double, nb::ndim<1>>;
using namespace nb::literals;

static void validate(Input x, int fs, double floor, double ceil, double period) {
  if (x.size() == 0 || x.size() > INT_MAX)
    throw std::invalid_argument("x must be nonempty and contain at most INT_MAX samples");
  if (fs < 8000 || fs > 192000)
    throw std::invalid_argument("fs must be between 8000 and 192000 Hz");
  if (!std::isfinite(floor) || !std::isfinite(ceil) || floor <= 0 || ceil <= floor)
    throw std::invalid_argument("require finite 0 < f0_floor < f0_ceil");
  int ratio = std::min(12, std::max(1, int(fs / 8000.0 + 0.5)));
  if (x.size() < size_t(2 * ratio))
    throw std::invalid_argument("x is too short: require at least two samples after internal decimation");
  if (ceil >= double(fs) / ratio / 2)
    throw std::invalid_argument("f0_ceil must be below the internal Nyquist frequency");
  if (!std::isfinite(period) || period <= 0)
    throw std::invalid_argument("frame_period must be finite and positive (milliseconds)");
  if (1000.0 * x.size() / fs / period >= INT_MAX ||
      1000.0 * x.size() / fs >= INT_MAX ||
      double(x.size()) / ratio + 10.0 * fs / ratio / floor >= (1 << 28))
    throw std::invalid_argument("input or analysis settings exceed supported allocation sizes");
  bool finite = true;
  for (size_t i = 0; i < x.size(); ++i) finite &= std::isfinite(x.data()[i]);
  if (!finite) throw std::invalid_argument("x must contain only finite samples");
}

static Output array(std::vector<double>&& values) {
  auto owner = std::make_unique<std::vector<double>>(std::move(values));
  auto* ptr = owner.get();
  nb::capsule capsule(ptr, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  owner.release();
  return Output(ptr->data(), {ptr->size()}, capsule);
}

NB_MODULE(_core, m) {
  using harvest_cuda::GpuWorkspaceCache;
  harvest_cuda::initialize_host_cache();
  // Python owns the cache through these functions, so it is destroyed before
  // native CUDA teardown on Windows, including when no explicit clear is used.
  auto cache = std::make_unique<GpuWorkspaceCache>();
  nb::capsule owner(cache.get(), [](void* p) noexcept {
    nb::gil_scoped_release release;
    delete static_cast<GpuWorkspaceCache*>(p);
  });
  cache.release();
  m.def("clear_cache", [owner] {
    auto* cache = static_cast<GpuWorkspaceCache*>(owner.data());
    nb::gil_scoped_release release;
    cache->clear();
    harvest_cuda::clear_host_cache();
  }, "Release cached CPU and GPU working memory across threads and devices. In-use GPU workspaces are released when their calls finish. Does not reset CUDA.");
  m.def("cuda_available", &harvest_cuda::cuda_available, "device"_a = 0,
        "Whether the selected visible GPU supports Harvest (compute capability 7.5+). Does not guarantee sufficient free memory.");
  m.def("harvest", [owner](Input x, int fs, double floor, double ceil, double period, int device) {
    validate(x, fs, floor, ceil, period);
    auto* cache = static_cast<GpuWorkspaceCache*>(owner.data());
    harvest_cuda::Result result;
    {
      nb::gil_scoped_release release;
      GpuWorkspaceCache::Lease workspace(*cache, device);
      result = harvest_cuda::analyze(x.data(), int(x.size()), fs, floor, ceil, period, workspace.get());
    }
    return nb::make_tuple(array(std::move(result.f0)), array(std::move(result.times)));
  }, "x"_a.noconvert(), "fs"_a, "f0_floor"_a = 71.0, "f0_ceil"_a = 800.0,
     "frame_period"_a = 5.0, nb::kw_only(), "device"_a = 0,
     "Estimate F0 on CUDA. x: contiguous 1D float64 NumPy array; returns (f0, times). device indexes GPUs visible through CUDA_VISIBLE_DEVICES.");
  m.def("_reference", [](Input x, int fs, double floor, double ceil, double period) {
    validate(x, fs, floor, ceil, period);
    harvest_cuda::Result result;
    {
      nb::gil_scoped_release release;
      result = harvest_cuda::reference(x.data(), int(x.size()), fs, floor, ceil, period);
    }
    return nb::make_tuple(array(std::move(result.f0)), array(std::move(result.times)));
  }, "x"_a.noconvert(), "fs"_a, "f0_floor"_a = 71.0, "f0_ceil"_a = 800.0, "frame_period"_a = 5.0);
  m.def("_profile", [owner](Input x, int fs, double floor, double ceil, double period, bool gpu, int device) {
    validate(x, fs, floor, ceil, period);
    harvest_cuda::Result result;
    harvest_cuda::Timings timings;
    auto* cache = static_cast<GpuWorkspaceCache*>(owner.data());
    {
      nb::gil_scoped_release release;
      std::optional<GpuWorkspaceCache::Lease> workspace;
      if (gpu) workspace.emplace(*cache, device);
      result = harvest_cuda::analyze(x.data(), int(x.size()), fs, floor, ceil, period,
                                    workspace ? workspace->get() : nullptr, &timings);
    }
    return nb::make_tuple(array(std::move(result.f0)), array(std::move(result.times)), timings);
  }, "x"_a.noconvert(), "fs"_a, "f0_floor"_a = 71.0, "f0_ceil"_a = 800.0,
     "frame_period"_a = 5.0, "gpu"_a = true, nb::kw_only(), "device"_a = 0);
}
