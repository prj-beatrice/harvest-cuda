#include "cuda_device.h"
#include <stdexcept>

using harvest_cuda::CudaPrimaryContext;
using harvest_cuda::CudaContextGuard;
void require(bool condition) {
  if (!condition) throw std::runtime_error("Context guard test failed");
}
int main() {
  // Availability follows visible device indices, rejects unsupported GPUs,
  // and explains environment failures before attempting context creation.
  auto fails_with = [](auto operation, const char* message) {
    bool failed = false;
    try { operation(); }
    catch (const std::exception& error) {
      failed = std::string(error.what()).find(message) != std::string::npos;
    }
    require(failed);
  };
  harvest_cuda::check_harvest_device(0);
  device_properties[0].minor = 0;
  fails_with([] { harvest_cuda::check_harvest_device(0); }, "test GPU) has 7.0");
  device_properties[0].major = 8;
  harvest_cuda::check_harvest_device(0);
  visible_device_count = 0;
  fails_with([] { harvest_cuda::check_harvest_device(0); }, "CUDA_VISIBLE_DEVICES");
  visible_device_count = 1;
  device_status = cudaErrorInsufficientDriver;
  fails_with([] { harvest_cuda::check_harvest_device(0); }, "driver compatible with CUDA 13");
  error_strings_available = false;
  fails_with([] { harvest_cuda::check_harvest_device(0); }, "CUDA: 35");
  fails_with([] { harvest_cuda::check_harvest_device(0); }, "driver compatible with CUDA 13");
  error_strings_available = true;
  device_status = cudaSuccess;
  fails_with([] { harvest_cuda::check_cuda(cudaErrorMemoryAllocation); }, "GPU memory allocation failed");
  fails_with([] { harvest_cuda::check_harvest_device(-1); }, "nonnegative");
  fails_with([] { harvest_cuda::check_harvest_device(1); }, "out of range");
  visible_device_count = 2;
  device_properties[0].major = 6;
  harvest_cuda::check_harvest_device(1);
  require(last_queried_device == 1);
  fails_with([] { harvest_cuda::check_harvest_device(0); }, "device 0");
  device_properties[0].major = 8;

  auto custom = &custom_context;
  auto primary = &primary_contexts[0];
  {
    CudaPrimaryContext owner(0);
    require(references[0] == 1 && current_context == custom);
    {
      CudaContextGuard guard(owner.get());
      require(current_context == primary);
      {
        CudaPrimaryContext other(2);
        CudaContextGuard nested(other.get());
        require(current_context == &primary_contexts[2]);
      }
      require(current_context == primary && references[2] == 0);
    }
    require(current_context == custom);
    require(switches == std::vector<CUcontext>({primary, &primary_contexts[2], primary, custom}));

    switches.clear();
    try {
      CudaContextGuard guard(primary);
      throw std::runtime_error("processing failure");
    } catch (const std::runtime_error&) {}
    require(current_context == custom);
    require(switches == std::vector<CUcontext>({primary, custom}));

    switches.clear();
    current_context = primary;
    { CudaContextGuard guard(primary); }
    require(switches.empty());

    current_context = nullptr;
    { CudaContextGuard guard(primary); }
    require(current_context == nullptr);
    require(switches == std::vector<CUcontext>({primary, nullptr}));

    switches.clear();
    current_context = custom;
    fail_get = true;
    bool threw = false;
    try { CudaContextGuard guard(primary); } catch (const std::runtime_error&) { threw = true; }
    require(threw && current_context == custom && switches.empty());
    fail_get = false;

    fail_set = primary;
    threw = false;
    try { CudaContextGuard guard(primary); } catch (const std::runtime_error&) { threw = true; }
    require(threw && current_context == custom);
    require(switches == std::vector<CUcontext>({primary}));
    fail_set = nullptr;
  }
  require(references[0] == 0);
  fail_retain = true;
  bool threw = false;
  try { CudaPrimaryContext owner(0); } catch (const std::runtime_error&) { threw = true; }
  require(threw && references[0] == 0);
}
