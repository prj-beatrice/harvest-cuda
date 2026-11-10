#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace harvest_cuda {
inline void check_cuda(cudaError_t code) {
  if (code == cudaSuccess) return;
  // Driver-side runtimes can return null descriptions when no driver is present.
  const char* name = cudaGetErrorName(code);
  const char* detail = cudaGetErrorString(code);
  std::string message = "CUDA: ";
  message += name ? name : std::to_string(int(code));
  if (detail) message += std::string(": ") + detail;
  if (code == cudaErrorInsufficientDriver)
    message += ". Install an NVIDIA driver compatible with CUDA 13.";
  else if (code == cudaErrorNoDevice)
    message += ". Check the NVIDIA driver and CUDA_VISIBLE_DEVICES before starting Python.";
  else if (code == cudaErrorMemoryAllocation)
    message += ". GPU memory allocation failed; try shorter audio or release unused GPU allocations.";
  throw std::runtime_error(message);
}

// Indices follow CUDA_VISIBLE_DEVICES filtering/reordering.
inline void check_harvest_device(int device) {
  if (device < 0) throw std::invalid_argument("device must be a nonnegative visible GPU index");
  int count = 0;
  check_cuda(cudaGetDeviceCount(&count));
  if (!count) check_cuda(cudaErrorNoDevice);
  if (device >= count)
    throw std::invalid_argument("device " + std::to_string(device) + " is out of range; " +
        std::to_string(count) + " GPU(s) are visible through CUDA_VISIBLE_DEVICES");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device));
  if (properties.major * 10 + properties.minor < 75)
    throw std::runtime_error("Harvest requires compute capability 7.5+; device " + std::to_string(device) + " (" +
        properties.name + ") has " + std::to_string(properties.major) + "." +
        std::to_string(properties.minor) + ". Select a supported GPU with CUDA_VISIBLE_DEVICES before starting Python.");
}

// Resolve through cudart so building a wheel does not require a driver library
// or driver stubs on the build machine. These signatures are stable since 11.0.
template<class Function> Function driver_entry(const char* name) {
  void* pointer = nullptr;
  auto status = cudaGetDriverEntryPointByVersion(name, &pointer, 11000, cudaEnableDefault);
  if (status != cudaSuccess || !pointer)
    throw std::runtime_error(std::string("CUDA driver entry unavailable: ") + name);
  return reinterpret_cast<Function>(pointer);
}
struct CudaDriverApi {
  decltype(&cuInit) init = driver_entry<decltype(init)>("cuInit");
  decltype(&cuDevicePrimaryCtxRetain) retain = driver_entry<decltype(retain)>("cuDevicePrimaryCtxRetain");
  decltype(&cuDevicePrimaryCtxRelease) release = driver_entry<decltype(release)>("cuDevicePrimaryCtxRelease");
  decltype(&cuCtxGetCurrent) get_current = driver_entry<decltype(get_current)>("cuCtxGetCurrent");
  decltype(&cuCtxSetCurrent) set_current = driver_entry<decltype(set_current)>("cuCtxSetCurrent");
  decltype(&cuGetErrorString) error_string = driver_entry<decltype(error_string)>("cuGetErrorString");
};
inline const CudaDriverApi& driver_api() {
  static const CudaDriverApi api;
  return api;
}

inline void check_driver(CUresult code) {
  if (code != CUDA_SUCCESS) {
    const char* message = nullptr;
    driver_api().error_string(code, &message);
    throw std::runtime_error(std::string("CUDA driver: ") +
                             (message ? message : "unknown error"));
  }
}

// Keep the runtime's primary context alive for as long as its cached resources.
class CudaPrimaryContext {
 public:
  explicit CudaPrimaryContext(int device) : device_(device) {
    check_driver(driver_api().init(0));
    check_driver(driver_api().retain(&context_, device_));
  }
  ~CudaPrimaryContext() noexcept { driver_api().release(device_); }
  CudaPrimaryContext(const CudaPrimaryContext&) = delete;
  CudaPrimaryContext& operator=(const CudaPrimaryContext&) = delete;
  CUcontext get() const noexcept { return context_; }

 private:
  int device_;
  CUcontext context_ = nullptr;
};

// Preserve the exact caller context, including a custom context on the same
// device or no current context, on normal return and exception unwinding.
class CudaContextGuard {
 public:
  explicit CudaContextGuard(CUcontext target) {
    check_driver(driver_api().get_current(&previous_));
    switched_ = previous_ != target;
    if (switched_) check_driver(driver_api().set_current(target));
  }
  ~CudaContextGuard() noexcept {
    if (switched_) driver_api().set_current(previous_);
  }
  CudaContextGuard(const CudaContextGuard&) = delete;
  CudaContextGuard& operator=(const CudaContextGuard&) = delete;

 private:
  CUcontext previous_ = nullptr;
  bool switched_ = false;
};
}
