#pragma once
#include "cuda.h"
#include <cstring>

constexpr int cudaSuccess = 0;
using cudaError_t = int;
constexpr int cudaErrorInsufficientDriver = 35;
constexpr int cudaErrorNoDevice = 100;
constexpr int cudaErrorMemoryAllocation = 2;
struct cudaDeviceProp {
  char name[256] = "test GPU";
  int major = 7, minor = 5;
};
inline std::array<cudaDeviceProp, 3> device_properties;
inline int last_queried_device = -1;
inline int visible_device_count = 1;
inline cudaError_t device_status = cudaSuccess;
inline bool error_strings_available = true;
inline const char* cudaGetErrorName(cudaError_t) {
  return error_strings_available ? "test_cuda_error" : nullptr;
}
inline const char* cudaGetErrorString(cudaError_t) {
  return error_strings_available ? "test error description" : nullptr;
}
inline cudaError_t cudaGetDeviceCount(int* count) {
  *count = visible_device_count;
  return device_status;
}
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* properties, int device) {
  if (device < 0 || device >= visible_device_count) return cudaErrorNoDevice;
  last_queried_device = device;
  *properties = device_properties.at(device);
  return device_status;
}
constexpr unsigned long long cudaEnableDefault = 0;
inline int cudaGetDriverEntryPointByVersion(const char* name, void** pointer,
                                          unsigned, unsigned long long) {
  *pointer = nullptr;
#define FAKE_ENTRY(function) \
  if (std::strcmp(name, #function) == 0) *pointer = reinterpret_cast<void*>(&function)
  FAKE_ENTRY(cuInit);
  FAKE_ENTRY(cuDevicePrimaryCtxRetain);
  FAKE_ENTRY(cuDevicePrimaryCtxRelease);
  FAKE_ENTRY(cuCtxGetCurrent);
  FAKE_ENTRY(cuCtxSetCurrent);
  FAKE_ENTRY(cuGetErrorString);
#undef FAKE_ENTRY
  return cudaSuccess;
}
