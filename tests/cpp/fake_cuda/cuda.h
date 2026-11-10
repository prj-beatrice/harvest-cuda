#pragma once
#include <array>
#include <vector>

using CUresult = int;
using CUcontext = int*;
constexpr CUresult CUDA_SUCCESS = 0;
inline std::array<int, 3> primary_contexts{};
inline std::array<int, 3> references{};
inline int custom_context = 0;
inline CUcontext current_context = &custom_context;
inline bool fail_get = false, fail_retain = false;
inline CUcontext fail_set = nullptr;
inline std::vector<CUcontext> switches;

inline CUresult cuInit(unsigned) { return CUDA_SUCCESS; }
inline CUresult cuDevicePrimaryCtxRetain(CUcontext* context, int device) {
  if (fail_retain) return 1;
  *context = &primary_contexts.at(device);
  ++references.at(device);
  return CUDA_SUCCESS;
}
inline CUresult cuDevicePrimaryCtxRelease(int device) {
  --references.at(device);
  return CUDA_SUCCESS;
}
inline CUresult cuCtxGetCurrent(CUcontext* context) {
  if (fail_get) return 1;
  *context = current_context;
  return CUDA_SUCCESS;
}
inline CUresult cuCtxSetCurrent(CUcontext context) {
  switches.push_back(context);
  if (context && context == fail_set) return 1;
  current_context = context;
  return CUDA_SUCCESS;
}
inline CUresult cuGetErrorString(CUresult, const char** message) {
  *message = "injected failure";
  return CUDA_SUCCESS;
}
