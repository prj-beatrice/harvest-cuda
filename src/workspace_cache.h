#pragma once
#include "engine.h"
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace harvest_cuda {
// One workspace per GPU. Calls on that GPU take turns using it, avoiding costly
// repeated allocation in worker pools. Different GPUs have independent locks.
class GpuWorkspaceCache {
  struct Delete {
    void operator()(GpuWorkspace* p) const noexcept { delete_gpu_workspace(p); }
  };
  using Workspace = std::unique_ptr<GpuWorkspace, Delete>;
  struct Slot {
    std::mutex mutex;
    Workspace workspace;
  };
  std::mutex mutex_;
  std::map<int, std::shared_ptr<Slot>> slots_;
  uint64_t generation_ = 0;

 public:
  class Lease {
    // Unlock before releasing the final owner and destroying CUDA resources.
    std::shared_ptr<Slot> slot_;
    std::unique_lock<std::mutex> lock_;

   public:
    Lease(GpuWorkspaceCache& cache, int device) {
      if (device < 0)
        throw std::invalid_argument("device must be a nonnegative visible GPU index");
      uint64_t generation;
      {
        std::lock_guard<std::mutex> lock(cache.mutex_);
        generation = cache.generation_;
        auto it = cache.slots_.find(device);
        if (it != cache.slots_.end()) slot_ = it->second;
      }
      if (!slot_) {
        auto created = std::make_shared<Slot>();
        created->workspace.reset(create_gpu_workspace(device));
        {
          std::lock_guard<std::mutex> lock(cache.mutex_);
          // Do not cache a resource if clear() ran during its creation.
          slot_ = generation == cache.generation_
              ? cache.slots_.try_emplace(device, created).first->second : created;
        }
        // Losing a concurrent insertion destroys the unused resource here,
        // outside the cache lock. Invalid devices never enter the map.
      }
      lock_ = std::unique_lock<std::mutex>(slot_->mutex);
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    GpuWorkspace* get() const noexcept { return slot_->workspace.get(); }
  };

  void clear() {
    std::map<int, std::shared_ptr<Slot>> retired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++generation_;
      retired.swap(slots_);
    }
    // In-use workspaces remain valid and are discarded when their leases end.
    // Destruction may synchronize CUDA; keep it outside the cache lock.
  }
};
}
