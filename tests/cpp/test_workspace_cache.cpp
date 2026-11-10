#include "workspace_cache.h"
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <stdexcept>
#include <thread>

namespace harvest_cuda {
struct GpuWorkspace { int device, id; };
std::array<std::atomic<int>, 2> live{};
std::atomic<int> created{0};
std::function<void()> on_create, on_delete;
GpuWorkspace* create_gpu_workspace(int device) {
  if (device < 0 || device >= 2) throw std::invalid_argument("invalid device");
  if (on_create) on_create();
  auto* result = new GpuWorkspace{device, ++created};
  ++live[device];
  return result;
}
void delete_gpu_workspace(GpuWorkspace* workspace) noexcept {
  if (!workspace) return;
  --live[workspace->device];
  delete workspace;
  if (on_delete) on_delete();
}
}

using namespace harvest_cuda;
using Lease = GpuWorkspaceCache::Lease;
void require(bool condition) {
  if (!condition) throw std::runtime_error("Workspace cache test failed");
}

int main() {
  {
    GpuWorkspaceCache cache;
    int original;
    { Lease lease(cache, 0); original = lease.get()->id; }
    // A different worker and switching GPUs reuse the existing idle resources.
    std::async(std::launch::async, [&] {
      { Lease lease(cache, 0); require(lease.get()->id == original); }
      { Lease lease(cache, 1); require(lease.get()->device == 1); }
      { Lease lease(cache, 0); require(lease.get()->id == original); }
    }).get();
    require(live[0] == 1 && live[1] == 1);
    bool failed = false;
    try { Lease lease(cache, 2); } catch (const std::invalid_argument&) { failed = true; }
    require(failed);
    { Lease lease(cache, 0); require(lease.get()->id == original); }
    // Central cleanup also reaches idle resources returned by another worker.
    std::async(std::launch::async, [&] { cache.clear(); }).get();
    require(live[0] == 0 && live[1] == 0);
    cache.clear();
    { Lease lease(cache, 0); require(lease.get()->id != original); }
  }
  require(live[0] == 0 && live[1] == 0);
  {
    GpuWorkspaceCache cache;
    // A busy GPU reuses its workspace once available. Other GPUs do not wait.
    std::promise<void> started;
    std::future<int> waiting;
    int original;
    {
      Lease first(cache, 0);
      original = first.get()->id;
      waiting = std::async(std::launch::async, [&] {
        started.set_value();
        Lease second(cache, 0);
        return second.get()->id;
      });
      started.get_future().wait();
      require(waiting.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout);
      { Lease other(cache, 1); require(other.get()->device == 1); }
      require(live[0] == 1);
    }
    require(waiting.get() == original);
    require(live[0] == 1);
    {
      Lease in_use(cache, 0);
      std::async(std::launch::async, [&] { cache.clear(); }).get();
      require(live[0] == 1 && in_use.get()->device == 0);
      { Lease newer(cache, 0); require(newer.get() != in_use.get()); }
      require(live[0] == 2);
    }
    // The old call cannot repopulate a cache cleared while it was running.
    require(live[0] == 1);
    cache.clear();
    require(live[0] == 0);
    try {
      Lease lease(cache, 0);
      throw std::runtime_error("analysis failed");
    } catch (const std::runtime_error&) {}
    require(live[0] == 1);
    // Resource destruction must not hold the pool lock.
    on_delete = [&] { cache.clear(); };
    cache.clear();
    on_delete = {};
  }
  require(live[0] == 0);
  {
    GpuWorkspaceCache cache;
    std::promise<void> creating, proceed;
    auto resume = proceed.get_future();
    on_create = [&] { creating.set_value(); resume.wait(); };
    auto worker = std::async(std::launch::async, [&] { Lease lease(cache, 0); });
    creating.get_future().wait();
    cache.clear();
    proceed.set_value();
    worker.get();
    on_create = {};
    require(live[0] == 0);
  }
  {
    GpuWorkspaceCache cache;
    std::array<std::future<void>, 4> workers;
    for (int i = 0; i < 4; ++i)
      workers[i] = std::async(std::launch::async, [&, i] {
        for (int call = 0; call < 500; ++call) {
          Lease lease(cache, i % 2);
          require(lease.get()->device == i % 2);
          if (call % 17 == 0) cache.clear();
          std::this_thread::yield();
        }
      });
    for (auto& worker : workers) worker.get();
    require(live[0] <= 1 && live[1] <= 1);
  }
  require(live[0] == 0 && live[1] == 0);
}
