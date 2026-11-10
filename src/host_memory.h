#pragma once
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

namespace harvest_cuda {
// Borrowed arena for one call. Allocations that do not fit use temporary heap
// storage instead; the arena itself never grows.
struct HostArena {
  void* next = nullptr;
  size_t remaining = 0;
  void* allocate(size_t bytes, size_t alignment) {
    if (!next || !std::align(alignment, bytes, next, remaining)) return nullptr;
    void* result = next;
    next = static_cast<std::byte*>(next) + bytes;
    remaining -= bytes;
    return result;
  }
};

// Scratch values are overwritten before use. Unlike vector::resize, allocate
// does not zero-initialize or preserve old contents.
template<class T> class HostArray {
  static_assert(std::is_arithmetic_v<T>);
  HostArena* arena_;
  std::unique_ptr<T[]> owned_;
  T* data_ = nullptr;
  size_t size_ = 0;
 public:
  explicit HostArray(HostArena* arena = nullptr) : arena_(arena) {}
  void allocate(size_t n) {
    owned_.reset();
    data_ = nullptr;
    size_ = 0;
    if (!n) return;
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_array_new_length();
    if (arena_) data_ = static_cast<T*>(arena_->allocate(n * sizeof(T), alignof(T)));
    if (data_) std::uninitialized_default_construct_n(data_, n);
    else {
      owned_.reset(new T[n]);
      data_ = owned_.get();
    }
    size_ = n;
  }
  T* data() { return data_; }
  const T* data() const { return data_; }
  T* begin() { return data_; }
  T* end() { return size_ ? data_ + size_ : data_; }
  T& back() { return data_[size_ - 1]; }
  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }
};
}
