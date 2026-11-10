#include "engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <vector>
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

// Keep the pinned upstream source untouched. Including it gives this adapter
// access to the exact internal routines and retains Harvest as an oracle.
#include "../third_party/world/src/harvest.cpp"

namespace harvest_cuda {
namespace {
constexpr size_t host_cache_bytes = size_t(20) << 20;
struct HostCache {
  std::mutex mutex;
  std::unique_ptr<std::byte[]> storage;
  void initialize() {
    if (!storage) storage.reset(new std::byte[host_cache_bytes]{});
  }
};
HostCache& host_cache() {
  static HostCache cache;
  return cache;
}
struct HostLease {
  std::unique_lock<std::mutex> lock;
  HostArena arena;
  HostLease() : lock(host_cache().mutex, std::try_to_lock) {
    // One process-wide arena. Concurrent callers use temporary storage rather
    // than waiting or retaining another cache for each calling thread.
    if (lock.owns_lock()) {
      auto& cache = host_cache();
      cache.initialize();
      arena = {cache.storage.get(), host_cache_bytes};
    }
  }
};
struct Matrix {
  std::vector<double> storage;
  std::vector<double*> rows;
  Matrix(int n, int m) : storage(size_t(n) * m, 0.0), rows(n) {
    for (int i = 0; i < n; ++i) rows[i] = storage.data() + size_t(i) * m;
  }
  double** data() { return rows.data(); }
};
using Clock = std::chrono::steady_clock;

// Extend a voiced section without allocating an index list. Avoid division
// for clearly worse candidates, retaining WORLD's rounded comparisons and
// its preference for the last candidate when errors tie.
int extend_contour(double* values, int origin, int last, int shift,
                   const Candidates& candidates, int frame_offset = 0) {
  double reference = values[origin];
  int end = origin, misses = 0;
  for (int frame = origin; shift > 0 ? frame <= last : frame >= last; frame += shift) {
    double best = 0, distance = reference * 0.18;
    int row = frame_offset + frame + shift;
    for (int j = candidates.offsets[row]; j < candidates.offsets[row + 1]; ++j) {
      double value = candidates.f0[j];
      double difference = fabs(reference - value);
      double margin = distance * 0x1p-49;
      if (difference > distance + margin) continue;
      // Only almost-equal errors require the original rounded division.
      if (difference >= distance - margin) {
        double error = best == 0 ? 0.18 : distance / reference;
        if (difference / reference > error) continue;
      }
      best = value;
      distance = difference;
    }
    values[frame + shift] = best;
    if (best == 0) {
      if (++misses == 4) break;
    } else {
      reference = best;
      misses = 0;
      end = frame + shift;
    }
  }
  return end;
}

// Match scores two candidates at a time on x86-64; other hosts use the
// same scalar comparisons. MAX's operand order ignores NaNs as WORLD does.
double search_score(double value, const double* candidates, const double* scores, int count) {
  int i = 0;
  double result = 0;
#if defined(__SSE2__) || defined(_M_X64)
  __m128d target = _mm_set1_pd(value);
  __m128d best[2] = {_mm_setzero_pd(), _mm_setzero_pd()};
  for (; i + 4 <= count; i += 4)
    for (int j = 0; j < 2; ++j) {
      __m128d mask = _mm_cmpeq_pd(target, _mm_loadu_pd(candidates + i + 2 * j));
      __m128d score = _mm_and_pd(mask, _mm_loadu_pd(scores + i + 2 * j));
      best[j] = _mm_max_pd(score, best[j]);
    }
  double lanes[2];
  _mm_storeu_pd(lanes, _mm_max_pd(best[0], best[1]));
  result = std::max(lanes[0], lanes[1]);
#endif
  for (; i < count; ++i)
    if (value == candidates[i] && result < scores[i]) result = scores[i];
  return result;
}

// A section can extend by at most 101 frames in either direction. Keep only
// that neighborhood instead of a full-recording array for every section.
void connect_contour(const double* input, int frames, const Candidates& candidates,
                     double* output) {
  struct Section {
    int offset, start, end;
    std::vector<double> values;
    double at(int frame) const {
      int i = frame - offset;
      return i >= 0 && size_t(i) < values.size() ? values[i] : 0.0;
    }
  };
  std::copy(input, input + frames, output);
  std::vector<int> boundaries(frames);
  int count = GetBoundaryList(input, frames, boundaries.data()) / 2;
  std::vector<Section> sections;
  sections.reserve(count);
  for (int i = 0; i < count; ++i) {
    int start = boundaries[2*i], end = boundaries[2*i+1];
    int offset = std::max(0, start - 101);
    int size = std::min(frames, end + 102) - offset;
    Section section{offset, start, end, std::vector<double>(size)};
    std::copy(input + start, input + end + 1, section.values.begin() + start - offset);
    double* values = section.values.data();
    section.end = offset + extend_contour(values, end - offset,
        std::min(size - 2, end - offset + 100), 1, candidates, offset);
    section.start = offset + extend_contour(values, start - offset,
        std::max(1, start - offset - 100), -1, candidates, offset);
    sections.push_back(std::move(section));
  }
  int selected = 0;
  // Preserve WORLD's carried mean, selection order, and subsequent swaps.
  double mean = 0;
  for (int i = 0; i < count; ++i) {
    auto& section = sections[i];
    for (int j = section.start; j < section.end; ++j) mean += section.at(j);
    mean /= section.end - section.start;
    if (2200.0 / mean < section.end - section.start)
      std::swap(sections[selected++], sections[i]);
  }
  if (!selected) return;
  for (int i = 0; i < selected; ++i) {
    boundaries[2*i] = sections[i].start;
    boundaries[2*i+1] = sections[i].end;
  }
  std::vector<int> order(selected);
  MakeSortedOrder(boundaries.data(), selected, order.data());
  for (int i = 0; i < frames; ++i) output[i] = sections[0].at(i);
  // WORLD mutates boundary pair 0 while merging, even if order later refers
  // to section 0 again. Keep that aliasing and its original starting channel.
  for (int i = 1; i < selected; ++i) {
    int index = order[i];
    const auto& section = sections[index];
    int start = boundaries[2*index], end = boundaries[2*index+1];
    if (start > boundaries[1]) {
      for (int j = start; j <= end; ++j) output[j] = section.at(j);
      boundaries[0] = start;
    } else {
      if (boundaries[0] <= start && boundaries[1] >= end) continue;
      double score1 = 0, score2 = 0;
      for (int j = start; j <= boundaries[1]; ++j) {
        int begin = candidates.offsets[j], count = candidates.offsets[j + 1] - begin;
        if (!count) continue;
        const double* f0 = candidates.f0.data() + begin;
        const double* scores = candidates.scores.data() + begin;
        score1 += search_score(output[j], f0, scores, count);
        score2 += search_score(section.at(j), f0, scores, count);
      }
      int from = score1 > score2 ? boundaries[1] : start;
      for (int j = from; j <= end; ++j) output[j] = section.at(j);
    }
    boundaries[1] = end;
  }
}

void fix_contour(const Candidates& candidates, int frames, double* output, const double* base) {
  std::vector<double> first(frames), second(frames);
  std::copy(base, base + frames, first.begin());
  FixStep1(first.data(), frames, 0.008, second.data());
  FixStep2(second.data(), frames, 6, first.data());
  connect_contour(first.data(), frames, candidates, second.data());
  FixStep4(second.data(), frames, 9, output);
}

void smooth_contour(const double* f0, int frames, double* output) {
  const double b[2] = {0.0078202080334971724, 0.015640416066994345};
  const double a[2] = {1.7347257688092754, -0.76600660094326412};
  // WORLD pads by 300 frames to let this stable IIR settle. Apply the same
  // padding to each voiced section, avoiding a full-recording filter pass
  // for every section. Only the already-decayed boundary transient changes.
  constexpr int lag = 300;
  std::vector<double> padded, filtered;
  for (int start = 0; start < frames;) {
    if (f0[start] <= 0) { output[start++] = 0; continue; }
    int end = start + 1;
    while (end < frames && f0[end] > 0) ++end;
    int size = end - start + 2 * lag;
    padded.resize(size);
    filtered.resize(size);
    std::copy(f0 + start, f0 + end, padded.begin() + lag);
    FilteringF0(a, b, padded.data(), size, lag, size - lag - 1, filtered.data());
    std::copy(filtered.begin() + lag, filtered.end() - lag, output + start);
    start = end;
  }
}
}

void initialize_host_cache() {
  auto& cache = host_cache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  cache.initialize();
}

void clear_host_cache() {
  auto& cache = host_cache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  cache.storage.reset();
}

Result reference(const double* x, int length, int fs, double floor, double ceil,
                 double period) {
  Result result;
  int output_length = GetSamplesForHarvest(fs, length, period);
  result.f0.resize(output_length);
  result.times.resize(output_length);
  HarvestOption option{floor, ceil, period};
  Harvest(x, length, fs, &option, result.times.data(), result.f0.data());
  return result;
}

Result analyze(const double* x, int length, int fs, double floor, double ceil,
               double period, GpuWorkspace* workspace, Timings* timings) {
  Result result;
  Clock::time_point mark{};
  if (timings) {
    timings->clear();
    mark = Clock::now();
  }
  auto stage = [&](const char* name) {
    if (!timings) return;
    auto now = Clock::now();
    (*timings)[name] += std::chrono::duration<double, std::milli>(now - mark).count();
    mark = now;
  };
  stage("initialize");
  int output_length = GetSamplesForHarvest(fs, length, period);
  result.f0.resize(output_length);
  result.times.resize(output_length);
  const int bands = 1 + int(log(ceil * 1.1 / (floor * 0.9)) / world::kLog2 * 40);
  std::vector<double> boundaries(bands);
  for (int i = 0; i < bands; ++i) boundaries[i] = floor * 0.9 * pow(2.0, (i + 1) / 40.0);
  const int ratio = std::clamp(matlab_round(fs / 8000.0), 1, 12);
  const int y_length = int(std::ceil(double(length) / ratio));
  const double actual_fs = double(fs) / ratio;
  const int fft_size = GetSuitableFFTSize(y_length + 5 + 2 * int(2 * actual_fs / boundaries[0]));
  const int frames = GetSamplesForHarvest(fs, length, 1.0);
  const int capacity = matlab_round(bands / 10.0) * 7;
  std::vector<double> basic_f0(frames), best(frames);

  if (workspace) {
    // CPU scratch borrows a fixed arena; overflow is freed at the end of this
    // branch. Every waveform/candidate element is overwritten before use.
    HostLease scratch;
    HostArray<double> y(&scratch.arena);
    y.allocate(y_length);
    const bool parallel_decimation = ratio > 1 && length >= 32768;
    if (parallel_decimation) gpu_decimate(*workspace, x, length, ratio, y.data(), y_length);
    else GetWaveformAndSpectrumSub(x, length, y_length, actual_fs, ratio, y.data());
    double mean = 0.0;
    for (int i = 0; i < y_length; ++i) mean += y[i];
    mean /= y_length;
    for (int i = 0; i < y_length; ++i) y[i] -= mean;
    stage("preprocess");
    Candidates candidates(&scratch.arena);
    gpu_analyze(*workspace, y.data(), y_length, fft_size, actual_fs,
                boundaries.data(), bands, frames, floor, ceil, capacity,
                candidates, basic_f0.data(), timings);
    if (timings) mark = Clock::now();
    fix_contour(candidates, frames, best.data(), basic_f0.data());
    stage("contour");
    smooth_contour(best.data(), frames, basic_f0.data());
    stage("smooth");
  } else {
    // Diagnostic CPU pipeline: preserve WORLD's stages for benchmark timing.
    // None of its dense matrices or FFT storage is needed by the GPU path.
    std::vector<double> y(fft_size), spectrum_storage(size_t(fft_size) * 2), times(frames);
    auto spectrum = reinterpret_cast<fft_complex*>(spectrum_storage.data());
    GetWaveformAndSpectrum(x, length, y_length, actual_fs, fft_size, ratio, y.data(), spectrum);
    for (int i = 0; i < frames; ++i) times[i] = i / 1000.0;
    stage("preprocess");
    Matrix candidates(frames, capacity), scores(frames, capacity);
    Matrix raw(bands, frames);
    std::vector<double> filtered(fft_size);
    for (int b = 0; b < bands; ++b) {
      GetFilteredSignal(boundaries[b], fft_size, actual_fs, spectrum, y_length, filtered.data());
      stage("filter");
      ZeroCrossings crossings{};
      GetFourZeroCrossingIntervals(filtered.data(), y_length, actual_fs, &crossings);
      GetF0CandidateContour(&crossings, boundaries[b], floor, ceil,
                           times.data(), frames, raw.rows[b]);
      DestroyZeroCrossings(&crossings);
      stage("events");
    }
    int count = DetectOfficialF0Candidates(raw.data(), bands, frames, capacity, candidates.data());
    OverlapF0Candidates(frames, count, candidates.data());
    count *= 7;
    stage("candidates");
    RefineF0Candidates(y.data(), y_length, actual_fs, times.data(), frames, count,
                       floor, ceil, candidates.data(), scores.data());
    stage("refine");
    RemoveUnreliableCandidates(frames, count, candidates.data(), scores.data());
    stage("prune");
    FixF0Contour(candidates.data(), scores.data(), frames, count, best.data());
    stage("contour");
    SmoothF0Contour(best.data(), frames, basic_f0.data());
    stage("smooth");
  }
  for (int i = 0; i < output_length; ++i) {
    result.times[i] = i * period / 1000.0;
    result.f0[i] = basic_f0[std::min(frames - 1, matlab_round(result.times[i] * 1000.0))];
  }
  stage("resample");
  return result;
}
}
