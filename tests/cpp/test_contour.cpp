// Compare the compact contour adapter directly against the pinned WORLD
// routines, independently of GPU arithmetic and the final smoothing pass.
#include "../../src/engine.cpp"
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

using namespace harvest_cuda;

Candidates compact(const Matrix& f0, const Matrix& scores, int count) {
  Candidates result;
  result.offsets.allocate(f0.rows.size() + 1);
  result.offsets[0] = 0;
  for (size_t row = 0; row < f0.rows.size(); ++row) {
    int nonzero = 0;
    for (int col = 0; col < count; ++col) nonzero += f0.rows[row][col] != 0;
    result.offsets[row + 1] = result.offsets[row] + nonzero;
  }
  result.f0.allocate(result.offsets.back());
  result.scores.allocate(result.offsets.back());
  int out = 0;
  for (size_t row = 0; row < f0.rows.size(); ++row) {
    for (int col = 0; col < count; ++col) {
      if (f0.rows[row][col] == 0) continue;
      result.f0[out] = f0.rows[row][col];
      result.scores[out++] = scores.rows[row][col];
    }
  }
  return result;
}

void check_host_scratch() {
  initialize_host_cache();
  auto* original = host_cache().storage.get();
  {
    HostLease scratch;
    HostArray<double> small(&scratch.arena), large(&scratch.arena), tail(&scratch.arena);
    small.allocate(16);
    size_t remaining = scratch.arena.remaining;
    large.allocate(host_cache_bytes / sizeof(double) + 1);
    if (scratch.arena.remaining != remaining)
      throw std::runtime_error("Oversized scratch consumed the fixed arena");
    tail.allocate(16);
    for (int i = 0; i < 16; ++i) { small[i] = i; tail[i] = -i; }
    std::fill(large.begin(), large.end(), 7.0);
    for (int i = 0; i < 16; ++i)
      if (small[i] != i || tail[i] != -i)
        throw std::runtime_error("Scratch allocations overlap");
    if (host_cache().storage.get() != original)
      throw std::runtime_error("Host cache grew on oversized input");
  }
  clear_host_cache();
  if (host_cache().storage)
    throw std::runtime_error("Host cache was not released");
  {
    HostLease scratch;
    if (scratch.arena.remaining != host_cache_bytes)
      throw std::runtime_error("Host cache was not recreated at its fixed size");
  }
  clear_host_cache();
}

void compare_candidate_searches() {
  std::mt19937 random(719);
  for (int count = 0; count <= 35; ++count) {
    std::vector<double> f0(count), scores(count);
    for (int i = 0; i < count; ++i) {
      f0[i] = (i % 4) * 100.0;
      scores[i] = random() % 11;
    }
    if (count > 3) scores[1] = std::numeric_limits<double>::quiet_NaN();
    if (count > 7) scores[6] = std::numeric_limits<double>::infinity();
    for (double value : {0.0, 100.0, 200.0, 300.0, 450.0})
      if (search_score(value, f0.data(), scores.data(), count) !=
          SearchScore(value, f0.data(), scores.data(), count))
        throw std::runtime_error("Candidate score lookup differs from WORLD");
  }
  // Adjacent representable values exercise both the 18% cutoff and rounded
  // ties between equally close candidates, in both extension directions.
  for (int trial = 0; trial < 2000; ++trial) {
    double reference = 20.0 + (random() % 4000000) / 1000.0;
    double value = reference * (trial % 3 == 0 ? 0.82 : trial % 3 == 1 ? 1.18 : 1.001);
    Matrix f0(3, 8);
    for (int j = 0; j < 8; ++j) {
      f0.rows[0][j] = f0.rows[2][j] = value;
      value = std::nextafter(value, trial % 2 ? 0.0 : 10000.0);
    }
    for (int shift : {-1, 1}) {
      std::shuffle(f0.rows[1 + shift], f0.rows[1 + shift] + 8, random);
      std::vector<double> expected{0.0, reference, 0.0}, actual = expected;
      int end = ExtendF0(expected.data(), 3, 1, 1, shift, f0.data(), 8, 0.18, expected.data());
      int actual_end = extend_contour(actual.data(), 1, 1, shift, compact(f0, f0, 8));
      if (end != actual_end || actual != expected)
        throw std::runtime_error("Rounded candidate selection differs from WORLD");
    }
  }
}

void compare_contour(const std::vector<double>& input, Matrix& f0, Matrix& scores,
                     int candidates, int trial) {
  int frames = int(input.size());
  std::vector<double> expected(frames), actual(frames), base(frames);
  auto packed = compact(f0, scores, candidates);
  FixStep3(input.data(), frames, candidates, f0.data(), 0.18, scores.data(), expected.data());
  connect_contour(input.data(), frames, packed, actual.data());
  if (actual != expected)
    throw std::runtime_error("Contour connection differs from WORLD, trial " + std::to_string(trial));
  FixF0Contour(f0.data(), scores.data(), frames, candidates, expected.data());
  SearchF0Base(f0.data(), scores.data(), frames, candidates, base.data());
  fix_contour(packed, frames, actual.data(), base.data());
  if (actual != expected)
    throw std::runtime_error("Contour pipeline differs from WORLD, trial " + std::to_string(trial));
}

int main() {
  check_host_scratch();
  compare_candidate_searches();
  std::mt19937 random(20261110);
  constexpr int candidates = 8;
  const double pitches[candidates] = {80, 90, 120, 150, 180, 250, 400, 600};
  for (int trial = 0; trial < 1000; ++trial) {
    int frames = 2 + int(random() % 2000);
    Matrix f0(frames, candidates), scores(frames, candidates);
    // Candidate tracks contain both gaps that can be crossed and gaps that
    // stop extension. Other trials allow extension to the full 101-frame limit.
    for (int col = 0; col < candidates; ++col) {
      int period = 10 + int(random() % 150), gap = int(random() % 8);
      int phase = int(random() % period);
      for (int row = 0; row < frames; ++row) {
        bool voiced = trial % 4 == 0 || (row + phase) % period >= gap;
        f0.rows[row][col] = voiced ? pitches[trial % 3 == 0 ? col % 4 : col] : 0;
        scores.rows[row][col] = voiced ? 1 + random() % 10 : 0;
      }
    }
    std::vector<double> input(frames);
    for (int start = 1; start < frames - 1;) {
      int end = std::min(frames - 2, start + 6 + int(random() % 80));
      double pitch = pitches[random() % candidates];
      for (int row = start; row <= end; ++row) input[row] = pitch;
      start = end + 2 + int(random() % 100);
    }
    if (trial % 10 == 0) std::fill(input.begin(), input.end(), 0);
    compare_contour(input, f0, scores, candidates, trial);
  }
  // No candidate or voiced section, including the shortest supported contour.
  for (int frames : {1, 2, 1000}) {
    Matrix f0(frames, 0), scores(frames, 0);
    compare_contour(std::vector<double>(frames), f0, scores, 0, -frames);
  }
}
