// Readers searching while batched writers run with parallel apply lanes.
//
// The existing concurrent smoke test predates all of this: it covers single
// inserts against concurrent readers. What is new and unproven under a race
// detector is the batch path -- lanes writing to different pages at once, the
// two clocks that make a batch atomic, and consolidation republishing the
// directory underneath a reader.
//
// Built by CTest and run under ThreadSanitizer in CI; a race here is silent
// corruption in the path most users will actually take.
#include "chronovec.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

int main() {
  constexpr size_t dimensions = 24;
  constexpr int64_t live = 4096;
  constexpr int64_t batch = 512;
  cv_index *index = cv_create_with_options(dimensions, CV_L2, 32, 16,
                                           CV_ENABLE_INT8_SCREENING, 8);
  if (!index)
    return 1;
  cv_set_threads(index, 4);

  const size_t rows = size_t(live) * 4;
  std::vector<float> vectors(rows * dimensions);
  for (size_t row = 0; row < rows; ++row)
    for (size_t column = 0; column < dimensions; ++column)
      vectors[row * dimensions + column] =
          float(((row * 7 + column * 13) % 97) + 1);

  std::vector<int64_t> ids(live);
  for (int64_t position = 0; position < live; ++position)
    ids[size_t(position)] = position;
  if (cv_insert_batch(index, ids.data(), vectors.data(), size_t(live),
                      nullptr) != size_t(live))
    return 2;

  std::atomic<bool> failed{false};
  std::atomic<bool> stop{false};

  // Readers pin a snapshot and check it does not move under them. A batch that
  // is not atomic shows up here as a count that changes while the snapshot is
  // held.
  auto reader = [&](int offset) {
    int64_t found[16];
    float distances[16];
    while (!stop.load(std::memory_order_relaxed)) {
      const uint64_t snapshot = cv_clock(index);
      if (!snapshot)
        continue;
      const float *query =
          vectors.data() + size_t((offset * 31) % live) * dimensions;
      const size_t first =
          cv_search(index, query, 16, snapshot, 16, found, distances);
      const size_t again =
          cv_search(index, query, 16, snapshot, 16, found, distances);
      if (first != again)
        failed.store(true, std::memory_order_relaxed);
      ++offset;
    }
  };

  auto writer = [&] {
    int64_t next = live;
    for (int round = 0; round < 12; ++round) {
      // Braced, not parenthesised: `vector<int64_t> fresh(size_t(batch))`
      // parses as a function declaration.
      std::vector<int64_t> fresh(static_cast<std::size_t>(batch), 0);
      for (int64_t position = 0; position < batch; ++position)
        fresh[size_t(position)] = next + position;
      if (cv_insert_batch(index, fresh.data(),
                          vectors.data() + size_t(next % int64_t(rows - batch)) *
                                               dimensions,
                          size_t(batch), nullptr) != size_t(batch))
        failed.store(true, std::memory_order_relaxed);
      if (cv_delete_batch(index, fresh.data(), size_t(batch), nullptr) !=
          size_t(batch))
        failed.store(true, std::memory_order_relaxed);
      next += batch;
      cv_vacuum(index, cv_clock(index) + 1, 0);
    }
  };

  std::vector<std::thread> threads;
  for (int reader_id = 0; reader_id < 4; ++reader_id)
    threads.emplace_back(reader, reader_id * 17);
  std::thread scribe(writer);
  scribe.join();
  stop.store(true, std::memory_order_relaxed);
  for (auto &thread : threads)
    thread.join();

  cv_vacuum(index, cv_clock(index) + 1, 0);
  cv_stats stats{};
  if (cv_get_stats(index, &stats) != 0)
    return 4;
  if (stats.live_vectors != uint64_t(live))
    failed.store(true, std::memory_order_relaxed);
  cv_destroy(index);
  return failed.load(std::memory_order_relaxed) ? 3 : 0;
}
