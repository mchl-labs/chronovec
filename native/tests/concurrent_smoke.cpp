#include "chronovec.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

int main() {
  constexpr size_t dimensions = 16;
  constexpr int initial = 256;
  cv_index *index = cv_create(dimensions, CV_COSINE, 32, 8);
  if (!index)
    return 1;
  std::vector<float> vectors((initial + 256) * dimensions);
  for (size_t row = 0; row < initial + 256; ++row)
    for (size_t column = 0; column < dimensions; ++column)
      vectors[row * dimensions + column] =
          float(((row + 3) * (column + 5)) % 29 + 1);
  for (int64_t id = 0; id < initial; ++id)
    if (cv_insert(index, id, vectors.data() + id * dimensions, 0, nullptr))
      return 2;

  std::atomic<bool> failed{false};
  auto reader = [&](int offset) {
    int64_t ids[10];
    float distances[10];
    for (int iteration = 0; iteration < 500; ++iteration) {
      const float *query =
          vectors.data() + ((iteration + offset) % initial) * dimensions;
      if (cv_search(index, query, 10, 0, 8, ids, distances) != 10)
        failed.store(true, std::memory_order_relaxed);
    }
  };
  auto writer = [&] {
    for (int64_t offset = 0; offset < 256; ++offset) {
      const int64_t id = offset % initial;
      if (cv_delete(index, id, 0, nullptr) ||
          cv_insert(index, id, vectors.data() + (initial + offset) * dimensions,
                    0, nullptr))
        failed.store(true, std::memory_order_relaxed);
      if ((offset & 31) == 31)
        cv_vacuum(index, cv_clock(index) + 1, 32);
    }
  };

  std::vector<std::thread> threads;
  for (int reader_id = 0; reader_id < 4; ++reader_id)
    threads.emplace_back(reader, reader_id * 17);
  threads.emplace_back(writer);
  for (auto &thread : threads)
    thread.join();
  cv_vacuum(index, cv_clock(index) + 1, 0);
  cv_destroy(index);
  return failed.load(std::memory_order_relaxed) ? 3 : 0;
}
