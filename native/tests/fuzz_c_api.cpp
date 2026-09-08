#include "chronovec.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  constexpr size_t dimensions = 8;
  if (size < dimensions * sizeof(float) + 1)
    return 0;

  const int metric = (data[0] & 1) ? CV_COSINE : CV_L2;
  cv_index *index = cv_create(dimensions, metric, 8, 4);
  if (!index)
    return 0;

  size_t position = 1;
  int64_t next_id = 0;
  while (position + dimensions * sizeof(float) <= size) {
    float vector[dimensions];
    std::memcpy(vector, data + position, sizeof(vector));
    position += sizeof(vector);
    const uint8_t operation = position < size ? data[position++] % 4 : 0;
    uint64_t committed = 0;
    if (operation == 0) {
      cv_insert(index, next_id++, vector, 0, &committed);
    } else if (operation == 1 && next_id > 0) {
      cv_delete(index, int64_t(data[position % size]) % next_id, 0, &committed);
    } else if (operation == 2) {
      int64_t ids[4];
      float distances[4];
      cv_search(index, vector, 4, 0, 4, ids, distances);
    } else {
      cv_vacuum(index, cv_clock(index) + 1, 8);
    }
  }
  cv_destroy(index);
  return 0;
}
