#include "chronovec.h"

#include <stdint.h>

int main(void) {
  cv_index *index = cv_create(4, CV_COSINE, 8, 2);
  if (!index)
    return 1;
  const float vector[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  uint64_t committed = 0;
  if (cv_insert(index, 42, vector, 0, &committed) != 0 || committed == 0)
    return 2;
  int64_t id = -1;
  float distance = -1.0f;
  if (cv_search(index, vector, 1, 0, 2, &id, &distance) != 1)
    return 3;
  if (id != 42 || distance < -0.00001f || distance > 0.00001f)
    return 4;
  const int64_t deleted[1] = {42};
  const int64_t upserted[1] = {43};
  if (cv_apply_changes(index, deleted, 1, upserted, vector, 1, &committed) != 0)
    return 5;
  if (cv_search(index, vector, 1, 0, 2, &id, &distance) != 1 || id != 43)
    return 6;
  cv_destroy(index);
  return 0;
}
