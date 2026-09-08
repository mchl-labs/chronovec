#ifndef CHRONOVEC_DISTANCE_H
#define CHRONOVEC_DISTANCE_H

#include <cstddef>
#include <cstdint>

namespace chronovec_internal {

float dot(const float *a, const float *b, std::size_t dimensions);
float l2sq(const float *a, const float *b, std::size_t dimensions);
std::int32_t int8_dot(const std::int8_t *a, const std::int8_t *b,
                      std::size_t dimensions);
// Dot product of an int8 query against 4-bit packed data codes. Two dimensions
// share a byte: dimension 2k in the low nibble, 2k+1 in the high nibble, each
// stored offset by +8 so the signed range is [-7, 7].
std::int32_t int4_dot(const std::int8_t *query, const std::uint8_t *packed,
                      std::size_t dimensions);
constexpr std::int32_t kInt4Scale = 7;

} // namespace chronovec_internal

#endif
