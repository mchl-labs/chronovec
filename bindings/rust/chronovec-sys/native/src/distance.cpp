#include "distance.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>

// Horizontal sum of 8 int32 lanes in a __m256i.
static inline int32_t hsum_i32_avx2(__m256i v) {
  __m128i lo = _mm256_extracti128_si256(v, 0);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s  = _mm_add_epi32(lo, hi);
  s = _mm_hadd_epi32(s, s);
  s = _mm_hadd_epi32(s, s);
  return _mm_cvtsi128_si32(s);
}

// Horizontal sum of 8 float lanes in a __m256.
static inline float hsum_f32_avx2(__m256 v) {
  __m128 lo = _mm256_extractf128_ps(v, 0);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s  = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}
#endif

namespace chronovec_internal {

float dot(const float *a, const float *b, std::size_t dimensions) {
#if defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 8 <= dimensions; index += 8)
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + index),
                          _mm256_loadu_ps(b + index), sum);
  float result = hsum_f32_avx2(sum);
  for (; index < dimensions; ++index)
    result += a[index] * b[index];
  return result;
#elif defined(__aarch64__) || defined(_M_ARM64)
  float32x4_t sum = vdupq_n_f32(0);
  std::size_t index = 0;
  for (; index + 4 <= dimensions; index += 4)
    sum = vmlaq_f32(sum, vld1q_f32(a + index), vld1q_f32(b + index));
  float result = vaddvq_f32(sum);
  for (; index < dimensions; ++index)
    result += a[index] * b[index];
  return result;
#else
  float result = 0;
  for (std::size_t index = 0; index < dimensions; ++index)
    result += a[index] * b[index];
  return result;
#endif
}

float l2sq(const float *a, const float *b, std::size_t dimensions) {
#if defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 8 <= dimensions; index += 8) {
    __m256 delta = _mm256_sub_ps(_mm256_loadu_ps(a + index),
                                 _mm256_loadu_ps(b + index));
    sum = _mm256_fmadd_ps(delta, delta, sum);
  }
  float result = hsum_f32_avx2(sum);
  for (; index < dimensions; ++index) {
    float delta = a[index] - b[index];
    result += delta * delta;
  }
  return result;
#elif defined(__aarch64__) || defined(_M_ARM64)
  float32x4_t sum = vdupq_n_f32(0);
  std::size_t index = 0;
  for (; index + 4 <= dimensions; index += 4) {
    float32x4_t delta = vsubq_f32(vld1q_f32(a + index), vld1q_f32(b + index));
    sum = vmlaq_f32(sum, delta, delta);
  }
  float result = vaddvq_f32(sum);
  for (; index < dimensions; ++index) {
    float delta = a[index] - b[index];
    result += delta * delta;
  }
  return result;
#else
  float result = 0;
  for (std::size_t index = 0; index < dimensions; ++index) {
    float delta = a[index] - b[index];
    result += delta * delta;
  }
  return result;
#endif
}

std::int32_t int8_dot(const std::int8_t *a, const std::int8_t *b,
                      std::size_t dimensions) {
#if defined(__AVX2__)
  // Expand int8 → int16, multiply pairs and accumulate into int32.
  // _mm256_madd_epi16 adds adjacent int16 products into int32 lanes.
  __m256i acc = _mm256_setzero_si256();
  std::size_t index = 0;
  for (; index + 32 <= dimensions; index += 32) {
    __m256i a16 = _mm256_cvtepi8_epi16(
        _mm_loadu_si128((const __m128i *)(a + index)));
    __m256i b16 = _mm256_cvtepi8_epi16(
        _mm_loadu_si128((const __m128i *)(b + index)));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, b16));
  }
  std::int32_t result = hsum_i32_avx2(acc);
  for (; index < dimensions; ++index)
    result += std::int32_t(a[index]) * std::int32_t(b[index]);
  return result;
#elif defined(__aarch64__) || defined(_M_ARM64)
  int32x4_t sum = vdupq_n_s32(0);
  std::size_t index = 0;
  for (; index + 16 <= dimensions; index += 16) {
    int8x16_t left = vld1q_s8(a + index);
    int8x16_t right = vld1q_s8(b + index);
    int16x8_t low = vmull_s8(vget_low_s8(left), vget_low_s8(right));
    int16x8_t high = vmull_s8(vget_high_s8(left), vget_high_s8(right));
    sum = vaddq_s32(sum, vpaddlq_s16(low));
    sum = vaddq_s32(sum, vpaddlq_s16(high));
  }
  // Embedding widths such as GloVe-25 leave an eight-element tail after the
  // 16-wide loop.  Falling back to scalar multiply-adds there put nine scalar
  // operations in the innermost screening loop for every candidate.  Keep
  // the same exact integer arithmetic, but consume that tail with NEON too.
  for (; index + 8 <= dimensions; index += 8) {
    int8x8_t left = vld1_s8(a + index);
    int8x8_t right = vld1_s8(b + index);
    sum = vaddq_s32(sum, vpaddlq_s16(vmull_s8(left, right)));
  }
  std::int32_t result = vaddvq_s32(sum);
  for (; index < dimensions; ++index)
    result += std::int32_t(a[index]) * std::int32_t(b[index]);
  return result;
#else
  std::int32_t result = 0;
  for (std::size_t index = 0; index < dimensions; ++index)
    result += std::int32_t(a[index]) * std::int32_t(b[index]);
  return result;
#endif
}

std::int32_t int4_dot(const std::int8_t *query, const std::uint8_t *packed,
                      std::size_t dimensions) {
  std::size_t index = 0;
  std::int32_t result = 0;
#if defined(__AVX2__)
  // Process 32 dimensions per iteration = 16 packed bytes.
  // Packed nibble format: byte[i] = dim[2i] (low 4 bits) | dim[2i+1] (high 4 bits).
  // Each nibble represents a value in [0,15]; subtract 8 to get signed [-8,7].
  //
  // Strategy: extract low and high nibbles into two int8 vectors (lo, hi),
  // then extract even-indexed and odd-indexed query bytes (q_even, q_odd),
  // and accumulate dot(q_even, lo) + dot(q_odd, hi).
  const __m128i mask4  = _mm_set1_epi8(0x0F);
  const __m128i bias8  = _mm_set1_epi8(8);
  // Shuffle masks to extract alternating bytes from a 16-byte register.
  const __m128i shuf_even = _mm_setr_epi8(0,2,4,6,8,10,12,14,
                                           -1,-1,-1,-1,-1,-1,-1,-1);
  const __m128i shuf_odd  = _mm_setr_epi8(1,3,5,7,9,11,13,15,
                                           -1,-1,-1,-1,-1,-1,-1,-1);
  __m256i acc = _mm256_setzero_si256();
  for (; index + 32 <= dimensions; index += 32) {
    // 16 packed bytes → low nibbles (even dims) and high nibbles (odd dims).
    __m128i raw  = _mm_loadu_si128((const __m128i *)(packed + (index >> 1)));
    __m128i lo   = _mm_sub_epi8(_mm_and_si128(raw, mask4), bias8);
    __m128i hi   = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw, 4), mask4), bias8);

    // Deinterleave 32 query bytes into even-index and odd-index halves.
    __m128i q0   = _mm_loadu_si128((const __m128i *)(query + index));
    __m128i q1   = _mm_loadu_si128((const __m128i *)(query + index + 16));
    // q_even: q[0],q[2],...,q[14] followed by q[16],q[18],...,q[30]
    __m128i qe   = _mm_unpacklo_epi64(_mm_shuffle_epi8(q0, shuf_even),
                                      _mm_shuffle_epi8(q1, shuf_even));
    // q_odd:  q[1],q[3],...,q[15] followed by q[17],q[19],...,q[31]
    __m128i qo   = _mm_unpacklo_epi64(_mm_shuffle_epi8(q0, shuf_odd),
                                      _mm_shuffle_epi8(q1, shuf_odd));

    // Expand int8 → int16 and accumulate pairs into int32 via madd_epi16.
    __m256i qe16 = _mm256_cvtepi8_epi16(qe);
    __m256i lo16 = _mm256_cvtepi8_epi16(lo);
    __m256i qo16 = _mm256_cvtepi8_epi16(qo);
    __m256i hi16 = _mm256_cvtepi8_epi16(hi);
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(qe16, lo16));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(qo16, hi16));
  }
  result = hsum_i32_avx2(acc);
#elif defined(__aarch64__) || defined(_M_ARM64)
  const uint8x16_t low_mask = vdupq_n_u8(0x0F);
  const int8x16_t bias = vdupq_n_s8(8);
  int32x4_t sum = vdupq_n_s32(0);
  // 32 dimensions per iteration = 16 packed bytes.
  for (; index + 32 <= dimensions; index += 32) {
    uint8x16_t raw = vld1q_u8(packed + (index >> 1));
    int8x16_t even = vsubq_s8(
        vreinterpretq_s8_u8(vandq_u8(raw, low_mask)), bias);
    int8x16_t odd =
        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), bias);
    // vld2q deinterleaves the query so lane i holds dimensions 2i and 2i+1,
    // matching the low/high nibble split.
    int8x16x2_t q = vld2q_s8(query + index);
    int16x8_t acc = vmull_s8(vget_low_s8(q.val[0]), vget_low_s8(even));
    acc = vmlal_s8(acc, vget_low_s8(q.val[1]), vget_low_s8(odd));
    sum = vaddq_s32(sum, vpaddlq_s16(acc));
    acc = vmull_s8(vget_high_s8(q.val[0]), vget_high_s8(even));
    acc = vmlal_s8(acc, vget_high_s8(q.val[1]), vget_high_s8(odd));
    sum = vaddq_s32(sum, vpaddlq_s16(acc));
  }
  result = vaddvq_s32(sum);
#endif
  for (; index < dimensions; ++index) {
    const std::uint8_t byte = packed[index >> 1];
    const int nibble = (index & 1) ? (byte >> 4) : (byte & 0x0F);
    result += std::int32_t(query[index]) * (std::int32_t(nibble) - 8);
  }
  return result;
}

} // namespace chronovec_internal
