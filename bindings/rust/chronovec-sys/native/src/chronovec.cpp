#include "chronovec.h"
#include "distance.h"
#if defined(CHRONOVEC_BLAS_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(CHRONOVEC_BLAS)
#include <cblas.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#if defined(_WIN32)
// Must precede <windows.h>: without it, the header's max/min macros wreck
// every std::numeric_limits<T>::max()/min() call below.
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
namespace {
using chronovec_internal::dot;
using chronovec_internal::int4_dot;
using chronovec_internal::int8_dot;
using chronovec_internal::kInt4Scale;
using chronovec_internal::l2sq;

constexpr uint64_t MAX_TS = std::numeric_limits<uint64_t>::max();
// Flat centroid routing is cheaper through the medium-sized regime.  The
// graph's heap, visited bitmap, and pointer-chasing outweigh the contiguous
// centroid scan at 2,470, 7,109 (1M SIFT), and 8,509 (1.18M GloVe) pages:
// it scored fewer centroids but took 2.65x, 3.23x, and 1.86x as long,
// respectively. At 4,096 pages the flat path switches to the 4-bit scan, so
// it also avoids paying float-centroid bandwidth for this larger range. Keep
// graph routing for directories that genuinely exceed the cache-friendly scan
// regime.
constexpr size_t ROUTING_GRAPH_THRESHOLD = 16384;
thread_local std::string last_error;
thread_local cv_search_metrics last_search_metrics{};

// Optional file-backed storage for the authoritative float vectors. BETA.
//
// Codes and routing stay resident; only the float payload moves to a mapping,
// and a query touches it for roughly `rerank_factor * k` records. The whole file
// is mapped once at a fixed size, so block pointers never move and no remap can
// invalidate a reader mid-search. Blocks are page-sized and uniform, so reuse is
// a free list rather than an allocator.
//
// Known limit, measured rather than assumed: copy-on-write rewrites a whole page
// to change one slot, dirtying ~128 KB per insert at capacity 256 and d=128. At
// several thousand inserts a second that is hundreds of MB/s of writeback, more
// than a commodity SSD absorbs. Suitable for read-heavy and bulk-load-then-serve
// workloads; sustained high-rate mutation needs the immutable-base-plus-delta
// write path before this mode is appropriate.
struct VectorArena {
  int descriptor = -1;
  void *base = nullptr;
  size_t block_bytes = 0, blocks = 0, mapped_bytes = 0;
  std::vector<size_t> free_blocks;
  std::mutex guard;
  std::filesystem::path path;

  VectorArena(const std::filesystem::path &file, size_t block, size_t count)
      : block_bytes(block), blocks(count), path(file) {
    mapped_bytes = block_bytes * blocks;
#if defined(_WIN32)
    throw std::runtime_error("disk-backed vectors are not supported on Windows yet");
#else
    descriptor = ::open(path.string().c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (descriptor < 0)
      throw std::runtime_error("cannot create vector arena file");
    if (::ftruncate(descriptor, off_t(mapped_bytes)) != 0) {
      ::close(descriptor);
      throw std::runtime_error("cannot size vector arena file");
    }
    base = ::mmap(nullptr, mapped_bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                  descriptor, 0);
    if (base == MAP_FAILED) {
      ::close(descriptor);
      base = nullptr;
      throw std::runtime_error("cannot map vector arena file");
    }
    free_blocks.reserve(blocks);
    for (size_t i = blocks; i-- > 0;)
      free_blocks.push_back(i);
#endif
  }
  ~VectorArena() {
#if !defined(_WIN32)
    if (base)
      ::munmap(base, mapped_bytes);
    if (descriptor >= 0)
      ::close(descriptor);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
#endif
  }
  // Destructors may run on reader threads when the last reference to a retired
  // page drops, so allocation and release are both guarded.
  void *acquire_bytes() {
    std::lock_guard lock(guard);
    if (free_blocks.empty())
      throw std::runtime_error(
          "arena is full; raise max_vectors at index creation");
    size_t block = free_blocks.back();
    free_blocks.pop_back();
    return static_cast<char *>(base) + block * block_bytes;
  }
  void release_bytes(void *pointer) {
    if (!pointer)
      return;
    std::lock_guard lock(guard);
    size_t block = size_t(static_cast<char *>(pointer) -
                          static_cast<char *>(base)) / block_bytes;
    free_blocks.push_back(block);
  }
  float *acquire() {
    std::lock_guard lock(guard);
    if (free_blocks.empty())
      throw std::runtime_error(
          "vector arena is full; raise max_vectors at index creation");
    size_t block = free_blocks.back();
    free_blocks.pop_back();
    return reinterpret_cast<float *>(static_cast<char *>(base) +
                                     block * block_bytes);
  }
  void release(float *pointer) {
    if (!pointer)
      return;
    std::lock_guard lock(guard);
    size_t block = size_t(reinterpret_cast<char *>(pointer) -
                          static_cast<char *>(base)) / block_bytes;
    free_blocks.push_back(block);
  }
  size_t resident_blocks() const { return blocks - free_blocks.size(); }
  // Flush the mapping so its pages become clean, and hint that they need not
  // stay resident.
  //
  // What this does and does not do, measured rather than assumed. Freshly
  // written mapped pages are dirty, and a dirty page cannot be evicted, so
  // without a flush the mode is inert -- resident memory is no better than the
  // heap. After msync the data is on disk (file blocks go from 188 MB to 220 MB
  // in the 120k-vector test) and the pages are evictable.
  //
  // Eviction itself is the kernel's decision. On macOS MADV_DONTNEED does not
  // force MAP_SHARED file pages out, so resident memory does not drop until
  // there is real memory pressure -- which is the correct behaviour, since
  // evicting early would only cost faults later. The return value is therefore
  // bytes made evictable, not bytes reclaimed.
  size_t make_evictable() {
#if defined(_WIN32)
    return 0;
#else
    if (!base)
      return 0;
    if (::msync(base, mapped_bytes, MS_SYNC) != 0)
      return 0;
    // MADV_DONTNEED discards the resident copy; a later read faults it back
    // from the file. Safe only because the flush above already persisted it.
    // Advisory: honoured on Linux, a no-op for these mappings on macOS.
    ::madvise(base, mapped_bytes, MADV_DONTNEED);
    return mapped_bytes;
#endif
  }
};
// The float payload of a page, shared between a page and its copy-on-write
// clones.
//
// Copying it was 5.34us of a 33.8us insert -- 128 KB memcpy at capacity 256 and
// d=128 -- to change one slot. Sharing is sound because an insert only ever
// writes a slot that no published version counts as occupied, and no published
// version reads such a slot: visibility comes from `filled` and the MVCC
// interval, both of which stay per-version. All mutation holds the writer mutex,
// so two clones can never be writing the same free slot.
//
// Split and merge build fresh pages with different slot layouts and so get
// their own store. Vacuum clears occupancy without touching the floats.
struct VectorArena;
// The screening codes, optionally on the same mapped file as the vectors.
//
// Vectors are 80% of the payload and screening the other 20%, so mapping only
// the vectors left a fifth of a large index pinned in memory and capped how
// far past RAM it could go. Same lifetime rules as VectorStore: shared between
// a page and its copy-on-write clones, released when the last one drops.
struct RecordStore {
  std::vector<uint8_t> heap;
  uint8_t *block = nullptr;
  std::shared_ptr<VectorArena> arena;
  RecordStore(size_t bytes, std::shared_ptr<VectorArena> backing);
  ~RecordStore();
  RecordStore(const RecordStore &other);
  RecordStore &operator=(const RecordStore &) = delete;
  size_t bytes = 0;
  uint8_t *data() { return block ? block : heap.data(); }
  const uint8_t *data() const { return block ? block : heap.data(); }
  bool empty() const { return bytes == 0; }
};
struct VectorStore {
  std::vector<float> heap;
  float *block = nullptr;
  std::shared_ptr<VectorArena> arena;
  VectorStore(size_t floats, std::shared_ptr<VectorArena> backing);
  ~VectorStore();
  float *data() { return block ? block : heap.data(); }
  const float *data() const { return block ? block : heap.data(); }
};

struct Page {
  size_t capacity, dimensions;
  bool track_radius;
  std::vector<int64_t> ids;
  // Shared with this page's copy-on-write clones; see VectorStore.
  std::shared_ptr<VectorStore> store;
  // Hot screening fields are interleaved per record: [norm f32][rcp f32][code
  // bytes]. Scanning them as parallel arrays measured 1.17x slower on an
  // isolated 30k-candidate loop; interleaving only these three captures the
  // whole gain, so the MVCC arrays stay separate (they are small and stay
  // resident per page).
  // Shared with this page's copy-on-write clones, exactly as the float payload
  // is. Cloning a page to append one row was copying this whole array: at
  // d=768 that is 256 slots x 392 bytes = 100KB per inserted row, and it was
  // 49% of an insert at that width.
  //
  // Sharing is safe for the same reason sharing the vectors is. A slot is only
  // reused after vacuum, and vacuum only releases versions whose end timestamp
  // precedes the oldest live snapshot, so no reader that could still consult a
  // slot can have it taken away. Writes that rewrite *every* slot -- rebinding
  // the residual codes -- break that, and take a private copy first.
  std::shared_ptr<RecordStore> records;
  size_t record_stride = 0;
  void ensure_private_records() {
    if (records && records.use_count() > 1)
      records = std::make_shared<RecordStore>(*records);
  }
  float *record_scalars(size_t s) {
    return reinterpret_cast<float *>(records->data() + s * record_stride);
  }
  const float *record_scalars(size_t s) const {
    return reinterpret_cast<const float *>(records->data() + s * record_stride);
  }
  uint8_t *record_codes(size_t s) {
    return records->data() + s * record_stride + 2 * sizeof(float);
  }
  const uint8_t *record_codes(size_t s) const {
    return records->data() + s * record_stride + 2 * sizeof(float);
  }
  // Residual scalar quantisation. codes[] holds the unit residual of each
  // vector against code_centroid, scaled to int8; code_norm[] holds that
  // residual's length. Queries reconstruct
  //   ||q-v||^2 = ||q-c||^2 + ||v-c||^2 - 2||q-c||*||v-c||*<u_q,u_v>
  // which is metric-agnostic, so screening now applies to L2 as well as
  // cosine. code_centroid is deliberately pinned at bind time rather than
  // tracking the live centroid, so codes and queries always agree; structural
  // events (split, merge, load) rebind it.

  // Reciprocal of the per-vector correction factor g = <c, u>, where c is the
  // integer code vector and u the unit residual. Estimating <u_q,u> as
  // <c,u_q>/<c,u> debiases the projection that quantisation shrinks; dropping
  // this term costs real recall at 4 bits. Reciprocal is stored so the
  // candidate loop multiplies instead of divides.

  std::vector<float> code_centroid;
  bool codes_bound = false;
  // Incremental centroid updates since the last exact re-sum.
  uint32_t incremental_since_refresh = 0;
  // Residual coding pays for itself only when a compressed candidate is much
  // cheaper than an exact one. Below ~64 dimensions an exact float distance is
  // already so cheap that the per-page query-residual setup costs more than it
  // saves, so low-dimensional cosine keeps the original global int8 sketch.
  bool residual = true;
  float code_scale = 0.0f;
  // Residual codes are 4-bit packed (two dimensions per byte); the global
  // sketch stays int8. Simulation showed 3 bits lossless against an fp32
  // control at a rerank budget of 50, so 4 bits carries margin.
  size_t code_stride = 0;
  std::vector<uint64_t> begin, end;
  std::vector<uint8_t> occupied;
  std::vector<uint32_t> filled;
  // Attribute labels, allocated only when the index enables them. label_union
  // is the OR of every live label, so a page that cannot satisfy a filter is
  // skipped without touching a single record. That page-level pruning is the
  // structural reason a partition index filters well where a graph degrades.
  std::vector<uint64_t> labels;
  uint64_t label_union = 0;
  std::vector<double> sum;
  std::vector<float> centroid;
  float radius = 0.0f;
  size_t current_count = 0;
  Page(size_t c, size_t d, bool sketches, bool bounds = false,
       bool residual_codes = true, bool with_labels = false,
       std::shared_ptr<VectorArena> with_arena = nullptr,
       std::shared_ptr<VectorArena> with_record_arena = nullptr)
      : capacity(c), dimensions(d), track_radius(bounds), ids(c, -1),
        records(std::make_shared<RecordStore>(
            sketches ? c * (2 * sizeof(float) +
                            (residual_codes ? (d + 1) / 2 : d))
                     : 0,
            with_record_arena)),
        record_stride(2 * sizeof(float) + (residual_codes ? (d + 1) / 2 : d)),
        code_centroid(sketches ? d : 0),
        residual(residual_codes),
        code_stride(residual_codes ? (d + 1) / 2 : d), begin(c), end(c),
        occupied(c), sum(d), centroid(d) {
    if (with_labels)
      labels.assign(c, 0);
    store = std::make_shared<VectorStore>(c * d, std::move(with_arena));
    filled.reserve(c);
  }
  // Copy-on-write must take its own block: sharing one would let a published
  // page observe a mutation made to its clone.
  // Shares the float payload; only per-version metadata is copied.
  Page(const Page &other)
      : capacity(other.capacity), dimensions(other.dimensions),
        track_radius(other.track_radius), ids(other.ids),
        store(other.store), records(other.records),
        record_stride(other.record_stride), code_centroid(other.code_centroid),
        codes_bound(other.codes_bound),
        incremental_since_refresh(other.incremental_since_refresh),
        residual(other.residual),
        code_scale(other.code_scale), code_stride(other.code_stride),
        begin(other.begin), end(other.end), occupied(other.occupied),
        filled(other.filled), labels(other.labels),
        label_union(other.label_union), sum(other.sum),
        centroid(other.centroid), radius(other.radius),
        current_count(other.current_count) {}
  Page &operator=(const Page &) = delete;
  float *storage() { return store->data(); }
  const float *storage() const { return store->data(); }
  float *vec(size_t s) { return storage() + s * dimensions; }
  const float *vec(size_t s) const { return storage() + s * dimensions; }
  bool visible(size_t s, uint64_t snap) const {
    return occupied[s] && begin[s] <= snap && snap < end[s];
  }
  size_t free_slot() const {
    for (size_t i = 0; i < capacity; ++i)
      if (!occupied[i])
        return i;
    return capacity;
  }
  size_t occupied_count() const { return filled.size(); }
  void encode(size_t s) {
    if (records->empty())
      return;
    const float *v = vec(s);
    if (!residual) {
      // Original global sketch: valid for unit-norm cosine vectors and needs
      // no per-page query setup.
      record_scalars(s)[0] = 1.0f;
      record_scalars(s)[1] = 1.0f;
      auto *raw = reinterpret_cast<int8_t *>(record_codes(s));
      for (size_t j = 0; j < dimensions; ++j)
        raw[j] = int8_t(std::clamp(std::lround(v[j] * 127.0f), -127l, 127l));
      return;
    }
    // The residual is materialised once and reused for both the norm and the
    // quantisation. The buffer is on the stack for realistic dimensions: a
    // thread_local vector here measured slower than recomputing the residual,
    // because every encode paid a TLS guard check.
    constexpr size_t kStackDimensions = 1024;
    float stack_residual[kStackDimensions];
    std::vector<float> heap_residual;
    float *r = stack_residual;
    if (dimensions > kStackDimensions) {
      heap_residual.resize(dimensions);
      r = heap_residual.data();
    }
    float squared = 0.0f;
    for (size_t j = 0; j < dimensions; ++j) {
      r[j] = v[j] - code_centroid[j];
      squared += r[j] * r[j];
    }
    const float length = std::sqrt(squared);
    record_scalars(s)[0] = length;
    float peak = 0.0f;
    for (size_t j = 0; j < dimensions; ++j)
      peak = std::max(peak, std::fabs(r[j]));
    const float scale = peak > 0.0f ? float(kInt4Scale) / peak : 0.0f;
    uint8_t *out = record_codes(s);
    auto nibble = [&](size_t j) {
      return uint8_t(std::clamp(std::lround(r[j] * scale),
                                -long(kInt4Scale), long(kInt4Scale)) +
                     8);
    };
    size_t j = 0;
    for (; j + 1 < dimensions; j += 2)
      out[j >> 1] = uint8_t(nibble(j) | uint8_t(nibble(j + 1) << 4));
    if (dimensions & 1)
      out[j >> 1] = nibble(j);
    // g = <c, u> with c the integer code and u = r/length.
    const float inverse_length = length > 0.0f ? 1.0f / length : 0.0f;
    float g = 0.0f;
    for (size_t index = 0; index < dimensions; ++index) {
      const uint8_t byte = out[index >> 1];
      const int value = ((index & 1) ? (byte >> 4) : (byte & 0x0F)) - 8;
      g += float(value) * r[index] * inverse_length;
    }
    record_scalars(s)[1] = g > 0.0f ? 1.0f / g : 0.0f;
  }
  // Pin code_centroid to the current centroid and re-encode every live slot.
  void bind_codes() {
    if (records->empty())
      return;
    // Rewrites every slot, so it cannot run on a buffer a published version is
    // still reading.
    ensure_private_records();
    if (residual)
      std::copy(centroid.begin(), centroid.end(), code_centroid.begin());
    codes_bound = true;
    double total = 0.0;
    size_t live = 0;
    for (uint32_t slot : filled)
      if (occupied[slot]) {
        encode(slot);
        total += record_scalars(slot)[0];
        ++live;
      }
    code_scale = live ? float(total / double(live)) : 0.0f;
  }
  // Split and merge copy codes rather than re-encoding, so a page's codes can
  // outlive the centroid they were bound to. Rebind only once the live centroid
  // has drifted far enough to blunt the estimates.
  void rebind_if_drifted() {
    if (records->empty() || !residual || !codes_bound || code_scale <= 0.0f)
      return;
    float drift = 0.0f;
    for (size_t j = 0; j < dimensions; ++j) {
      const float delta = centroid[j] - code_centroid[j];
      drift += delta * delta;
    }
    if (std::sqrt(drift) > 0.25f * code_scale)
      bind_codes();
  }
  // Fold one slot into the page's aggregates instead of re-summing every
  // filled slot, which measured 2.08us of a 33.8us insert.
  //
  // This is exact rather than approximate: an insert adds a vector that was not
  // counted, a delete removes one that was, and neither path touches `occupied`
  // so `filled` needs no compaction. An exact re-sum still runs every few
  // capacities to stop floating-point error accumulating over a long-lived
  // page, and immediately whenever the incremental path cannot maintain a
  // value exactly.
  void apply_delta(size_t slot, bool added, int metric) {
    // Adaptive bounds need the page radius, which is a max over all members and
    // cannot be maintained from one slot.
    if (track_radius || ++incremental_since_refresh > 4 * capacity) {
      incremental_since_refresh = 0;
      recompute(metric);
      return;
    }
    const float *v = vec(slot);
    if (added) {
      for (size_t j = 0; j < dimensions; ++j)
        sum[j] += v[j];
      ++current_count;
      if (!labels.empty())
        label_union |= labels[slot];
    } else {
      for (size_t j = 0; j < dimensions; ++j)
        sum[j] -= v[j];
      if (current_count)
        --current_count;
      // label_union is deliberately left alone. Recomputing it would cost a
      // pass over the page, and a stale union can only be too permissive: the
      // page is visited when it need not be, and the per-record check rejects
      // the record. It can never cause a match to be missed.
    }
    if (!current_count) {
      incremental_since_refresh = 0;
      recompute(metric);
      return;
    }
    for (size_t j = 0; j < dimensions; ++j)
      centroid[j] = float(sum[j] / current_count);
    if (metric == CV_COSINE) {
      float n = std::sqrt(dot(centroid.data(), centroid.data(), dimensions));
      if (n > 0)
        for (float &x : centroid)
          x /= n;
    }
    if (!records->empty() && !codes_bound)
      bind_codes();
    else
      rebind_if_drifted();
  }
  void recompute(int metric) {
    incremental_since_refresh = 0;
    filled.erase(std::remove_if(filled.begin(), filled.end(),
                                [&](uint32_t s) { return !occupied[s]; }),
                 filled.end());
    std::fill(sum.begin(), sum.end(), 0.0);
    current_count = 0;
    for (uint32_t s : filled)
      if (visible(s, MAX_TS - 1)) {
        for (size_t j = 0; j < dimensions; ++j)
          sum[j] += vec(s)[j];
        ++current_count;
      }
    if (!current_count) {
      std::fill(centroid.begin(), centroid.end(), 0);
      radius = track_radius && !filled.empty()
                   ? std::numeric_limits<float>::infinity()
                   : 0.0f;
      return;
    }
    for (size_t j = 0; j < dimensions; ++j)
      centroid[j] = float(sum[j] / current_count);
    if (metric == CV_COSINE) {
      float n = std::sqrt(dot(centroid.data(), centroid.data(), dimensions));
      if (n > 0)
        for (float &x : centroid)
          x /= n;
    }
    if (!labels.empty()) {
      label_union = 0;
      for (uint32_t slot : filled)
        if (occupied[slot])
          label_union |= labels[slot];
    }
    if (!records->empty() && !codes_bound)
      bind_codes();
    else
      rebind_if_drifted();
    if (track_radius) {
      radius = 0.0f;
      for (uint32_t slot : filled)
        if (occupied[slot]) {
          float value = metric == CV_COSINE
                            ? 1.0f - dot(centroid.data(), vec(slot), dimensions)
                            : l2sq(centroid.data(), vec(slot), dimensions);
          radius = std::max(radius, std::max(0.0f, value));
        }
    }
  }
};
VectorStore::VectorStore(size_t floats, std::shared_ptr<VectorArena> backing) {
  if (backing) {
    arena = std::move(backing);
    block = arena->acquire();
    std::memset(block, 0, floats * sizeof(float));
  } else {
    heap.assign(floats, 0.0f);
  }
}
VectorStore::~VectorStore() {
  if (arena && block)
    arena->release(block);
}
RecordStore::RecordStore(size_t n, std::shared_ptr<VectorArena> backing)
    : bytes(n) {
  if (!n)
    return;
  if (backing) {
    arena = std::move(backing);
    block = static_cast<uint8_t *>(arena->acquire_bytes());
    std::memset(block, 0, n);
  } else {
    heap.assign(n, 0);
  }
}
// A private copy, taken when a bulk rewrite cannot run on a shared buffer.
RecordStore::RecordStore(const RecordStore &other) : bytes(other.bytes) {
  if (!bytes)
    return;
  if (other.arena) {
    arena = other.arena;
    block = static_cast<uint8_t *>(arena->acquire_bytes());
    std::memcpy(block, other.data(), bytes);
  } else {
    heap.assign(other.heap.begin(), other.heap.end());
  }
}
RecordStore::~RecordStore() {
  if (arena && block)
    arena->release_bytes(block);
}

template <class T> struct AtomicShared {
  mutable std::shared_ptr<T> value;
  AtomicShared(std::shared_ptr<T> v) : value(std::move(v)) {}
  std::shared_ptr<T> load(std::memory_order = std::memory_order_seq_cst) const {
    return std::atomic_load(&value);
  }
  void store(std::shared_ptr<T> v,
             std::memory_order = std::memory_order_seq_cst) {
    std::atomic_store(&value, std::move(v));
  }
  bool compare_exchange_strong(std::shared_ptr<T> &expected,
                               std::shared_ptr<T> desired,
                               std::memory_order = std::memory_order_seq_cst,
                               std::memory_order = std::memory_order_seq_cst) {
    return std::atomic_compare_exchange_strong(&value, &expected,
                                               std::move(desired));
  }
};
struct Descriptor {
  uint64_t id;
  AtomicShared<const Page> image;
  Descriptor(uint64_t i, std::shared_ptr<const Page> p)
      : id(i), image(std::move(p)) {}
};
// out[n x m] = rows[n x d] . cents[m x d]^T
//
// Batched insert is the only caller: assignment is the bulk-ingest bottleneck,
// and it is the one place where enough rows are known at once for a matmul to
// beat a per-row scan. Where a BLAS is available this reaches the platform's
// matrix unit; the fallback is a blocked scalar loop, which is slower but
// keeps a dependency-free build working.
inline void inner_products(const float *rows, size_t n, const float *cents,
                           size_t m, size_t d, float *out) {
#if defined(CHRONOVEC_BLAS)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, int(n), int(m), int(d),
              1.0f, rows, int(d), cents, int(d), 0.0f, out, int(m));
#else
  // Blocked so a strip of centroids stays hot across several rows rather than
  // being re-streamed for each one.
  constexpr size_t kRowBlock = 8;
  std::fill(out, out + n * m, 0.0f);
  for (size_t r0 = 0; r0 < n; r0 += kRowBlock) {
    const size_t rn = std::min(kRowBlock, n - r0);
    for (size_t c = 0; c < m; ++c) {
      const float *cv = cents + c * d;
      for (size_t r = 0; r < rn; ++r)
        out[(r0 + r) * m + c] = dot(rows + (r0 + r) * d, cv, d);
    }
  }
#endif
}

// Write-ahead log.
//
// Without one a crash loses every write since the last checkpoint, which for
// an index whose whole point is a corpus that keeps changing is not a
// footnote. Records are appended before the change is applied, so recovery
// redoes rather than undoes, and the log is the authority for anything the
// checkpoint predates.
//
// Each record is length-prefixed and CRC'd. A crash mid-append leaves a torn
// tail; replay stops at the first record that does not verify and the file is
// truncated there, because a partially written record describes a change that
// never committed.
enum : uint8_t { WAL_INSERT = 1, WAL_DELETE = 2 };

// CRC-32 lookup table (reflected polynomial 0xEDB88320, same as zlib/Ethernet).
// One table lookup per byte instead of eight iterations; same polynomial and
// output as the old bit-by-bit loop, so existing WAL files remain valid.
static constexpr auto kCrc32Table = [] {
  std::array<uint32_t, 256> t{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int bit = 0; bit < 8; ++bit)
      c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    t[i] = c;
  }
  return t;
}();

inline uint32_t crc32_of(const uint8_t *data, size_t n) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i)
    crc = (crc >> 8) ^ kCrc32Table[(crc ^ data[i]) & 0xFFu];
  return ~crc;
}

struct Wal {
  std::string path;
  // A descriptor held open, not an ofstream reopened per flush. Reopening the
  // file to reach fsync cost more than the fsync on the single-insert path.
  int handle = -1;
#if defined(_WIN32)
  std::ofstream out;
#endif
  bool sync_on_commit = true;
  std::vector<uint8_t> pending;   // staged records for the current operation

  explicit Wal(std::string file, bool sync)
      : path(std::move(file)), sync_on_commit(sync) {
#if defined(_WIN32)
    out.open(path, std::ios::binary | std::ios::app);
    if (!out)
      throw std::runtime_error("cannot open write-ahead log: " + path);
#else
    handle = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (handle < 0)
      throw std::runtime_error("cannot open write-ahead log: " + path);
#endif
  }
  ~Wal() {
#if !defined(_WIN32)
    if (handle >= 0)
      ::close(handle);
#endif
  }
  Wal(const Wal &) = delete;
  Wal &operator=(const Wal &) = delete;

  static void put(std::vector<uint8_t> &buffer, const void *data, size_t n) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    buffer.insert(buffer.end(), bytes, bytes + n);
  }

  void stage(uint8_t op, uint64_t ts, int64_t id, const float *vector,
             uint32_t dimensions) {
    std::vector<uint8_t> body;
    put(body, &op, 1);
    put(body, &ts, sizeof ts);
    put(body, &id, sizeof id);
    put(body, &dimensions, sizeof dimensions);
    if (dimensions)
      put(body, vector, size_t(dimensions) * sizeof(float));
    const uint32_t crc = crc32_of(body.data(), body.size());
    const uint32_t length = uint32_t(body.size());
    put(pending, &length, sizeof length);
    pending.insert(pending.end(), body.begin(), body.end());
    put(pending, &crc, sizeof crc);
  }

  // Everything staged becomes durable together, before the change is applied.
  void flush() {
    if (pending.empty())
      return;
#if defined(_WIN32)
    out.write(reinterpret_cast<const char *>(pending.data()),
              std::streamsize(pending.size()));
    out.flush();
#else
    size_t written = 0;
    while (written < pending.size()) {
      const ssize_t n = ::write(handle, pending.data() + written,
                                pending.size() - written);
      if (n <= 0)
        throw std::runtime_error("write-ahead log write failed");
      written += size_t(n);
    }
    // write() reaches the OS; only fsync reaches the disk, and the difference
    // is exactly what a power cut takes.
    if (sync_on_commit)
      ::fsync(handle);
#endif
    pending.clear();
  }

  void discard() { pending.clear(); }
};

struct WalRecord {
  uint8_t op;
  uint64_t ts;
  int64_t id;
  std::vector<float> vector;
};

// Read a log back, stopping at the first record that does not verify.
inline std::vector<WalRecord> replay_wal(const std::string &path,
                                         uint64_t after, size_t *good_bytes) {
  std::vector<WalRecord> out;
  *good_bytes = 0;
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return out;
  std::vector<uint8_t> body;
  while (true) {
    uint32_t length = 0;
    if (!in.read(reinterpret_cast<char *>(&length), sizeof length))
      break;
    if (!length || length > (1u << 28))
      break;
    body.resize(length);
    if (!in.read(reinterpret_cast<char *>(body.data()), length))
      break;
    uint32_t stored = 0;
    if (!in.read(reinterpret_cast<char *>(&stored), sizeof stored))
      break;
    if (crc32_of(body.data(), body.size()) != stored)
      break;      // torn tail: this change never committed
    size_t at = 0;
    WalRecord record{};
    record.op = body[at];
    at += 1;
    std::memcpy(&record.ts, body.data() + at, sizeof record.ts);
    at += sizeof record.ts;
    std::memcpy(&record.id, body.data() + at, sizeof record.id);
    at += sizeof record.id;
    uint32_t dimensions = 0;
    std::memcpy(&dimensions, body.data() + at, sizeof dimensions);
    at += sizeof dimensions;
    if (dimensions) {
      record.vector.resize(dimensions);
      std::memcpy(record.vector.data(), body.data() + at,
                  size_t(dimensions) * sizeof(float));
    }
    *good_bytes += sizeof(uint32_t) + length + sizeof(uint32_t);
    if (record.ts > after)
      out.push_back(std::move(record));
  }
  return out;
}

struct Directory {
  std::vector<std::shared_ptr<Descriptor>> pages;
  std::vector<std::vector<uint32_t>> neighbors;
  std::vector<float> centroids;
  // int8 copy of `centroids`, scanned instead of the float array during
  // routing. At 1M SIFT the float scan is 6,947 x 128 x 4 B = 3.5 MB per
  // query, which matched the measured ~100 us almost exactly: routing was
  // bandwidth-bound, not algorithm-bound. The quantised scan selects a
  // widened candidate set which is then rescored exactly from `centroids`,
  // so the pages finally probed are ordered by true distance.
  // Nibble-packed (4-bit) centroid codes, two dims per byte: low nibble = even
  // dim, high nibble = odd dim. Each nibble is offset by 8 so 0..15 maps to
  // the signed range [-8, 7], matching the int4_dot convention. Stride per
  // page is (dimensions+1)/2 bytes. Halves the bandwidth of the centroid scan
  // versus the previous int8 layout (page_count × d bytes → × d/2 bytes).
  std::vector<uint8_t> centroid_codes;
  std::vector<float> centroid_peak;   // per-centroid quantisation scale (peak abs value)
  std::vector<float> centroid_norm2;  // ||c||^2, for the L2 expansion

  // Coarse second level over the page centroids.
  //
  // Scanning every page centroid is O(pages), and instrumenting the insert
  // path showed that scan is 77% of an insert once the directory passes a few
  // thousand pages -- a faster kernel only trims the constant. Grouping the
  // centroids and visiting a few groups makes both placement and query routing
  // O(sqrt(pages)).
  //
  // Only group *selection* is approximate: the pages a group contributes are
  // then scored exactly, so a stale or imperfect grouping costs candidate
  // quality, never correctness.
  std::vector<float> coarse_centroids;              // groups * dimensions
  std::vector<std::vector<uint32_t>> coarse_members;
  size_t coarse_built_for = 0;   // page count the grouping was built at
};
struct Location {
  std::shared_ptr<Descriptor> page;
  size_t slot;
};
struct Retired {
  uint64_t end, begin;
  int64_t id;
};
struct Candidate {
  float distance;
  int64_t id;
};
struct Worse {
  bool operator()(const Candidate &a, const Candidate &b) const {
    return a.distance < b.distance;
  }
};
struct Screened {
  float estimate;
  uint32_t page;
  uint32_t slot;
};
// Max-heap on the estimate so the worst retained candidate sits at top().
struct BetterScreen {
  bool operator()(const Screened &a, const Screened &b) const {
    return a.estimate < b.estimate;
  }
};


constexpr char CHECKPOINT_MAGIC[8] = {'C', 'H', 'R', 'O', 'N', 'O', '0', '1'};
// Quantised routing carries a fixed cost (wider sort + exact shortlist
// rescore) that only amortises once the centroid scan dominates routing.
constexpr size_t ROUTING_QUANTISED_THRESHOLD = 4096;
// How much farther than the geometrically nearest page a vector may be placed
// in order to join a page that already holds its label.
//
// DEFAULT 0: DISABLED, on measurement. At 0.15 on 60k vectors across 8 scattered
// tenant labels this consolidated 518 pages into 316 and cut capacity
// amplification from 2.21x to 1.35x, but it made pages geometrically worse: at
// an equal candidate budget recall fell (0.867 against 0.90), so at equal recall
// queries were ~11% slower. It also failed at its actual purpose -- page pruning
// still fired on only 1.5 of 256 routed pages, because a page that mixes tenants
// has a saturated label union no matter how it was filled.
//
// The mechanism is kept because the trade is real rather than absurd -- 38% less
// memory for ~11% slower queries suits a memory-bound deployment -- but it is
// off until someone has evidence for their own workload. Pruning needs labels
// that correlate with position in the embedding space; biasing placement is not
// enough to manufacture that correlation.
constexpr float LABEL_PLACEMENT_TOLERANCE = 0.0f;
// Splitting needs at least two live rows on each side, and the coarse and
// merge heuristics assume a page holds enough rows to have a meaningful
// centroid. Below this the structure degenerates into one page per vector.
// How full a merge packs a destination page, as a fraction of capacity.
// Below this, merging leaves too many pages; at capacity, every later insert
// into the page splits it again. Measured across sustained full turnover.
constexpr size_t MERGE_FILL_NUMERATOR = 3;
constexpr size_t MERGE_FILL_DENOMINATOR = 4;
constexpr size_t MIN_PAGE_CAPACITY = 8;
constexpr size_t PLACEMENT_CANDIDATES = 16;
// Coarse groups scored per row by the batched assignment. choose_page gathers a
// widened candidate set that can span more than the two nearest groups, so
// stopping at two placed a small fraction of rows in a slightly worse page:
// measurably lower recall at every nprobe, not just at the frontier.
constexpr size_t ASSIGN_GROUPS = 3;
// Below this the flat scan is cheaper than maintaining a second level. Chosen
// by measurement: at 100k vectors (758 pages) lowering it from 1024 to 256 took
// bulk insert from 8.88us to 6.82us with recall and query latency unchanged.
constexpr size_t COARSE_MIN_PAGES = 256;
// Bounds the pause a single vacuum call can introduce.
constexpr size_t MAX_MERGES_PER_VACUUM = 4096;
// Rebuild the grouping once the directory has grown by this factor, so the
// O(pages * groups) rebuild happens a logarithmic number of times.
constexpr double COARSE_REBUILD_GROWTH = 1.6;
constexpr uint64_t CHECKPOINT_MAX_PAGES = uint64_t{1} << 32;
constexpr uint64_t CHECKPOINT_MAX_DIMENSIONS = uint64_t{1} << 20;

template <class T>
void append_value(std::vector<uint8_t> &out, const T &value) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <class T>
T read_value(const std::vector<uint8_t> &input, size_t &position) {
  if (position > input.size() || input.size() - position < sizeof(T))
    throw std::runtime_error("truncated checkpoint");
  T value;
  std::memcpy(&value, input.data() + position, sizeof(T));
  position += sizeof(T);
  return value;
}

uint64_t checkpoint_hash(const uint8_t *data, size_t size) {
  uint64_t hash = 1469598103934665603ULL;
  for (size_t index = 0; index < size; ++index) {
    hash ^= data[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

void write_all(int descriptor, const uint8_t *data, size_t size) {
  while (size) {
#if defined(_WIN32)
    int count =
        _write(descriptor, data, unsigned(std::min<size_t>(size, 1U << 30)));
#else
    ssize_t count = ::write(descriptor, data, size);
#endif
    if (count <= 0)
      throw std::runtime_error("checkpoint write failed");
    data += count;
    size -= size_t(count);
  }
}
} // namespace

// Tracks the oldest snapshot any in-flight search might still need, so
// cv_vacuum never reclaims (and a subsequent insert never overwrites) a
// slot a concurrent, lock-free reader hasn't finished reading yet.
//
// A slot starts at UINT64_MAX (inactive). A reader claims one for the
// duration of a single search pass via ReaderPin, which registers a
// maximally-conservative 0 *before* it even resolves which snapshot it will
// use, then tightens that to the real value once known. That ordering is
// the whole point: registering only after resolving the snapshot leaves a
// window -- between reading the clock and publishing what was read -- where
// vacuum could scan the registry, not see this reader at all, and reclaim
// exactly the slot it is about to read. Registering conservatively first
// means vacuum can never observe "no protection" while this reader could
// still need an old version; it only ever sees "protect everything" or
// "protect this exact snapshot", never neither.
//
// Slots are recycled through a small mutex-protected free list rather than
// a lock-free structure: a naive lock-free (Treiber-stack) free list has a
// classic ABA hazard on the recycle path, and a plain mutex held for a
// handful of nanoseconds (list push/pop, no allocation in the common case)
// is not worth that risk here -- a search already does heavier work than
// this per call (e.g. `prepare()`'s own heap allocation). The mutex is
// touched by acquire/release (once per search pass) and by cv_vacuum's scan
// (already the expensive, infrequent operation on the write side); nothing
// else on the read path ever contends for it.
struct ReaderRegistry {
  std::mutex mutex;
  std::deque<std::atomic<uint64_t>> slots; // pointer-stable across growth
  std::vector<size_t> free_list;

  size_t acquire() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!free_list.empty()) {
      size_t index = free_list.back();
      free_list.pop_back();
      return index;
    }
    slots.emplace_back(UINT64_MAX);
    return slots.size() - 1;
  }
  void release(size_t index) {
    slots[index].store(UINT64_MAX, std::memory_order_release);
    std::lock_guard<std::mutex> lock(mutex);
    free_list.push_back(index);
  }
  // The oldest snapshot any currently-registered reader might still need,
  // clamped by the caller's requested `oldest`. Called only from cv_vacuum,
  // which already holds writer_mutex, so this only ever races search's
  // acquire/release -- never another vacuum call.
  uint64_t safe_oldest(uint64_t oldest) {
    std::lock_guard<std::mutex> lock(mutex);
    uint64_t floor = oldest;
    for (auto &slot : slots) {
      uint64_t v = slot.load(std::memory_order_acquire);
      if (v < floor)
        floor = v;
    }
    return floor;
  }
};

// RAII: registers this thread's in-flight search snapshot for the lifetime
// of one search pass. See ReaderRegistry for why the constructor pins
// conservatively before `refine()` narrows it to the real snapshot.
struct ReaderPin {
  ReaderRegistry &registry;
  size_t slot;
  explicit ReaderPin(ReaderRegistry &r) : registry(r), slot(r.acquire()) {
    registry.slots[slot].store(0, std::memory_order_release);
  }
  void refine(uint64_t snap) {
    registry.slots[slot].store(snap, std::memory_order_release);
  }
  ~ReaderPin() { registry.release(slot); }
};

struct cv_index {
  size_t dimensions, page_capacity, default_nprobe, rerank_factor;
  int metric;
  bool screening, adaptive_bounds, residual_codes, labels_enabled;
  bool label_partition;
  std::shared_ptr<VectorArena> arena;
  mutable std::mutex writer_mutex;
  AtomicShared<const Directory> directory;
  std::unordered_map<int64_t, std::vector<Location>> locations;
  std::deque<Retired> retired;
  ReaderRegistry reader_registry;
  // Two clocks, because allocating a timestamp and making it visible are
  // different events. `clock` hands out timestamps to writers; `committed` is
  // what a reader snapshots. A write allocates, does its work, and only then
  // commits, so no reader can obtain a snapshot that sees half of it.
  //
  // With one clock a batch insert was not atomic: rows became visible as their
  // pages were published, and a reader sampling the clock mid-batch saw a few
  // hundred rows of four thousand. Nine of twelve sampled reads saw a partial
  // batch.
  std::atomic<uint64_t> clock{0};
  std::atomic<uint64_t> committed{0};
  std::unique_ptr<Wal> wal;
  // Lanes for the batch apply phase. One by default: the rest of the write
  // path is serialised behind writer_mutex, and threading a phase that is a
  // quarter of the work cannot pay back more than a quarter.
  size_t threads = 1;
  std::shared_ptr<VectorArena> record_arena;
  // Consolidate at the end of a large batch insert. On by default: leaving
  // split debris in place is a silent 14x on memory in the worst case, and a
  // caller who never calls vacuum -- which is most of them after a bulk load
  // -- has no way to know. It costs about 19% of build time on clustered data
  // and much more on uniform high-dimensional data, where there is far more
  // debris to clear, so it can be turned off by a caller who vacuums itself.
  bool auto_consolidate = true;
  uint64_t splits = 0, merges = 0, reclaimed = 0, next_page_id = 1;
  uint64_t routing_full_rebuilds = 0, routing_incremental_updates = 0;
  size_t merge_cursor = 0;
  cv_index(size_t d, int m, size_t c, size_t p, uint32_t flags, size_t rerank)
      : dimensions(d), page_capacity(c), default_nprobe(p),
        rerank_factor(rerank ? rerank : 4), metric(m),
        screening(flags & CV_ENABLE_INT8_SCREENING),
        adaptive_bounds(flags & CV_ENABLE_ADAPTIVE_BOUNDS),
        residual_codes(m == CV_L2 || d >= 64),
        labels_enabled(flags & CV_ENABLE_LABELS),
        label_partition((flags & CV_ENABLE_LABEL_PARTITION) &&
                        (flags & CV_ENABLE_LABELS)),
        directory(std::make_shared<const Directory>()) {}
  std::vector<float> prepare(const float *in) const {
    if (!in)
      throw std::invalid_argument("null vector");
    std::vector<float> v(in, in + dimensions);
    for (float x : v)
      if (!std::isfinite(x))
        throw std::invalid_argument("non-finite vector");
    if (metric == CV_COSINE) {
      float n = std::sqrt(dot(v.data(), v.data(), dimensions));
      if (!(n > 0))
        throw std::invalid_argument("zero cosine vector");
      for (float &x : v)
        x /= n;
    }
    return v;
  }
  float distance(const float *a, const float *b) const {
    return metric == CV_COSINE ? 1 - dot(a, b, dimensions)
                               : l2sq(a, b, dimensions);
  }
  float page_lower_bound(const float *query, const Page &page) const {
    float center_distance = distance(query, page.centroid.data());
    if (metric == CV_L2) {
      float gap = std::max(0.0f, std::sqrt(std::max(0.0f, center_distance)) -
                                     std::sqrt(page.radius));
      return gap * gap;
    }
    // Unit-vector cosine distance maps exactly to squared Euclidean / 2, so
    // triangle inequality yields a safe bound even though 1-dot is not a
    // metric.
    float gap =
        std::max(0.0f, std::sqrt(2.0f * std::max(0.0f, center_distance)) -
                           std::sqrt(2.0f * page.radius));
    return 0.5f * gap * gap;
  }
  // Allocate a timestamp. Not visible until commit().
  uint64_t tick(uint64_t ts) {
    uint64_t c = clock.load();
    if (!ts)
      ts = c + 1;
    // An explicit timestamp equal to the clock is allowed: it means "part of
    // the same commit", which is what every row of a batch is, and what
    // replaying that batch from the log must be able to say. Going backwards
    // is still refused.
    if (ts < c)
      throw std::invalid_argument("timestamp must not go backwards");
    clock.store(ts, std::memory_order_release);
    return ts;
  }
  // Make everything stamped at or below `ts` visible to new snapshots. Release
  // ordering pairs with the reader's acquire load, so a reader that sees the
  // timestamp also sees the page images published before it.
  void commit(uint64_t ts) {
    uint64_t seen = committed.load(std::memory_order_relaxed);
    while (ts > seen &&
           !committed.compare_exchange_weak(seen, ts, std::memory_order_release,
                                            std::memory_order_relaxed)) {
    }
  }
  std::shared_ptr<Descriptor> make_desc(std::shared_ptr<Page> p) {
    return std::make_shared<Descriptor>(next_page_id++, std::move(p));
  }
  void publish_page(const std::shared_ptr<Descriptor> &d,
                    const std::shared_ptr<const Page> &before,
                    std::shared_ptr<const Page> after) {
    auto expected = before;
    if (!d->image.compare_exchange_strong(expected, std::move(after),
                                          std::memory_order_release,
                                          std::memory_order_acquire))
      throw std::runtime_error("page publication conflict");
  }
  std::vector<uint32_t> nearest_pages(const Directory &directory, size_t source,
                                      size_t limit = 12) const {
    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(directory.pages.size());
    auto source_page = directory.pages[source]->image.load();
    for (size_t target = 0; target < directory.pages.size(); ++target) {
      if (source == target)
        continue;
      auto target_page = directory.pages[target]->image.load();
      ranked.push_back(
          {distance(source_page->centroid.data(), target_page->centroid.data()),
           uint32_t(target)});
    }
    size_t take = std::min(limit, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + take, ranked.end());
    std::vector<uint32_t> result;
    result.reserve(take + 2);
    for (size_t index = 0; index < take; ++index)
      result.push_back(ranked[index].second);
    return result;
  }

  void add_bounded_edge(Directory &directory, size_t source, uint32_t target,
                        size_t max_degree = 16) const {
    if (source == target)
      return;
    auto &edges = directory.neighbors[source];
    if (std::find(edges.begin(), edges.end(), target) != edges.end())
      return;
    if (edges.size() < max_degree) {
      edges.push_back(target);
      return;
    }
    auto source_page = directory.pages[source]->image.load();
    size_t farthest = 0;
    float farthest_distance = -1.0f;
    for (size_t index = 0; index < edges.size(); ++index) {
      auto page = directory.pages[edges[index]]->image.load();
      float candidate =
          distance(source_page->centroid.data(), page->centroid.data());
      if (candidate > farthest_distance) {
        farthest_distance = candidate;
        farthest = index;
      }
    }
    edges[farthest] = target;
  }

  void reconnect_node(Directory &directory, size_t source) const {
    directory.neighbors[source] = nearest_pages(directory, source);
    if (source)
      add_bounded_edge(directory, source, uint32_t(source - 1));
    if (source + 1 < directory.pages.size())
      add_bounded_edge(directory, source, uint32_t(source + 1));
    auto outbound = directory.neighbors[source];
    for (uint32_t target : outbound)
      add_bounded_edge(directory, target, uint32_t(source));
  }

  void rebuild_routing(Directory &next) {
    size_t n = next.pages.size();
    next.neighbors.assign(n, {});
    if (n <= ROUTING_GRAPH_THRESHOLD)
      return;
    for (size_t page = 0; page < n; ++page)
      next.neighbors[page] = nearest_pages(next, page);
    for (size_t page = 0; page + 1 < n; ++page) {
      add_bounded_edge(next, page, uint32_t(page + 1));
      add_bounded_edge(next, page + 1, uint32_t(page));
    }
    ++routing_full_rebuilds;
  }
  void encode_centroid(Directory &next, size_t page_index) const {
    const float *c = next.centroids.data() + page_index * dimensions;
    float peak = 0.0f, norm2 = 0.0f;
    for (size_t j = 0; j < dimensions; ++j) {
      peak = std::max(peak, std::fabs(c[j]));
      norm2 += c[j] * c[j];
    }
    next.centroid_peak[page_index] = peak;
    next.centroid_norm2[page_index] = norm2;
    // Pack centroid into 4-bit nibbles (range [-8,7], stored as unsigned [0,15]).
    // Low nibble = even dim, high nibble = odd dim; matches int4_dot layout.
    const float scale = peak > 0.0f ? 7.0f / peak : 0.0f;
    const size_t stride = (dimensions + 1) / 2;
    uint8_t *out = next.centroid_codes.data() + page_index * stride;
    for (size_t j = 0; j < dimensions; j += 2) {
      const int lo = int(std::clamp(std::lround(c[j] * scale), -8l, 7l)) + 8;
      const int hi = (j + 1 < dimensions)
                         ? int(std::clamp(std::lround(c[j + 1] * scale), -8l, 7l)) + 8
                         : 8; // neutral (zero) for padding
      out[j >> 1] = uint8_t(lo | (hi << 4));
    }
  }
  void size_centroid_codes(Directory &next) const {
    next.centroid_codes.resize(next.pages.size() * ((dimensions + 1) / 2));
    next.centroid_peak.resize(next.pages.size());
    next.centroid_norm2.resize(next.pages.size());
  }
  size_t coarse_group_of(const Directory &next, const float *centroid) const {
    size_t best = 0;
    float best_distance = std::numeric_limits<float>::infinity();
    for (size_t g = 0; g < next.coarse_members.size(); ++g) {
      const float d = distance(centroid,
                               next.coarse_centroids.data() + g * dimensions);
      if (d < best_distance) {
        best_distance = d;
        best = g;
      }
    }
    return best;
  }
  // Group the page centroids. Seeds are taken at a stride rather than at
  // random so the grouping is deterministic, then two Lloyd passes sharpen it.
  // Two is a deliberate stopping point: group membership only selects
  // candidates, which are rescored exactly, so further passes buy accuracy the
  // rescoring already provides.
  void rebuild_coarse(Directory &next) const {
    const size_t n = next.pages.size();
    next.coarse_members.clear();
    next.coarse_centroids.clear();
    next.coarse_built_for = n;
    if (n < COARSE_MIN_PAGES)
      return;
    const size_t groups = std::max<size_t>(
        8, size_t(std::sqrt(double(n))));
    next.coarse_centroids.resize(groups * dimensions);
    const size_t stride = std::max<size_t>(1, n / groups);
    for (size_t g = 0; g < groups; ++g)
      std::copy_n(next.centroids.data() + std::min(g * stride, n - 1) * dimensions,
                  dimensions, next.coarse_centroids.begin() + g * dimensions);
    std::vector<uint32_t> owner(n, 0);
    for (int pass = 0; pass < 6; ++pass) {
      for (size_t page = 0; page < n; ++page)
        owner[page] = uint32_t(coarse_group_of(
            next, next.centroids.data() + page * dimensions));
      std::vector<double> sums(groups * dimensions, 0.0);
      std::vector<size_t> counts(groups, 0);
      for (size_t page = 0; page < n; ++page) {
        const float *c = next.centroids.data() + page * dimensions;
        double *acc = sums.data() + size_t(owner[page]) * dimensions;
        for (size_t j = 0; j < dimensions; ++j)
          acc[j] += c[j];
        ++counts[owner[page]];
      }
      for (size_t g = 0; g < groups; ++g)
        if (counts[g])
          for (size_t j = 0; j < dimensions; ++j)
            next.coarse_centroids[g * dimensions + j] =
                float(sums[g * dimensions + j] / double(counts[g]));
    }
    // Each page joins its two nearest groups. A single assignment makes the
    // group boundary a recall cliff: a query whose nearest page sits just the
    // other side of a boundary never sees it, and gathering more groups to
    // compensate gives back the saving. Replicating page *indices* is nearly
    // free -- four bytes per page per extra group -- unlike replicating the
    // vectors themselves, which was measured and rejected earlier.
    next.coarse_members.assign(groups, {});
    for (size_t page = 0; page < n; ++page) {
      const float *c = next.centroids.data() + page * dimensions;
      size_t best = 0, second = 0;
      float best_d = std::numeric_limits<float>::infinity();
      float second_d = std::numeric_limits<float>::infinity();
      for (size_t g = 0; g < groups; ++g) {
        const float d = distance(c, next.coarse_centroids.data() + g * dimensions);
        if (d < best_d) {
          second_d = best_d;
          second = best;
          best_d = d;
          best = g;
        } else if (d < second_d) {
          second_d = d;
          second = g;
        }
      }
      next.coarse_members[best].push_back(uint32_t(page));
      if (groups > 1 && second != best)
        next.coarse_members[second].push_back(uint32_t(page));
    }
  }
  bool coarse_usable(const Directory &d) const {
    return !d.coarse_members.empty() &&
           d.coarse_centroids.size() ==
               d.coarse_members.size() * dimensions &&
           d.pages.size() >= COARSE_MIN_PAGES &&
           double(d.pages.size()) <
               double(d.coarse_built_for) * COARSE_REBUILD_GROWTH;
  }
  // Walk groups nearest-first, gathering page indices until `wanted` are in
  // hand. Returns the gathered pages; the caller scores them exactly.
  // Gather into a caller-owned fixed buffer. Returning a std::vector cost a
  // heap allocation and a sort on every insert, which is most of what visiting
  // a few groups instead of every page was meant to save.
  size_t gather_coarse_into(const Directory &d, const float *query, size_t wanted,
                            uint32_t *out, size_t capacity) const {
    const size_t groups = d.coarse_members.size();
    constexpr size_t kMaxSelected = 24;
    std::array<std::pair<float, uint32_t>, kMaxSelected> chosen;
    const size_t average = std::max<size_t>(1, d.pages.size() / groups);
    const size_t want_groups =
        std::min(kMaxSelected, std::max<size_t>(2, wanted / average + 1));
    size_t held = 0;
    for (size_t g = 0; g < groups; ++g) {
      const float score =
          distance(query, d.coarse_centroids.data() + g * dimensions);
      if (held < want_groups) {
        chosen[held++] = {score, uint32_t(g)};
        std::push_heap(chosen.begin(), chosen.begin() + held);
      } else if (score < chosen.front().first) {
        std::pop_heap(chosen.begin(), chosen.begin() + held);
        chosen[held - 1] = {score, uint32_t(g)};
        std::push_heap(chosen.begin(), chosen.begin() + held);
      }
    }
    std::sort_heap(chosen.begin(), chosen.begin() + held);
    size_t count = 0;
    for (size_t index = 0; index < held && count < capacity; ++index)
      for (uint32_t member : d.coarse_members[chosen[index].second]) {
        if (count == capacity)
          break;
        out[count++] = member;
      }
    return count;
  }
  void gather_coarse(const Directory &d, const float *query, size_t wanted,
                     std::vector<uint32_t> &out) const {
    const size_t groups = d.coarse_members.size();
    // Select the few nearest groups with a bounded stack heap. Ranking every
    // group into a heap-allocated vector and sorting it costs two allocations
    // per call, which on the insert path is most of what the second level was
    // meant to save.
    constexpr size_t kMaxSelected = 24;
    std::array<std::pair<float, uint32_t>, kMaxSelected> chosen;
    const size_t average = std::max<size_t>(1, d.pages.size() / groups);
    const size_t want_groups =
        std::min(kMaxSelected, std::max<size_t>(2, wanted / average + 1));
    size_t held = 0;
    for (size_t g = 0; g < groups; ++g) {
      const float score =
          distance(query, d.coarse_centroids.data() + g * dimensions);
      if (held < want_groups) {
        chosen[held++] = {score, uint32_t(g)};
        std::push_heap(chosen.begin(), chosen.begin() + held);
      } else if (score < chosen.front().first) {
        std::pop_heap(chosen.begin(), chosen.begin() + held);
        chosen[held - 1] = {score, uint32_t(g)};
        std::push_heap(chosen.begin(), chosen.begin() + held);
      }
    }
    std::sort_heap(chosen.begin(), chosen.begin() + held);
    out.clear();
    for (size_t index = 0; index < held; ++index) {
      const auto &members = d.coarse_members[chosen[index].second];
      out.insert(out.end(), members.begin(), members.end());
    }
    // Replication means a page can arrive from two groups; probing it twice
    // would waste part of the probe budget.
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
  }
  void refresh_directory_centroids(Directory &next) const {
    next.centroids.resize(next.pages.size() * dimensions);
    for (size_t page_index = 0; page_index < next.pages.size(); ++page_index) {
      auto page = next.pages[page_index]->image.load();
      std::copy(page->centroid.begin(), page->centroid.end(),
                next.centroids.begin() + page_index * dimensions);
    }
    size_centroid_codes(next);
    for (size_t page_index = 0; page_index < next.pages.size(); ++page_index)
      encode_centroid(next, page_index);
  }
  // Rebuild the grouping only when the caller has not carried one forward.
  // Split and merge both hand over a maintained grouping, so the O(pages *
  // groups) rebuild runs a logarithmic number of times rather than per merge.
  void ensure_coarse(Directory &next) const {
    if (next.pages.size() < COARSE_MIN_PAGES) {
      next.coarse_members.clear();
      next.coarse_centroids.clear();
      return;
    }
    const bool carried =
        !next.coarse_members.empty() &&
        next.coarse_centroids.size() == next.coarse_members.size() * dimensions &&
        double(next.pages.size()) <
            double(next.coarse_built_for) * COARSE_REBUILD_GROWTH;
    if (!carried)
      rebuild_coarse(next);
  }
  // Assign one page to its nearest group, without disturbing the others.
  void place_in_coarse(Directory &next, size_t page_index) const {
    if (next.coarse_members.empty())
      return;
    const size_t g = coarse_group_of(
        next, next.centroids.data() + page_index * dimensions);
    next.coarse_members[g].push_back(uint32_t(page_index));
  }
  void publish_dir(Directory next) {
    refresh_directory_centroids(next);
    ensure_coarse(next);
    rebuild_routing(next);
    directory.store(std::make_shared<const Directory>(std::move(next)),
                    std::memory_order_release);
  }

  void publish_split_directory(const std::shared_ptr<const Directory> &current,
                               Directory next, size_t changed_page) {
    const size_t new_page = next.pages.size() - 1;
    if (next.pages.size() <= ROUTING_GRAPH_THRESHOLD &&
        current->centroids.size() == current->pages.size() * dimensions) {
      next.centroids = current->centroids;
      next.centroids.resize(next.pages.size() * dimensions);
      next.centroid_codes = current->centroid_codes;
      next.centroid_peak = current->centroid_peak;
      next.centroid_norm2 = current->centroid_norm2;
      size_centroid_codes(next);
      auto update_centroid = [&](size_t page_index) {
        auto page = next.pages[page_index]->image.load();
        std::copy(page->centroid.begin(), page->centroid.end(),
                  next.centroids.begin() + page_index * dimensions);
        encode_centroid(next, page_index);
      };
      update_centroid(changed_page);
      update_centroid(new_page);
      constexpr size_t refresh_partitions = 64;
      size_t chunk =
          (current->pages.size() + refresh_partitions - 1) / refresh_partitions;
      size_t begin = (splits % refresh_partitions) * chunk;
      size_t end = std::min(current->pages.size(), begin + chunk);
      for (size_t page_index = begin; page_index < end; ++page_index)
        update_centroid(page_index);
      // A split only appends, so existing page indices stay valid and the
      // grouping carries forward; the new page joins its nearest group and the
      // split page keeps its old membership, which stays approximately right
      // until the next rebuild.
      next.coarse_centroids = current->coarse_centroids;
      next.coarse_members = current->coarse_members;
      next.coarse_built_for = current->coarse_built_for;
      ensure_coarse(next);
      if (!next.coarse_members.empty() &&
          next.coarse_built_for != next.pages.size())
        place_in_coarse(next, new_page);
      directory.store(std::make_shared<const Directory>(std::move(next)),
                      std::memory_order_release);
      return;
    }
    if (current->neighbors.size() != current->pages.size() ||
        current->pages.size() <= ROUTING_GRAPH_THRESHOLD) {
      publish_dir(std::move(next));
      return;
    }
    next.neighbors = current->neighbors;
    next.neighbors.resize(next.pages.size());
    reconnect_node(next, changed_page);
    reconnect_node(next, new_page);
    refresh_directory_centroids(next);
    ++routing_incremental_updates;
    directory.store(std::make_shared<const Directory>(std::move(next)),
                    std::memory_order_release);
  }

  // A merge drops two pages and appends one, which renumbers every page after
  // the removed pair. Group membership is therefore remapped rather than
  // rebuilt: remapping is O(pages), rebuilding is O(pages * groups * d) and
  // would cost tens of milliseconds on every merge.
  void carry_coarse_through_merge(const Directory &current, Directory &next,
                                  size_t removed_a, size_t removed_b) const {
    if (current.coarse_members.empty() ||
        current.coarse_centroids.size() !=
            current.coarse_members.size() * dimensions)
      return;
    std::vector<uint32_t> remap(current.pages.size(), UINT32_MAX);
    uint32_t running = 0;
    for (size_t old = 0; old < current.pages.size(); ++old)
      if (old != removed_a && old != removed_b)
        remap[old] = running++;
    next.coarse_centroids = current.coarse_centroids;
    next.coarse_built_for = current.coarse_built_for;
    next.coarse_members.assign(current.coarse_members.size(), {});
    for (size_t g = 0; g < current.coarse_members.size(); ++g) {
      auto &target = next.coarse_members[g];
      target.reserve(current.coarse_members[g].size());
      for (uint32_t member : current.coarse_members[g])
        if (member < remap.size() && remap[member] != UINT32_MAX)
          target.push_back(remap[member]);
    }
  }
  void publish_merge_directory(const std::shared_ptr<const Directory> &current,
                               Directory next, size_t removed_a,
                               size_t removed_b) {
    const size_t merged_page = next.pages.size() - 1;
    if (next.pages.size() <= ROUTING_GRAPH_THRESHOLD ||
        current->neighbors.size() != current->pages.size()) {
      carry_coarse_through_merge(*current, next, removed_a, removed_b);
      refresh_directory_centroids(next);
      ensure_coarse(next);
      if (!next.coarse_members.empty() &&
          next.coarse_built_for != next.pages.size())
        place_in_coarse(next, merged_page);
      rebuild_routing(next);
      directory.store(std::make_shared<const Directory>(std::move(next)),
                      std::memory_order_release);
      return;
    }
    std::vector<uint32_t> remap(current->pages.size(), UINT32_MAX);
    uint32_t next_index = 0;
    for (size_t old = 0; old < current->pages.size(); ++old)
      if (old != removed_a && old != removed_b)
        remap[old] = next_index++;
    const uint32_t merged_index = uint32_t(next.pages.size() - 1);
    next.neighbors.assign(next.pages.size(), {});
    for (size_t old = 0; old < current->pages.size(); ++old) {
      if (remap[old] == UINT32_MAX)
        continue;
      auto &target_edges = next.neighbors[remap[old]];
      for (uint32_t neighbor : current->neighbors[old]) {
        uint32_t mapped = remap[neighbor];
        if (mapped == UINT32_MAX)
          mapped = merged_index;
        if (mapped != remap[old] &&
            std::find(target_edges.begin(), target_edges.end(), mapped) ==
                target_edges.end())
          target_edges.push_back(mapped);
      }
    }
    reconnect_node(next, merged_index);
    refresh_directory_centroids(next);
    ++routing_incremental_updates;
    directory.store(std::make_shared<const Directory>(std::move(next)),
                    std::memory_order_release);
  }
  void close_current(int64_t id, uint64_t ts, bool required) {
    auto f = locations.find(id);
    if (f != locations.end())
      for (auto it = f->second.rbegin(); it != f->second.rend(); ++it) {
        auto before = it->page->image.load();
        size_t s = it->slot;
        if (before->occupied[s] && before->ids[s] == id &&
            before->end[s] == MAX_TS) {
          auto after = std::make_shared<Page>(*before);
          after->end[s] = ts;
          after->apply_delta(s, false, metric);
          publish_page(it->page, before, after);
          retired.push_back({ts, before->begin[s], id});
          return;
        }
      }
    if (required)
      throw std::out_of_range("id not found");
  }
  // Read-only twin of close_current's lookup, for callers that must validate
  // every id before mutating any of them (a multi-row publish cannot afford
  // to discover a missing id after earlier rows already published).
  bool has_open_version(int64_t id) const {
    auto f = locations.find(id);
    if (f == locations.end())
      return false;
    for (auto it = f->second.rbegin(); it != f->second.rend(); ++it) {
      auto image = it->page->image.load();
      size_t s = it->slot;
      if (image->occupied[s] && image->ids[s] == id && image->end[s] == MAX_TS)
        return true;
    }
    return false;
  }
  void split(const std::shared_ptr<Descriptor> &old_desc) {
    auto old = old_desc->image.load();
    std::vector<size_t> slots;
    for (size_t s = 0; s < old->capacity; ++s)
      if (old->occupied[s])
        slots.push_back(s);
    if (slots.size() < 2)
      throw std::runtime_error("cannot split page");
    std::vector<float> c0(old->vec(slots[0]), old->vec(slots[0]) + dimensions),
        c1(dimensions);
    // Not named `far`: that identifier is a legacy Win16 macro (empty
    // expansion) still defined by <windows.h>, which silently deletes it.
    size_t far_slot = slots[0];
    float far_d = -1;
    for (size_t s : slots) {
      float d = distance(old->vec(s), c0.data());
      if (d > far_d) {
        far_d = d;
        far_slot = s;
      }
    }
    std::copy(old->vec(far_slot), old->vec(far_slot) + dimensions, c1.begin());
    std::vector<uint8_t> labels(slots.size());
    for (int iter = 0; iter < 4; ++iter) {
      size_t n0 = 0, n1 = 0;
      for (size_t i = 0; i < slots.size(); ++i) {
        labels[i] = distance(old->vec(slots[i]), c1.data()) <
                    distance(old->vec(slots[i]), c0.data());
        labels[i] ? ++n1 : ++n0;
      }
      if (!n0)
        labels[0] = 0;
      if (!n1)
        labels.back() = 1;
      std::vector<double> s0(dimensions), s1(dimensions);
      n0 = n1 = 0;
      for (size_t i = 0; i < slots.size(); ++i) {
        auto &s = labels[i] ? s1 : s0;
        labels[i] ? ++n1 : ++n0;
        for (size_t j = 0; j < dimensions; ++j)
          s[j] += old->vec(slots[i])[j];
      }
      for (size_t j = 0; j < dimensions; ++j) {
        c0[j] = float(s0[j] / n0);
        c1[j] = float(s1[j] / n1);
      }
      if (metric == CV_COSINE)
        for (auto *c : {&c0, &c1}) {
          float n = std::sqrt(dot(c->data(), c->data(), dimensions));
          if (n > 0)
            for (float &x : *c)
              x /= n;
        }
    }
    auto left = std::make_shared<Page>(page_capacity, dimensions, screening,
                                       adaptive_bounds, residual_codes, labels_enabled, arena, record_arena),
         right = std::make_shared<Page>(page_capacity, dimensions, screening,
                                        adaptive_bounds, residual_codes,
                                        labels_enabled, arena, record_arena);
    std::unordered_set<int64_t> moved;
    for (size_t i = 0; i < slots.size(); ++i) {
      Page *t = labels[i] ? right.get() : left.get();
      size_t dst = t->free_slot(), src = slots[i];
      t->ids[dst] = old->ids[src];
      t->begin[dst] = old->begin[src];
      t->end[dst] = old->end[src];
      t->occupied[dst] = 1;
      t->filled.push_back(uint32_t(dst));
      std::copy(old->vec(src), old->vec(src) + dimensions, t->vec(dst));
      if (screening) {
        std::copy_n(old->records->data() + src * old->record_stride,
                    old->record_stride,
                    t->records->data() + dst * t->record_stride);
      }
      if (!old->labels.empty())
        t->labels[dst] = old->labels[src];
      moved.insert(old->ids[src]);
    }
    // Children inherit the parent's code frame; recompute() rebinds only if
    // the new centroid has drifted far enough to matter.
    if (screening)
      for (auto &child : {left, right}) {
        child->code_centroid = old->code_centroid;
        child->code_scale = old->code_scale;
        child->codes_bound = old->codes_bound;
      }
    left->recompute(metric);
    right->recompute(metric);
    auto ld = make_desc(left), rd = make_desc(right);
    for (int64_t id : moved) {
      auto &items = locations[id];
      items.erase(std::remove_if(items.begin(), items.end(),
                                 [&](auto &x) { return x.page == old_desc; }),
                  items.end());
    }
    for (auto d : {ld, rd}) {
      auto p = d->image.load();
      for (uint32_t s : p->filled)
        locations[p->ids[s]].push_back({d, s});
    }
    auto cur = directory.load();
    Directory next;
    next.pages = cur->pages;
    auto pos = std::find(next.pages.begin(), next.pages.end(), old_desc);
    size_t changed_page = size_t(std::distance(next.pages.begin(), pos));
    *pos = ld;
    next.pages.push_back(rd);
    publish_split_directory(cur, std::move(next), changed_page);
    ++splits;
  }
  // Place a whole block of prepared rows against a frozen directory image.
  //
  // choose_page walks the centroids once per row. With the rows known up front
  // the same work becomes a matmul, which is the difference between a scalar
  // scan and the platform's matrix unit. The directory is not mutated here:
  // the caller applies the result, and any row whose page filled in the
  // meantime falls back to choose_page, so a stale assignment costs one
  // re-placement and never correctness.
  //
  // The coarse level is preserved rather than bypassed. Scoring every page for
  // every row would be a bigger matmul but an O(pages) one, and the whole point
  // of the coarse level is that placement cost must not grow linearly with the
  // directory. Stage one picks the two nearest groups per row, stage two scores
  // only those groups' members -- the same candidate set gather_coarse_into
  // produces, computed densely.
  // `row_labels` non-null restricts each row to pages already carrying its
  // label, which is what keeps a partitioned index pure through a bulk insert.
  // The caller guarantees every label in the block already has a page, so
  // there is always a candidate and no row needs a sentinel.
  void assign_block(const Directory &dir, const float *rows, size_t n,
                    uint32_t *out,
                    const uint64_t *row_labels = nullptr,
                    const uint64_t *page_labels = nullptr) const {
    const size_t d = dimensions;
    const size_t m = dir.pages.size();
    const bool l2 = metric == CV_L2;
    std::vector<float> owned_norm2;
    const float *norm2 = nullptr;
    if (l2) {
      if (dir.centroid_norm2.size() == m) {
        norm2 = dir.centroid_norm2.data();
      } else {
        owned_norm2.resize(m);
        for (size_t c = 0; c < m; ++c)
          owned_norm2[c] = dot(dir.centroids.data() + c * d,
                               dir.centroids.data() + c * d, d);
        norm2 = owned_norm2.data();
      }
    }
    // ||row||^2 is common to every candidate for a row, so it is dropped: it
    // shifts all of that row's scores equally and cannot change the argmin.
    auto score = [&](float inner, size_t page) {
      return l2 ? norm2[page] - 2.0f * inner : -inner;
    };

    // Label-restricted assignment takes the dense path. The coarse path
    // prunes to a few groups before scoring pages, and those groups are built
    // on geometry alone, so a row's own label may have no page in any of them.
    const size_t groups = dir.coarse_members.size();
    const bool coarse = groups >= 2 &&
                        dir.coarse_centroids.size() == groups * d &&
                        !(row_labels && page_labels);
    if (!coarse) {
      if (row_labels && page_labels) {
        std::vector<float> s(n * m);
        inner_products(rows, n, dir.centroids.data(), m, d, s.data());
        for (size_t r = 0; r < n; ++r) {
          const float *sr = s.data() + r * m;
          size_t best = m;
          float bv = std::numeric_limits<float>::infinity();
          for (size_t c = 0; c < m; ++c) {
            if (page_labels[c] != row_labels[r])
              continue;
            const float v = score(sr[c], c);
            if (v < bv) { bv = v; best = c; }
          }
          // No page for this label yet: the nearest page overall keeps the
          // round making progress, and the row is corrected by the pending
          // path rather than silently landing in a mixed page.
          if (best == m) {
            best = 0;
            bv = score(sr[0], 0);
            for (size_t c = 1; c < m; ++c) {
              const float v = score(sr[c], c);
              if (v < bv) { bv = v; best = c; }
            }
          }
          out[r] = uint32_t(best);
        }
        return;
      }
      std::vector<float> s(n * m);
      inner_products(rows, n, dir.centroids.data(), m, d, s.data());
      for (size_t r = 0; r < n; ++r) {
        const float *sr = s.data() + r * m;
        size_t best = 0;
        float bv = score(sr[0], 0);
        for (size_t c = 1; c < m; ++c) {
          const float v = score(sr[c], c);
          if (v < bv) { bv = v; best = c; }
        }
        out[r] = uint32_t(best);
      }
      return;
    }

    std::vector<float> gn2(groups);
    for (size_t g = 0; g < groups; ++g)
      gn2[g] = dot(dir.coarse_centroids.data() + g * d,
                   dir.coarse_centroids.data() + g * d, d);
    std::vector<float> s1(n * groups);
    inner_products(rows, n, dir.coarse_centroids.data(), groups, d, s1.data());

    // Rows are bucketed by group so stage two is one dense matmul per group
    // rather than a ragged one per row. Each row appears in the buckets of both
    // of its chosen groups; the second visit can only improve its best.
    std::vector<std::vector<uint32_t>> bucket(groups);
    std::array<size_t, ASSIGN_GROUPS> pick;
    std::array<float, ASSIGN_GROUPS> pick_score;
    for (size_t r = 0; r < n; ++r) {
      const float *sr = s1.data() + r * groups;
      pick.fill(SIZE_MAX);
      pick_score.fill(std::numeric_limits<float>::infinity());
      for (size_t g = 0; g < groups; ++g) {
        const float v = l2 ? gn2[g] - 2.0f * sr[g] : -sr[g];
        if (v >= pick_score[ASSIGN_GROUPS - 1])
          continue;
        size_t at = ASSIGN_GROUPS - 1;
        while (at > 0 && pick_score[at - 1] > v) {
          pick_score[at] = pick_score[at - 1];
          pick[at] = pick[at - 1];
          --at;
        }
        pick_score[at] = v;
        pick[at] = g;
      }
      for (size_t k = 0; k < ASSIGN_GROUPS; ++k) {
        if (pick[k] == SIZE_MAX || dir.coarse_members[pick[k]].empty())
          continue;
        bool seen = false;
        for (size_t j = 0; j < k; ++j)
          seen = seen || pick[j] == pick[k];
        if (!seen)
          bucket[pick[k]].push_back(uint32_t(r));
      }
      out[r] = UINT32_MAX;
    }

    std::vector<float> best(n, std::numeric_limits<float>::infinity());
    std::vector<float> rowbuf, centbuf, s2;
    for (size_t g = 0; g < groups; ++g) {
      const auto &members = dir.coarse_members[g];
      const auto &rowsg = bucket[g];
      if (members.empty() || rowsg.empty())
        continue;
      const size_t mg = members.size(), rg = rowsg.size();
      centbuf.resize(mg * d);
      for (size_t c = 0; c < mg; ++c)
        std::copy_n(dir.centroids.data() + size_t(members[c]) * d, d,
                    centbuf.data() + c * d);
      rowbuf.resize(rg * d);
      for (size_t r = 0; r < rg; ++r)
        std::copy_n(rows + size_t(rowsg[r]) * d, d, rowbuf.data() + r * d);
      s2.assign(rg * mg, 0.0f);
      inner_products(rowbuf.data(), rg, centbuf.data(), mg, d, s2.data());
      for (size_t r = 0; r < rg; ++r) {
        const float *sr = s2.data() + r * mg;
        const uint32_t row = rowsg[r];
        for (size_t c = 0; c < mg; ++c) {
          const float v = score(sr[c], members[c]);
          if (v < best[row]) { best[row] = v; out[row] = members[c]; }
        }
      }
    }
    // A row whose groups were all empty has no assignment; the caller routes it
    // through choose_page rather than guessing.
    for (size_t r = 0; r < n; ++r)
      if (out[r] == UINT32_MAX)
        out[r] = 0;
  }
  std::shared_ptr<Descriptor> choose_page(const float *v, uint64_t label = 0) {
    auto cur = directory.load();
    if (cur->pages.empty()) {
      Directory next;
      auto d = make_desc(std::make_shared<Page>(
          page_capacity, dimensions, screening, adaptive_bounds,
          residual_codes, labels_enabled, arena, record_arena));
      next.pages.push_back(d);
      publish_dir(std::move(next));
      return d;
    }
    const size_t n = cur->pages.size();
    if (cur->centroids.size() != n * dimensions) {
      // Directory published without a centroid snapshot: fall back to the page
      // images rather than guessing.
      auto best = cur->pages[0];
      float bd = distance(v, best->image.load()->centroid.data());
      for (size_t i = 1; i < n; ++i) {
        float d = distance(v, cur->pages[i]->image.load()->centroid.data());
        if (d < bd) {
          bd = d;
          best = cur->pages[i];
        }
      }
      if (best->image.load()->filled.size() < page_capacity)
        return best;
      split(best);
      return choose_page(v, label);
    }

    // Rank on the contiguous centroid array, so placement touches no
    // shared_ptr; only the handful of candidates below are ever loaded.
    //
    // Above the quantised threshold the codes are scanned instead of the floats
    // and the survivors rescored exactly, exactly as the query path does. The
    // insert path is where this matters most: fitting insert cost as
    // `11.4us + 7.69ns * pages` puts routing at 73% of an insert once the
    // directory passes a few thousand pages, so the float scan was the single
    // largest term in the write path.
    std::array<std::pair<float, uint32_t>, PLACEMENT_CANDIDATES> nearest;
    size_t held = 0;
    auto consider = [&](float score, uint32_t page) {
      if (held < PLACEMENT_CANDIDATES) {
        nearest[held++] = {score, page};
        std::push_heap(nearest.begin(), nearest.begin() + held);
      } else if (score < nearest.front().first) {
        std::pop_heap(nearest.begin(), nearest.begin() + held);
        nearest[held - 1] = {score, page};
        std::push_heap(nearest.begin(), nearest.begin() + held);
      }
    };
    if (coarse_usable(*cur)) {
      // Placement only needs a page near the vector, so a few groups suffice.
      // Duplicates from two-way group membership are harmless: `consider`
      // keeps a bounded best-set, and a repeated page only ever ties itself.
      constexpr size_t kGatherCapacity = 512;
      uint32_t gathered[kGatherCapacity];
      const size_t found = gather_coarse_into(
          *cur, v, PLACEMENT_CANDIDATES * 4, gathered, kGatherCapacity);
      for (size_t index = 0; index < found; ++index)
        consider(distance(v, cur->centroids.data() +
                                 size_t(gathered[index]) * dimensions),
                 gathered[index]);
      std::sort_heap(nearest.begin(), nearest.begin() + held);
    } else if (cur->centroid_codes.size() == n * ((dimensions + 1) / 2) &&
               cur->centroid_peak.size() == n &&
               n >= ROUTING_QUANTISED_THRESHOLD) {
      float peak = 0.0f;
      for (size_t j = 0; j < dimensions; ++j)
        peak = std::max(peak, std::fabs(v[j]));
      std::vector<int8_t> code(dimensions);
      const float to_code = peak > 0.0f ? 127.0f / peak : 0.0f;
      for (size_t j = 0; j < dimensions; ++j)
        code[j] = int8_t(std::clamp(std::lround(v[j] * to_code), -127l, 127l));
      // Query is int8 (127-scaled), centroids are 4-bit nibbles (7-scaled).
      const float inner_scale = peak / (127.0f * 7.0f);
      const size_t cc_stride = (dimensions + 1) / 2;
      for (size_t i = 0; i < n; ++i) {
        const float inner =
            float(int4_dot(code.data(), cur->centroid_codes.data() + i * cc_stride,
                           dimensions)) *
            inner_scale * cur->centroid_peak[i];
        consider(metric == CV_COSINE ? -inner
                                     : cur->centroid_norm2[i] - 2.0f * inner,
                 uint32_t(i));
      }
      std::sort_heap(nearest.begin(), nearest.begin() + held);
      // Approximate scores only chose the shortlist; order it by true distance.
      for (size_t c = 0; c < held; ++c)
        nearest[c].first =
            distance(v, cur->centroids.data() + nearest[c].second * dimensions);
      std::sort(nearest.begin(), nearest.begin() + held);
    } else {
      for (size_t i = 0; i < n; ++i)
        consider(distance(v, cur->centroids.data() + i * dimensions), uint32_t(i));
      std::sort_heap(nearest.begin(), nearest.begin() + held);
    }

    // Strict partitioning: the page must hold this label and nothing else.
    // Unlike the tolerance below this does not give up when the nearest pages
    // are the wrong label -- it walks the whole ranking, because a label's
    // pages can be anywhere and settling for a mixed page is what produced
    // the thin spread this mode exists to remove. Failing to find one, the
    // nearest empty page is claimed for the label, and failing that a split
    // makes room.
    if (label && label_partition) {
      auto pick = [&](bool want_empty) -> std::shared_ptr<Descriptor> {
        float best = std::numeric_limits<float>::infinity();
        size_t chosen = n;
        for (size_t index = 0; index < n; ++index) {
          auto image = cur->pages[index]->image.load();
          if (image->filled.size() >= page_capacity)
            continue;
          const bool empty = image->label_union == 0;
          if (want_empty ? !empty : image->label_union != label)
            continue;
          const float d =
              distance(v, cur->centroids.data() + index * dimensions);
          if (d < best) {
            best = d;
            chosen = index;
          }
        }
        return chosen == n ? nullptr : cur->pages[chosen];
      };
      if (auto same = pick(false))
        return same;
      if (auto blank = pick(true))
        return blank;
      // Every page carrying this label is full. Splitting one keeps the
      // halves pure and makes room.
      size_t target = n;
      float best = std::numeric_limits<float>::infinity();
      for (size_t index = 0; index < n; ++index) {
        if (cur->pages[index]->image.load()->label_union != label)
          continue;
        const float d = distance(v, cur->centroids.data() + index * dimensions);
        if (d < best) {
          best = d;
          target = index;
        }
      }
      if (target != n) {
        split(cur->pages[uint32_t(target)]);
        return choose_page(v, label);
      }
      // The label has no page at all. Splitting another label's page would
      // not make room for this one and would recurse until the split failed,
      // so the directory grows by one page instead. This is where the mode
      // spends its page budget: a label costs at least one page whether it
      // holds a million records or one.
      Directory next;
      next.pages = cur->pages;
      auto fresh = make_desc(std::make_shared<Page>(
          page_capacity, dimensions, screening, adaptive_bounds,
          residual_codes, labels_enabled, arena, record_arena));
      next.pages.push_back(fresh);
      publish_dir(std::move(next));
      return fresh;
    }

    // Prefer a page that already holds this label, but only while it stays
    // within the tolerance of the nearest page overall.
    if (label && labels_enabled && held) {
      const float limit = nearest[0].first * (1.0f + LABEL_PLACEMENT_TOLERANCE);
      for (size_t c = 0; c < held && nearest[c].first <= limit; ++c) {
        auto descriptor = cur->pages[nearest[c].second];
        auto image = descriptor->image.load();
        if ((image->label_union & label) && image->filled.size() < page_capacity)
          return descriptor;
      }
    }
    // Without a usable label preference the behaviour is exactly as before:
    // take the nearest page, and split it when it is full rather than drifting
    // to a farther one. Falling back to farther pages raised fill but changed
    // placement geometry for unlabelled inserts too, which is not this change's
    // business.
    auto best = cur->pages[nearest[0].second];
    if (best->image.load()->filled.size() < page_capacity)
      return best;
    split(best);
    return choose_page(v, label);
  }
  // Consolidate up to `wanted` pairs of under-filled pages against one
  // directory image, publishing once.
  //
  // Doing this a pair at a time was 91% of vacuum: 230ms of a 252ms call at
  // 100k rows. Not because merging is expensive -- each merge copies two pages
  // -- but because each one rebuilt the directory. That is an O(pages) vector
  // of shared_ptr copies plus an O(pages * dimensions) centroid rebuild, paid
  // 781 times for 781 merges. The merges themselves were never the cost; the
  // republishing between them was.
  //
  // Pairs are disjoint, so a page never participates twice, and occupancy is
  // read once per page here instead of once per page per merge -- those loads
  // are atomic shared_ptr reads, which on libc++ take a lock from a global
  // pool.
  size_t merge_many(size_t wanted) {
    if (!wanted)
      return 0;
    auto cur = directory.load();
    const size_t n = cur->pages.size();
    if (n < 2 || cur->centroids.size() != n * dimensions)
      return 0;
    const size_t sparse_threshold = page_capacity * 3 / 5;

    std::vector<uint32_t> counts(n);
    for (size_t index = 0; index < n; ++index)
      counts[index] = uint32_t(cur->pages[index]->image.load()->occupied_count());

    // Groups, not pairs. Pairwise merging cannot fill a page: two pages
    // averaging 78 rows make one of 157, which is 61% of a 256-row page, and
    // two of *those* no longer fit together. So a pass halves the page count
    // and then stalls with the result still well under half empty. Under
    // sustained full turnover that left splits outpacing merges every cycle
    // and the page count climbing without bound -- amplification 1.89 to 3.35
    // over thirteen turnovers, in the workload this index exists for.
    //
    // Packing a destination from as many near sources as fit reaches the fill
    // the capacity actually allows, and it is the same scan: take the nearest
    // eligible page repeatedly instead of once.
    std::vector<uint8_t> used(n, 0);
    std::vector<std::vector<uint32_t>> groups;
    size_t merged_pages = 0;

    // Drop pages that hold nothing.
    //
    // This is where the unbounded growth came from. Vacuum frees every slot in
    // a page it has emptied, but nothing removed the page, and the merge scan
    // skipped it as having no rows to move. So empty pages accumulated for
    // ever, each costing a full page of capacity while holding no records. One
    // merge pass would leave 537 pages of which only 21 were sparse and some
    // 250 were empty, and the next pass had nothing to do. Amplification
    // climbed 1.89 to 3.35 over thirteen full turnovers and kept going.
    //
    // They need no data moved, so they are simply not carried into the next
    // directory. One page is always kept: an empty index still needs somewhere
    // for the next insert to land.
    size_t non_empty = 0;
    for (size_t index = 0; index < n; ++index)
      non_empty += counts[index] ? 1 : 0;
    size_t kept_empty = 0;
    for (size_t index = 0; index < n; ++index) {
      if (counts[index])
        continue;
      // Keep one only when nothing else would remain. The earlier condition
      // kept the first empty page unconditionally, so a churning index carried
      // a spare empty page for ever even with hundreds of live ones.
      if (non_empty == 0 && kept_empty == 0) {
        ++kept_empty;
        continue;
      }
      used[index] = 1;
      ++merged_pages;
    }
    const size_t dropped_empty = merged_pages;
    const size_t begin_at = n ? merge_cursor % n : 0;
    for (size_t step = 0; step < n && merged_pages < wanted; ++step) {
      const size_t source = (begin_at + step) % n;
      if (used[source] || counts[source] == 0 ||
          counts[source] >= sparse_threshold)
        continue;
      std::vector<uint32_t> group{uint32_t(source)};
      size_t held = counts[source];
      used[source] = 1;
      // Under partitioning a merge may only join pages carrying the same
      // label. Merging by geometry alone would undo the placement rule at the
      // first consolidation, and silently: the pages would still answer
      // correctly, just with the thin spread back.
      const uint64_t group_label =
          label_partition ? cur->pages[source]->image.load()->label_union : 0;
      // Pack to a fill target, not to capacity. Filling a page completely
      // makes the next insert that routes to it split immediately, and a split
      // of a full page yields two half-full ones -- packing tight measured
      // *worse* than pairwise merging (amplification 10.5 against 3.3) because
      // it drove the split rate up faster than it drove the page count down.
      const size_t pack_to = page_capacity * MERGE_FILL_NUMERATOR /
                             MERGE_FILL_DENOMINATOR;
      // Grow the group while another page still fits. Distances come off the
      // contiguous centroid array, so the scan touches no shared_ptr.
      const float *centroid = cur->centroids.data() + source * dimensions;
      while (held < pack_to) {
        size_t target = n;
        float best = std::numeric_limits<float>::infinity();
        for (size_t index = 0; index < n; ++index) {
          if (used[index] || counts[index] == 0)
            continue;
          if (held + counts[index] > pack_to)
            continue;
          if (label_partition &&
              cur->pages[index]->image.load()->label_union != group_label)
            continue;
          const float d =
              distance(centroid, cur->centroids.data() + index * dimensions);
          if (d < best) {
            best = d;
            target = index;
          }
        }
        if (target == n)
          break;
        used[target] = 1;
        held += counts[target];
        group.push_back(uint32_t(target));
      }
      if (group.size() < 2) {
        used[source] = 0;      // nothing to merge with; leave it for later
        continue;
      }
      merged_pages += group.size() - 1;   // pages removed by this group
      groups.push_back(std::move(group));
    }
    if (groups.empty() && !dropped_empty) {
      merge_cursor = 0;
      return 0;
    }
    if (!groups.empty())
      merge_cursor = (groups.back().front() + 1) % n;

    std::vector<std::shared_ptr<Descriptor>> made;
    made.reserve(groups.size());
    for (const auto &group : groups) {
      auto merged = std::make_shared<Page>(page_capacity, dimensions, screening,
                                           adaptive_bounds, residual_codes,
                                           labels_enabled, arena,
                                           record_arena);
      std::unordered_set<int64_t> moved;
      for (uint32_t which : group) {
        auto p = cur->pages[which]->image.load();
        for (uint32_t s : p->filled) {
          size_t d = merged->free_slot();
          merged->ids[d] = p->ids[s];
          merged->begin[d] = p->begin[s];
          merged->end[d] = p->end[s];
          merged->occupied[d] = 1;
          merged->filled.push_back(uint32_t(d));
          std::copy(p->vec(s), p->vec(s) + dimensions, merged->vec(d));
          if (screening) {
            std::copy_n(p->records->data() + s * p->record_stride,
                        p->record_stride,
                        merged->records->data() + d * merged->record_stride);
          }
          if (!p->labels.empty())
            merged->labels[d] = p->labels[s];
          moved.insert(p->ids[s]);
        }
      }
      merged->recompute(metric);
      if (screening)
        merged->bind_codes();  // two frames fused; a single frame is rebuilt
      auto md = make_desc(merged);
      for (int64_t id : moved) {
        auto &items = locations[id];
        items.erase(std::remove_if(items.begin(), items.end(),
                                   [&](auto &x) {
                                     for (uint32_t which : group)
                                       if (x.page == cur->pages[which])
                                         return true;
                                     return false;
                                   }),
                    items.end());
      }
      for (uint32_t s : merged->filled)
        locations[merged->ids[s]].push_back({md, s});
      made.push_back(md);
    }

    Directory next;
    next.pages.reserve(n - merged_pages);
    for (size_t index = 0; index < n; ++index)
      if (!used[index])
        next.pages.push_back(cur->pages[index]);
    for (auto &descriptor : made)
      next.pages.push_back(descriptor);
    // One full rebuild rather than an incremental fixup per pair: with many
    // merges at once the incremental path is the more expensive of the two.
    publish_dir(std::move(next));
    merges += merged_pages;
    return merged_pages;
  }
  bool merge_once() { return merge_many(1) > 0; }
  // Merge until the directory is no larger than the live data justifies.
  //
  // Merging only ever ran inside vacuum, so an index that was bulk loaded and
  // never vacuumed kept every page a split had produced: amplification 13.86
  // at d=768 against 1.22 after a single pass. That is the first thing anyone
  // does with an index, and the bloat was invisible until they measured it.
  size_t consolidate(size_t budget) {
    // Occupied slots, not live records: a slot holding a version some snapshot
    // can still reach is genuinely unavailable, so a pinned snapshot correctly
    // raises the target instead of driving merges that cannot help.
    size_t occupied_slots = 0;
    auto cur = directory.load();
    for (const auto &page : cur->pages)
      occupied_slots += page->image.load()->occupied_count();
    const size_t ideal = (occupied_slots + page_capacity - 1) / page_capacity;
    // Headroom, because merging to the exact minimum would make the next
    // insert split again immediately.
    const size_t allowed = ideal + ideal / 2 + 1;
    size_t wanted = cur->pages.size() > allowed ? cur->pages.size() - allowed : 0;
    wanted = std::min<size_t>(wanted, budget);
    size_t done_total = 0;
    while (wanted) {
      const size_t done = merge_many(wanted);
      if (!done)
        break;
      done_total += done;
      wanted -= done;
    }
    return done_total;
  }
  // Rank the pages a filter admits, and return the best `probes` of them.
  //
  // Separate from `route` because the two answer different questions. `route`
  // ranks the whole directory and takes a prefix, which is right when the
  // answer can come from any page. Under a filter the eligible pages are a
  // sparse subset of that ranking, so a prefix of it can contain almost none
  // of them -- that is what returned 1.6 of 10 requested neighbours before the
  // eligible set was ranked on its own.
  //
  // The ranking reuses the quantised centroid codes rather than scoring every
  // eligible page exactly. Scoring exactly cost O(eligible x dimensions) on
  // every query and grew with the directory, so it showed up as filtered
  // search degrading under churn: 29us of it on a fresh index and 48us after
  // the page count doubled, on top of a search that took 140us. Only shortlist
  // membership is approximate; the pages actually probed are ordered by true
  // distance.
  std::vector<uint32_t> route_filtered(
      const std::shared_ptr<const Directory> &cur, const float *q,
      size_t probes, const std::vector<uint32_t> &eligible) const {
    const size_t n = cur->pages.size();
    if (cur->centroids.size() != n * dimensions)
      throw std::runtime_error("invalid routing centroid snapshot");
    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(eligible.size());
    const bool quantised = cur->centroid_codes.size() == n * ((dimensions + 1) / 2) &&
                           cur->centroid_peak.size() == n &&
                           eligible.size() >= ROUTING_QUANTISED_THRESHOLD;
    if (quantised) {
      float query_peak = 0.0f;
      for (size_t j = 0; j < dimensions; ++j)
        query_peak = std::max(query_peak, std::fabs(q[j]));
      std::vector<int8_t> query_code(dimensions);
      const float to_code = query_peak > 0.0f ? 127.0f / query_peak : 0.0f;
      for (size_t j = 0; j < dimensions; ++j)
        query_code[j] =
            int8_t(std::clamp(std::lround(q[j] * to_code), -127l, 127l));
      const float inner_scale = query_peak / (127.0f * 7.0f);
      const size_t cc_stride = (dimensions + 1) / 2;
      for (uint32_t index : eligible) {
        const float inner =
            float(int4_dot(query_code.data(),
                           cur->centroid_codes.data() + size_t(index) * cc_stride,
                           dimensions)) *
            inner_scale * cur->centroid_peak[index];
        const float score = metric == CV_COSINE
                                ? -inner
                                : cur->centroid_norm2[index] - 2.0f * inner;
        ranked.push_back({score, index});
        ++last_search_metrics.centroid_scores;
      }
      const size_t want = std::min(probes, ranked.size());
      const size_t shortlist =
          std::min(ranked.size(), std::max(want + want / 4, want + 16));
      if (shortlist < ranked.size()) {
        std::nth_element(ranked.begin(), ranked.begin() + shortlist,
                         ranked.end());
        ranked.resize(shortlist);
      }
      for (auto &item : ranked)
        item.first =
            distance(q, cur->centroids.data() + size_t(item.second) * dimensions);
    } else {
      for (uint32_t index : eligible) {
        ranked.push_back(
            {distance(q, cur->centroids.data() + size_t(index) * dimensions),
             index});
        ++last_search_metrics.centroid_scores;
      }
    }
    const size_t take = std::min(probes ? probes : ranked.size(), ranked.size());
    if (take < ranked.size())
      std::nth_element(ranked.begin(), ranked.begin() + take, ranked.end());
    std::sort(ranked.begin(), ranked.begin() + take);
    std::vector<uint32_t> out;
    out.reserve(take);
    for (size_t index = 0; index < take; ++index)
      out.push_back(ranked[index].second);
    return out;
  }

  std::vector<uint32_t> route(const std::shared_ptr<const Directory> &cur,
                              const float *q, size_t probes,
                              bool force_linear = false) const {
    size_t n = cur->pages.size();
    last_search_metrics.directory_pages = n;
    probes = std::min(probes, n);
    if (cur->centroids.size() != n * dimensions)
      throw std::runtime_error("invalid routing centroid snapshot");
    // The second level is deliberately not used for query routing. Placement
    // only needs a page near the vector, so approximating the candidate set is
    // nearly free there. A query needs the best pages, and gathering from
    // groups measured 5% slower with slightly worse recall than scanning the
    // quantised centroids outright: the gather and its dedupe cost more than
    // the scan they save.
    if (force_linear || n <= ROUTING_GRAPH_THRESHOLD || probes * 8 >= n) {
      std::vector<std::pair<float, uint32_t>> ranked;
      ranked.reserve(n);
      // The quantised scan is ~3.9x faster per centroid but adds a widened
      // sort and an exact rescore of the shortlist. Those fixed costs only
      // amortise once the scan dominates routing: measured at 61% of routing
      // for ~1.4k pages (where it is a net loss) and 82% for ~7k pages.
      const bool quantised = cur->centroid_codes.size() == n * ((dimensions + 1) / 2) &&
                             cur->centroid_peak.size() == n &&
                             n >= ROUTING_QUANTISED_THRESHOLD;
      if (quantised) {
        // Rank on 4-bit nibble codes, then rescore a widened shortlist exactly.
        // Shortlist membership is approximate; the probed pages are ordered by
        // true distance after the rescore step.
        float query_peak = 0.0f;
        for (size_t j = 0; j < dimensions; ++j)
          query_peak = std::max(query_peak, std::fabs(q[j]));
        std::vector<int8_t> query_code(dimensions);
        const float to_code = query_peak > 0.0f ? 127.0f / query_peak : 0.0f;
        for (size_t j = 0; j < dimensions; ++j)
          query_code[j] =
              int8_t(std::clamp(std::lround(q[j] * to_code), -127l, 127l));
        // <q,c> ~ int4_dot * query_peak * peak / (127 * 7)
        const float inner_scale = query_peak / (127.0f * 7.0f);
        const size_t cc_stride = (dimensions + 1) / 2;
        for (size_t index = 0; index < n; ++index) {
          const float inner =
              float(int4_dot(query_code.data(),
                             cur->centroid_codes.data() + index * cc_stride,
                             dimensions)) *
              inner_scale * cur->centroid_peak[index];
          // Rank by a value monotone in true distance for this metric; the
          // query-only term is dropped because it is shared by every page.
          const float score = metric == CV_COSINE
                                  ? -inner
                                  : cur->centroid_norm2[index] - 2.0f * inner;
          ranked.push_back({score, uint32_t(index)});
          ++last_search_metrics.centroid_scores;
        }
        // Measured on 1M SIFT over 520 queries: the smallest quantised prefix
        // containing the entire exact top-P set never exceeded 1.12x P. 1.25x
        // with a floor keeps margin without rescoring pages that cannot matter.
        const size_t shortlist =
            std::min(n, std::max(probes + probes / 4, probes + 16));
        // nth_element is O(n) where partial_sort is O(n log k); the shortlist
        // does not need to be ordered, only selected, because it is rescored
        // exactly and re-sorted below.
        std::nth_element(ranked.begin(), ranked.begin() + shortlist,
                         ranked.end());
        ranked.resize(shortlist);
        for (auto &item : ranked)
          item.first =
              distance(q, cur->centroids.data() + item.second * dimensions);
      } else {
        for (size_t index = 0; index < n; ++index) {
          ranked.push_back(
              {distance(q, cur->centroids.data() + index * dimensions),
               uint32_t(index)});
          ++last_search_metrics.centroid_scores;
        }
      }
      probes = std::min(probes, ranked.size());
      // Select then sort: the k smallest are found in O(n), and only those k
      // are ordered. Output is identical to partial_sort.
      if (probes < ranked.size())
        std::nth_element(ranked.begin(), ranked.begin() + probes, ranked.end());
      std::sort(ranked.begin(), ranked.begin() + probes);
      std::vector<uint32_t> out;
      for (size_t index = 0; index < probes; ++index)
        out.push_back(ranked[index].second);
      return out;
    }
    using Item = std::pair<float, uint32_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> frontier;
    std::vector<uint8_t> seen(n);
    for (size_t seed : {size_t(0), n / 4, n / 2, n * 3 / 4})
      if (!seen[seed]) {
        seen[seed] = 1;
        frontier.push({distance(q, cur->centroids.data() + seed * dimensions),
                       uint32_t(seed)});
        ++last_search_metrics.centroid_scores;
      }
    size_t limit = std::min(n, std::max<size_t>(64, probes * 6));
    std::vector<Item> evaluated;
    while (!frontier.empty() && evaluated.size() < limit) {
      auto item = frontier.top();
      frontier.pop();
      evaluated.push_back(item);
      for (uint32_t neighbor : cur->neighbors[item.second])
        if (!seen[neighbor]) {
          seen[neighbor] = 1;
          frontier.push({distance(q, cur->centroids.data() +
                                         size_t(neighbor) * dimensions),
                         neighbor});
          ++last_search_metrics.centroid_scores;
        }
    }
    size_t take = std::min(probes, evaluated.size());
    std::partial_sort(evaluated.begin(), evaluated.begin() + take,
                      evaluated.end());
    std::vector<uint32_t> out;
    for (size_t index = 0; index < take; ++index)
      out.push_back(evaluated[index].second);
    return out;
  }
};

extern "C" {
cv_index *cv_create_with_options(size_t d, int metric, size_t cap,
                                 size_t probes, uint32_t flags,
                                 size_t rerank_factor) {
  try {
    if (!d)
      throw std::invalid_argument("dimensions must be positive");
    if (cap < MIN_PAGE_CAPACITY)
      throw std::invalid_argument(
          "page_capacity must be at least " + std::to_string(MIN_PAGE_CAPACITY));
    if (metric != CV_COSINE && metric != CV_L2)
      throw std::invalid_argument("metric must be CV_COSINE or CV_L2");
    return new cv_index(d, metric, cap, probes, flags, rerank_factor);
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}
static int cv_insert_impl(cv_index *i, int64_t id, const float *input,
                          uint64_t label, uint64_t timestamp,
                          uint64_t *committed);
cv_index *cv_create_disk_backed(size_t d, int metric, size_t cap, size_t probes,
                                uint32_t flags, size_t rerank_factor,
                                const char *arena_path, size_t max_vectors);
cv_index *cv_create_with_wal(size_t d, int metric, size_t cap, size_t probes,
                             uint32_t flags, size_t rerank_factor,
                             const char *wal_path, int sync_on_commit,
                             const char *arena_path, size_t max_vectors) {
  try {
    if (!wal_path || !*wal_path)
      throw std::invalid_argument("wal path is required");
    // The two can be combined, and the reason is worth stating: the arena is a
    // spill file, not a record. It holds whatever the pages currently need and
    // is thrown away on close. The log is the authority for what the index
    // contains. So recovery makes a fresh arena and replays the log into it --
    // the arena is rebuilt as a side effect of redoing the writes, and nothing
    // has to survive in it across a crash.
    cv_index *raw =
        (arena_path && *arena_path)
            ? cv_create_disk_backed(d, metric, cap, probes, flags,
                                    rerank_factor, arena_path, max_vectors)
            : cv_create_with_options(d, metric, cap, probes, flags,
                                     rerank_factor);
    if (!raw)
      return nullptr;
    std::unique_ptr<cv_index> index(raw);
    // Replay first, then open for appending, so recovery cannot be confused by
    // records this process is about to write.
    size_t good = 0;
    auto records = replay_wal(wal_path, 0, &good);
    for (const auto &record : records) {
      if (record.op == WAL_INSERT) {
        if (record.vector.size() != d)
          throw std::runtime_error("write-ahead log has a different width");
        if (cv_insert_impl(index.get(), record.id, record.vector.data(), 0,
                           record.ts, nullptr) != 0)
          throw std::runtime_error("replay failed: " + last_error);
      } else if (record.op == WAL_DELETE) {
        std::lock_guard lock(index->writer_mutex);
        index->tick(record.ts);
        index->close_current(record.id, record.ts, false);
        index->commit(record.ts);
      }
    }
    // Drop a torn tail: a partly written record describes a change that never
    // committed, and leaving it would make the next replay stop early for ever.
    if (std::filesystem::exists(wal_path) &&
        std::filesystem::file_size(wal_path) > good)
      std::filesystem::resize_file(wal_path, good);
    index->wal = std::make_unique<Wal>(wal_path, sync_on_commit != 0);
    return index.release();
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}
cv_index *cv_create_disk_backed(size_t d, int metric, size_t cap, size_t probes,
                                uint32_t flags, size_t rerank_factor,
                                const char *arena_path, size_t max_vectors) {
  try {
    if (!d)
      throw std::invalid_argument("dimensions must be positive");
    if (cap < MIN_PAGE_CAPACITY)
      throw std::invalid_argument(
          "page_capacity must be at least " + std::to_string(MIN_PAGE_CAPACITY));
    if (metric != CV_COSINE && metric != CV_L2)
      throw std::invalid_argument("metric must be CV_COSINE or CV_L2");
    if (!arena_path || !*arena_path)
      throw std::invalid_argument("arena path is required");
    if (!max_vectors)
      throw std::invalid_argument("max_vectors must be positive");
    // max_vectors is a live-vector budget, so the arena must provision for more
    // than max_vectors/capacity pages: pages do not fill completely (measured
    // capacity amplification runs 1.3x to 2.2x), copy-on-write holds two blocks
    // for a page while a clone is published, and split briefly holds three. The
    // file is sparse and the mapping is virtual, so over-provisioning costs
    // address space rather than disk.
    const size_t pages = (max_vectors + cap - 1) / cap;
    const size_t blocks = pages * 4 + 64;
    auto index = std::make_unique<cv_index>(d, metric, cap, probes, flags,
                                            rerank_factor);
    index->arena = std::make_shared<VectorArena>(
        std::filesystem::path(arena_path), cap * d * sizeof(float), blocks);
    // A second mapping for the screening codes. Vectors are 80% of the payload
    // and screening the other 20%, so mapping only the vectors left a fifth of
    // a large index pinned in memory. Its own file and block size, because the
    // stride is different; the arena is a fixed-block allocator.
    if (flags & CV_ENABLE_INT8_SCREENING) {
      // Mirrors how the index itself derives residual coding, so the block
      // size matches the stride the pages will actually use.
      const bool residual = (metric == CV_L2 || d >= 64);
      const size_t stride =
          2 * sizeof(float) + (residual ? (d + 1) / 2 : d);
      // More blocks than the vector arena needs. A clone shares the vectors
      // with the page it came from but takes a private copy of the codes
      // whenever a rebind rewrites every slot, so record blocks are acquired
      // and released far more often than vector blocks and the peak in flight
      // is higher.
      index->record_arena = std::make_shared<VectorArena>(
          std::filesystem::path(std::string(arena_path) + ".codes"),
          cap * stride, blocks * 2);
    }
    return index.release();
  } catch (const std::exception &e) {
    last_error = e.what();
    return nullptr;
  }
}
size_t cv_flush_vectors(cv_index *i) {
  if (!i || !i->arena) {
    last_error = "index is not disk backed";
    return 0;
  }
  // Both mappings: the vectors and, when screening is on, the codes beside
  // them. Reporting only the vectors understated what the index had made
  // evictable by the size of the codes arena.
  size_t total = i->arena->make_evictable();
  if (i->record_arena)
    total += i->record_arena->make_evictable();
  return total;
}
cv_index *cv_create(size_t d, int metric, size_t cap, size_t probes) {
  return cv_create_with_options(d, metric, cap, probes,
                                CV_ENABLE_INT8_SCREENING, 4);
}
void cv_destroy(cv_index *i) { delete i; }
static int cv_insert_impl(cv_index *i, int64_t id, const float *input,
                          uint64_t label, uint64_t timestamp,
                          uint64_t *committed) {
  try {
    if (!i)
      throw std::invalid_argument("null index");
    std::lock_guard lock(i->writer_mutex);
    auto v = i->prepare(input);
    uint64_t ts = i->tick(timestamp);
    i->close_current(id, ts, false);
    if (i->wal) {
      i->wal->stage(WAL_INSERT, ts, id, v.data(), uint32_t(i->dimensions));
      i->wal->flush();      // durable before the change is applied
    }
    auto d = i->choose_page(v.data(), label);
    auto before = d->image.load();
    auto after = std::make_shared<Page>(*before);
    size_t s = after->free_slot();
    after->ids[s] = id;
    after->begin[s] = ts;
    after->end[s] = MAX_TS;
    after->occupied[s] = 1;
    after->filled.push_back(uint32_t(s));
    std::copy(v.begin(), v.end(), after->vec(s));
    if (!after->labels.empty())
      after->labels[s] = label;
    after->encode(s);
    after->apply_delta(s, true, i->metric);
    i->publish_page(d, before, after);
    i->locations[id].push_back({d, s});
    i->commit(ts);
    if (committed)
      *committed = ts;
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}
int cv_insert(cv_index *i, int64_t id, const float *input, uint64_t timestamp,
              uint64_t *committed) {
  return cv_insert_impl(i, id, input, 0, timestamp, committed);
}
// Phase timings for the batched insert path, off unless CHRONOVEC_PROFILE_BATCH
// is set. Bulk ingestion has repeatedly been optimised against a guess about
// where its time goes and repeatedly proved the guess wrong, so the breakdown
// is measured in place rather than reasoned about.
struct BatchProfile {
  bool on = false;
  double prepare = 0, closes = 0, assign = 0, sort = 0, apply = 0, over = 0;
  size_t rows = 0;
  BatchProfile() {
    const char *e = std::getenv("CHRONOVEC_PROFILE_BATCH");
    on = e && *e && *e != '0';
  }
  ~BatchProfile() {
    if (!on || !rows)
      return;
    const double n = double(rows);
    std::fprintf(stderr,
                 "[batch] rows=%zu prepare=%.2f closes=%.2f assign=%.2f "
                 "sort=%.2f apply=%.2f overflow=%.2f (us/row)\n",
                 rows, prepare / n, closes / n, assign / n, sort / n,
                 apply / n, over / n);
  }
};
static BatchProfile batch_profile;
struct PhaseTimer {
  double *sink;
  std::chrono::steady_clock::time_point t0;
  explicit PhaseTimer(double *s)
      : sink(batch_profile.on ? s : nullptr),
        t0(std::chrono::steady_clock::now()) {}
  ~PhaseTimer() {
    if (sink)
      *sink += std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
  }
};

size_t cv_insert_batch_labeled(cv_index *i, const int64_t *ids,
                               const float *vectors, const uint64_t *labels,
                               size_t count, uint64_t *committed) {
  if (!i || (!ids && count) || (!vectors && count)) {
    last_error = "null argument";
    return 0;
  }
  if (labels && !i->labels_enabled) {
    last_error = "index was not created with CV_ENABLE_LABELS";
    return 0;
  }
  try {
    std::lock_guard lock(i->writer_mutex);
    const size_t d = i->dimensions;
    auto label_of = [&](size_t row) -> uint64_t {
      return labels ? labels[row] : 0;
    };
    size_t inserted = 0;
    // One timestamp for the whole call, committed once at the end. Every row
    // becomes visible at the same instant, so a reader sees the batch entire
    // or not at all. With per-row timestamps a reader sampling the clock
    // mid-batch saw a few hundred rows of four thousand.
    const uint64_t batch_ts =
        count ? i->tick(0) : i->committed.load(std::memory_order_acquire);
    uint64_t last = 0;

    // Appending one row to a page the caller already holds open, without going
    // back through choose_page or publishing.
    auto fill = [&](Page &page, size_t slot, int64_t id, const float *v,
                    uint64_t ts, uint64_t label) {
      page.ids[slot] = id;
      page.begin[slot] = ts;
      page.end[slot] = MAX_TS;
      page.occupied[slot] = 1;
      page.filled.push_back(uint32_t(slot));
      std::copy_n(v, d, page.vec(slot));
      if (!page.labels.empty()) {
        page.labels[slot] = label;
        // The page's union is what lets a filtered query skip it whole, so it
        // has to grow with every label the page accepts.
        page.label_union |= label;
      }
      page.encode(slot);
      page.apply_delta(slot, true, i->metric);
    };
    // The escape hatch for rows the batched path cannot take: it may split, so
    // it runs only once no page is held open. It must not call cv_insert_impl,
    // which takes the writer mutex this function already holds.
    auto place_one = [&](int64_t id, const float *v, uint64_t ts,
                         uint64_t label) {
      auto desc = i->choose_page(v, label);
      auto before = desc->image.load();
      auto after = std::make_shared<Page>(*before);
      const size_t slot = after->free_slot();
      if (slot >= i->page_capacity)
        return false;
      fill(*after, slot, id, v, ts, label);
      i->publish_page(desc, before, after);
      i->locations[id].push_back({desc, slot});
      return true;
    };

    // Blocked so the assignment buffers stay bounded for large batches and each
    // block assigns against a directory that reflects the previous block.
    constexpr size_t kBlock = 8192;
    std::vector<float> prepared;
    std::vector<uint64_t> stamps;
    std::vector<uint32_t> target, order, overflow;

    for (size_t base = 0; base < count && inserted == base; base += kBlock) {
      const size_t n = std::min(kBlock, count - base);
      prepared.resize(n * d);
      {
        PhaseTimer t(&batch_profile.prepare);
        for (size_t r = 0; r < n; ++r) {
          auto v = i->prepare(vectors + (base + r) * d);
          std::copy(v.begin(), v.end(), prepared.begin() + r * d);
        }
      }
      batch_profile.rows += n;
      // The whole block is logged and made durable before any of it is
      // applied, so recovery replays a block entire or not at all -- the same
      // guarantee the single commit timestamp gives a live reader.
      if (i->wal) {
        for (size_t r = 0; r < n; ++r)
          i->wal->stage(WAL_INSERT, batch_ts, ids[base + r],
                        prepared.data() + r * d, uint32_t(d));
        i->wal->flush();
      }

      auto dir = i->directory.load();
      // A repeated id inside one block would need its own earlier version
      // closed, which the up-front close pass cannot see. Rare enough that the
      // block is simply placed row by row instead of special-cased.
      bool duplicates = false;
      {
        std::unordered_set<int64_t> seen;
        seen.reserve(n * 2);
        for (size_t r = 0; r < n && !duplicates; ++r)
          duplicates = !seen.insert(ids[base + r]).second;
      }
      if (duplicates || dir->pages.empty() ||
          dir->centroids.size() != dir->pages.size() * d) {
        for (size_t r = 0; r < n; ++r) {
          auto existing = i->locations.find(ids[base + r]);
          if (existing != i->locations.end() && !existing->second.empty())
            i->close_current(ids[base + r], batch_ts, false);
          if (!place_one(ids[base + r], prepared.data() + r * d, batch_ts,
                         label_of(base + r)))
            break;
          last = batch_ts;
          ++inserted;
        }
        continue;
      }

      // Close replaced versions before anything is cloned, so no page is
      // mutated underneath a clone taken from it.
      stamps.assign(n, batch_ts);
      {
        PhaseTimer t(&batch_profile.closes);
        for (size_t r = 0; r < n; ++r) {
          auto existing = i->locations.find(ids[base + r]);
          if (existing != i->locations.end() && !existing->second.empty())
            i->close_current(ids[base + r], batch_ts, false);
        }
      }

      // Under partitioning a row may only land in a page carrying its label,
      // so every label in this block needs one before the rounds start. One
      // representative row per new label goes through the single-row path,
      // which creates the page; the rest of the block then assigns normally.
      // Without this the assignment would have to fall back per row, which is
      // what made a partitioned bulk insert 35x slower than an ordinary one.
      if (i->label_partition) {
        std::unordered_set<uint64_t> seeded;
        for (size_t r = 0; r < n; ++r) {
          const uint64_t label = label_of(base + r);
          if (!label || !seeded.insert(label).second)
            continue;
          auto rdir = i->directory.load();
          bool present = false;
          for (const auto &page : rdir->pages)
            if (page->image.load()->label_union == label) {
              present = true;
              break;
            }
          if (!present)
            i->choose_page(prepared.data() + r * d, label);
        }
      }

      // Rows are placed in rounds. A round assigns every row still pending with
      // one matmul, fills the pages it can, then splits the pages that ran out
      // of room. Rows that did not fit are pending for the next round and get a
      // matmul-assigned placement against the widened directory rather than an
      // individual centroid scan.
      //
      // This is where the bulk cost actually was. Routing every overflow row
      // through choose_page cost 1.69us of a 2.89us insert, more than
      // assignment, prepare, cloning and publishing combined, because a young
      // index overflows constantly while its pages are still filling.
      std::vector<uint32_t> pending(n);
      for (size_t r = 0; r < n; ++r)
        pending[r] = uint32_t(r);
      std::vector<float> rowbuf;
      std::vector<uint32_t> full_pages;
      // Every round either places a row or splits a page, so it makes progress;
      // the cap only stops a pathological directory from looping.
      constexpr size_t kMaxRounds = 32;
      for (size_t round = 0; round < kMaxRounds && !pending.empty(); ++round) {
        auto rdir = round == 0 ? dir : i->directory.load();
        if (rdir->pages.empty() ||
            rdir->centroids.size() != rdir->pages.size() * d)
          break;
        const size_t pn = pending.size();
        rowbuf.resize(pn * d);
        for (size_t k = 0; k < pn; ++k)
          std::copy_n(prepared.data() + size_t(pending[k]) * d, d,
                      rowbuf.data() + k * d);
        target.resize(pn);
        std::vector<uint64_t> row_labels, page_labels;
        if (i->label_partition) {
          row_labels.resize(pn);
          for (size_t k = 0; k < pn; ++k)
            row_labels[k] = label_of(base + pending[k]);
          page_labels.resize(rdir->pages.size());
          for (size_t c = 0; c < rdir->pages.size(); ++c)
            page_labels[c] = rdir->pages[c]->image.load()->label_union;
        }
        {
          PhaseTimer t(&batch_profile.assign);
          i->assign_block(*rdir, rowbuf.data(), pn, target.data(),
                          row_labels.empty() ? nullptr : row_labels.data(),
                          page_labels.empty() ? nullptr : page_labels.data());
        }
        order.resize(pn);
        for (size_t k = 0; k < pn; ++k)
          order[k] = uint32_t(k);
        {
          PhaseTimer t(&batch_profile.sort);
          std::stable_sort(order.begin(), order.end(),
                           [&](uint32_t a, uint32_t b) {
                             return target[a] < target[b];
                           });
        }

        overflow.clear();
        full_pages.clear();
        // Bucket boundaries first, so the work is a list of independent units.
        struct Bucket { uint32_t page; uint32_t from, to; };
        std::vector<Bucket> buckets;
        for (size_t pos = 0; pos < pn;) {
          const uint32_t page = target[order[pos]];
          size_t run = pos;
          while (run < pn && target[order[run]] == page)
            ++run;
          buckets.push_back({page, uint32_t(pos), uint32_t(run)});
          pos = run;
        }
        {
          PhaseTimer t(&batch_profile.apply);
          // Each bucket owns one page, and publishing is a compare-exchange on
          // that page's descriptor, so buckets do not interact -- except
          // through `locations`, which is one hash map for the whole index.
          // Threads collect their location updates and they are merged after
          // the join rather than locked per row.
          struct Local {
            std::vector<std::pair<int64_t, Location>> placed;
            std::vector<uint32_t> overflow, full;
            uint64_t last = 0;
            size_t inserted = 0;
            std::string error;
          };
          const size_t lanes =
              std::min<size_t>(std::max<size_t>(i->threads, 1),
                               std::max<size_t>(buckets.size(), 1));
          std::vector<Local> locals(lanes);
          auto run_lane = [&](size_t lane) {
            Local &out = locals[lane];
            try {
              for (size_t b = lane; b < buckets.size(); b += lanes) {
                const Bucket &bucket = buckets[b];
                auto desc = rdir->pages[bucket.page];
                auto before = desc->image.load();
                auto after = std::make_shared<Page>(*before);
                bool dirty = false, spilled = false;
                for (size_t k = bucket.from; k < bucket.to; ++k) {
                  const uint32_t r = pending[order[k]];
                  const size_t slot = after->free_slot();
                  if (slot >= i->page_capacity) {
                    out.overflow.push_back(r);
                    spilled = true;
                    continue;
                  }
                  fill(*after, slot, ids[base + r], prepared.data() + r * d,
                       stamps[r], label_of(base + r));
                  out.placed.push_back({ids[base + r], Location{desc, slot}});
                  dirty = true;
                  out.last = std::max(out.last, stamps[r]);
                  ++out.inserted;
                }
                if (dirty)
                  i->publish_page(desc, before, after);
                if (spilled)
                  out.full.push_back(bucket.page);
              }
            } catch (const std::exception &error) {
              out.error = error.what();
            }
          };
          if (lanes <= 1) {
            run_lane(0);
          } else {
            std::vector<std::thread> workers;
            workers.reserve(lanes - 1);
            for (size_t lane = 1; lane < lanes; ++lane)
              workers.emplace_back(run_lane, lane);
            run_lane(0);
            for (auto &worker : workers)
              worker.join();
          }
          for (Local &out : locals) {
            if (!out.error.empty())
              throw std::runtime_error(out.error);
            for (auto &[id, where] : out.placed)
              i->locations[id].push_back(where);
            overflow.insert(overflow.end(), out.overflow.begin(),
                            out.overflow.end());
            full_pages.insert(full_pages.end(), out.full.begin(),
                              out.full.end());
            last = std::max(last, out.last);
            inserted += out.inserted;
          }
          // Both lists are ordered before they are used, so the index built
          // does not depend on how the lanes happened to divide the work.
          // Overflow decides placement order; full_pages decides split order,
          // and leaving that one unsorted made the page count differ by lane
          // count -- same answers, different structure, which is still a
          // reproducibility bug.
          std::sort(overflow.begin(), overflow.end());
          std::sort(full_pages.begin(), full_pages.end());
        }

        if (overflow.empty()) {
          // Everything still pending was placed this round. Clearing matters:
          // the tail loop below places whatever `pending` holds, and leaving
          // the just-placed rows there inserts every one of them twice.
          pending.clear();
          break;
        }
        // Split every page that spilled, once, before the next round. A split
        // republishes the directory, so it runs only after every clone from
        // this round has been published.
        {
          PhaseTimer t(&batch_profile.over);
          for (uint32_t page : full_pages) {
            try {
              i->split(rdir->pages[page]);
            } catch (const std::exception &) {
              // A page with fewer than two live rows cannot be split; its rows
              // fall through to the single-row path below.
            }
          }
        }
        pending = overflow;
      }

      // Anything still unplaced after the round cap takes the single-row path,
      // which can always make room.
      for (uint32_t r : pending) {
        if (!place_one(ids[base + r], prepared.data() + r * d, stamps[r],
                       label_of(base + r)))
          break;
        last = std::max(last, stamps[r]);
        ++inserted;
      }
    }

    // Consolidate what the splits left behind.
    //
    // Splitting is how a batch makes room, and each split leaves two
    // half-empty pages. Without this the debris survived until someone called
    // vacuum, which for a bulk load is usually never: amplification 13.86 at
    // d=768 on a freshly built index against 1.22 after one pass. Charging it
    // to the batch that created it makes the default path the correct one.
    //
    // Only for batches large enough to have split much. A small insert into a
    // warm index should not pay for a directory scan.
    if (i->auto_consolidate && inserted >= i->page_capacity)
      i->consolidate(MAX_MERGES_PER_VACUUM);

    // Everything is published; now let readers see it.
    i->commit(batch_ts);
    if (committed)
      *committed = last;
    return inserted;
  } catch (const std::exception &error) {
    last_error = error.what();
    return 0;
  }
}
size_t cv_insert_batch(cv_index *i, const int64_t *ids, const float *vectors,
                       size_t count, uint64_t *committed) {
  return cv_insert_batch_labeled(i, ids, vectors, nullptr, count, committed);
}
size_t cv_delete_batch(cv_index *i, const int64_t *ids, size_t count,
                       uint64_t *committed) {
  if (!i || (!ids && count)) {
    last_error = "null argument";
    return 0;
  }
  // Deliberately row by row. Taking the writer mutex once for the whole batch
  // was measured at 0.122s per 100k either way -- an uncontended lock is tens
  // of nanoseconds and a delete is a page publish -- and it would hold the
  // mutex for the full batch, stalling concurrent single writers on the
  // streaming path for no gain.
  size_t removed = 0;
  uint64_t last = 0;
  for (size_t row = 0; row < count; ++row) {
    uint64_t stamp = 0;
    if (cv_delete(i, ids[row], 0, &stamp) != 0)
      break;
    last = stamp;
    ++removed;
  }
  if (committed)
    *committed = last;
  return removed;
}
int cv_apply_changes(cv_index *i, const int64_t *delete_ids,
                     size_t delete_count, const int64_t *upsert_ids,
                     const float *upsert_vectors, size_t upsert_count,
                     uint64_t *committed) {
  if (!i || (!delete_ids && delete_count) || (!upsert_ids && upsert_count) ||
      (!upsert_vectors && upsert_count)) {
    last_error = "null argument";
    return -1;
  }
  try {
    std::lock_guard lock(i->writer_mutex);
    const size_t d = i->dimensions;
    // Validate and normalize before reserving a timestamp. Everything after
    // tick is a single publication unit; a validation error therefore cannot
    // leave an invisible half-transaction behind.
    std::unordered_set<int64_t> deletes;
    deletes.reserve(delete_count * 2);
    for (size_t row = 0; row < delete_count; ++row) {
      if (!deletes.insert(delete_ids[row]).second)
        throw std::invalid_argument("duplicate delete id in apply_changes");
      // Must be checked here, not by close_current's own `required` throw in
      // the publish loop below: that loop runs after tick() and mutates pages
      // as it goes, so a missing id discovered mid-loop would leave earlier
      // rows in this same call published-but-uncommitted -- invisible for now,
      // but not undone, and silently exposed by any later unrelated commit.
      if (!i->has_open_version(delete_ids[row]))
        throw std::out_of_range("id not found");
    }
    std::unordered_set<int64_t> upserts;
    upserts.reserve(upsert_count * 2);
    std::vector<float> prepared(upsert_count * d);
    for (size_t row = 0; row < upsert_count; ++row) {
      if (!upserts.insert(upsert_ids[row]).second)
        throw std::invalid_argument("duplicate upsert id in apply_changes");
      if (deletes.count(upsert_ids[row]))
        throw std::invalid_argument("id cannot be deleted and upserted in one apply_changes");
      auto value = i->prepare(upsert_vectors + row * d);
      std::copy(value.begin(), value.end(), prepared.begin() + row * d);
    }
    if (!delete_count && !upsert_count) {
      if (committed)
        *committed = i->committed.load(std::memory_order_acquire);
      return 0;
    }

    const uint64_t ts = i->tick(0);
    if (i->wal) {
      for (size_t row = 0; row < delete_count; ++row)
        i->wal->stage(WAL_DELETE, ts, delete_ids[row], nullptr, 0);
      for (size_t row = 0; row < upsert_count; ++row)
        i->wal->stage(WAL_INSERT, ts, upsert_ids[row], prepared.data() + row * d,
                      uint32_t(d));
      i->wal->flush();
    }
    for (size_t row = 0; row < delete_count; ++row)
      i->close_current(delete_ids[row], ts, true);
    for (size_t row = 0; row < upsert_count; ++row) {
      const int64_t id = upsert_ids[row];
      const float *value = prepared.data() + row * d;
      i->close_current(id, ts, false);
      auto page = i->choose_page(value, 0);
      auto before = page->image.load();
      auto after = std::make_shared<Page>(*before);
      const size_t slot = after->free_slot();
      if (slot >= i->page_capacity)
        throw std::runtime_error("selected page has no free slot");
      after->ids[slot] = id;
      after->begin[slot] = ts;
      after->end[slot] = MAX_TS;
      after->occupied[slot] = 1;
      after->filled.push_back(uint32_t(slot));
      std::copy_n(value, d, after->vec(slot));
      after->encode(slot);
      after->apply_delta(slot, true, i->metric);
      i->publish_page(page, before, after);
      i->locations[id].push_back({page, slot});
    }
    i->commit(ts);
    if (committed)
      *committed = ts;
    return 0;
  } catch (const std::exception &error) {
    last_error = error.what();
    return -1;
  }
}
int cv_insert_labeled(cv_index *i, int64_t id, const float *input,
                      uint64_t label, uint64_t timestamp, uint64_t *committed) {
  if (i && !i->labels_enabled) {
    last_error = "index was not created with CV_ENABLE_LABELS";
    return -1;
  }
  return cv_insert_impl(i, id, input, label, timestamp, committed);
}
int cv_delete(cv_index *i, int64_t id, uint64_t timestamp,
              uint64_t *committed) {
  try {
    if (!i)
      throw std::invalid_argument("null index");
    std::lock_guard lock(i->writer_mutex);
    uint64_t ts = i->tick(timestamp);
    if (i->wal) {
      i->wal->stage(WAL_DELETE, ts, id, nullptr, 0);
      i->wal->flush();
    }
    i->close_current(id, ts, true);
    i->commit(ts);
    if (committed)
      *committed = ts;
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
    return -1;
  }
}
namespace {
// A record passes when it carries every required bit, at least one of any
// optional bits, and none of the excluded bits.
inline bool label_matches(uint64_t label, const cv_filter &f) {
  return (label & f.require_all) == f.require_all &&
         (f.require_any == 0 || (label & f.require_any) != 0) &&
         (label & f.exclude) == 0;
}
// A page can be skipped outright when the union of its labels cannot satisfy
// the required bits, or contains nothing the caller asked for.
inline bool page_can_match(uint64_t union_bits, const cv_filter &f) {
  return (union_bits & f.require_all) == f.require_all &&
         (f.require_any == 0 || (union_bits & f.require_any) != 0);
}
} // namespace

// One pass at a fixed probe width. `cv_search_impl` wraps this to widen the
// probe set when a filter leaves the result short.
static size_t cv_search_pass(cv_index *i, const float *input, size_t k,
                             uint64_t snapshot, size_t probes,
                             uint32_t search_flags, const cv_filter *filter,
                             int64_t *out_ids, float *out_d,
                             const std::vector<uint32_t> *explicit_pages =
                                 nullptr) {
  try {
    using SearchClock = std::chrono::steady_clock;
    auto total_started = SearchClock::now();
    if (!k)
      return 0;
    if (!i || !out_ids || !out_d)
      throw std::invalid_argument("invalid search arguments");
    if ((search_flags & CV_SEARCH_ADAPTIVE) && !i->adaptive_bounds)
      throw std::invalid_argument(
          "adaptive search requires CV_ENABLE_ADAPTIVE_BOUNDS at creation");
    // Pin before resolving `snap`: see ReaderRegistry for why the order
    // matters. Covers the whole pass, so every page this call touches is
    // protected against a concurrent vacuum reclaiming a slot out from
    // under it.
    ReaderPin pin(i->reader_registry);
    auto q = i->prepare(input);
    uint64_t snap = snapshot ? snapshot : i->committed.load(std::memory_order_acquire);
    pin.refine(snap);
    auto dir = i->directory.load();
    if (dir->pages.empty())
      return 0;
    auto routing_started = SearchClock::now();
    last_search_metrics.preparation_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(routing_started -
                                                             total_started)
            .count();
    // An explicit page list bypasses routing entirely. Routing ranks pages by
    // distance to the query, which is the right question only when the answer
    // can come from anywhere; when the caller has already determined which
    // pages can hold a match, ranking them is wasted work and truncating that
    // ranking is what loses recall.
    auto routed = explicit_pages
                      ? *explicit_pages
                      : i->route(dir, q.data(),
                                 probes ? probes : i->default_nprobe,
                                 search_flags & CV_SEARCH_LINEAR_ROUTING);
    auto routing_finished = SearchClock::now();
    last_search_metrics.routed_pages = routed.size();
    last_search_metrics.routing_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(routing_finished -
                                                             routing_started)
            .count();
    std::priority_queue<Candidate, std::vector<Candidate>, Worse> best;
    const size_t estimated = routed.size() * i->page_capacity;
    // Scalar/short scans beat sketch setup. Enable screening only once the
    // estimated exact-score work crosses the measured break-even region.
    // Residual coding is metric-agnostic, so screening is no longer restricted
    // to cosine; L2 was previously scanning every candidate in float32.
    const bool use_screening =
        i->screening && estimated * i->dimensions > 384000;
    if (use_screening) {
      std::vector<int8_t> query_code(i->dimensions);
      std::vector<float> query_residual(i->dimensions);
      // The exact-rescore shortlist. Screening keeps this many candidates by
      // estimated distance and scores only those exactly, so a true neighbour
      // the estimator ranks outside it is lost however many pages were probed.
      //
      // Fixed, not scaled with the candidate count, and that was tested rather
      // than assumed. Scaling it looked necessary on clustered synthetic data,
      // where recall plateaued at 0.943 across 16, 32 and 64 probes. On real
      // SIFT it is not: recall is monotone in nprobe at a fixed 80 -- 0.637,
      // 0.775, 0.891, 0.957, 0.986, 0.992 -- and scaling the shortlist cost
      // 1.5x at recall 0.99 for no recall it did not already reach.
      //
      // The synthetic pathology is real but is a property of that data: tight
      // Gaussian clusters make within-cluster distances nearly equal, so the
      // estimator's ranking is noisy relative to the gaps it has to resolve.
      // `rerank_factor` is the knob for data that behaves that way, and
      // `exact_scores` in the search metrics is how to tell that it does.
      const size_t rerank = std::max<size_t>(k * i->rerank_factor, 64);
      std::priority_queue<Screened, std::vector<Screened>, BetterScreen>
          screened;
      std::vector<std::shared_ptr<const Page>> images;
      images.reserve(routed.size());
      for (uint32_t page_index : routed)
        images.push_back(dir->pages[page_index]->image.load());
      if (!i->residual_codes)
        for (size_t j = 0; j < i->dimensions; ++j)
          query_code[j] =
              int8_t(std::clamp(std::lround(q[j] * 127.0f), -127l, 127l));
      // The branch is hoisted out of the candidate loop: the global-sketch
      // path must stay exactly as cheap as it was before residual coding
      // existed, since it is chosen precisely where per-candidate work is
      // already minimal.
      auto consider = [&](float estimate, uint32_t page, uint32_t slot) {
        Screened item{estimate, page, slot};
        if (screened.size() < rerank)
          screened.push(item);
        else if (estimate < screened.top().estimate) {
          screened.pop();
          screened.push(item);
        }
      };
      for (uint32_t page_index = 0; page_index < images.size(); ++page_index) {
        auto &p = images[page_index];
        if (filter && !p->labels.empty() &&
            !page_can_match(p->label_union, *filter)) {
          ++last_search_metrics.bound_pruned_pages;
          continue;
        }
        if (i->residual_codes) {
          // The query residual is taken against this page's pinned
          // code_centroid, matching how its codes were built.
          float query_squared = 0.0f;
          for (size_t j = 0; j < i->dimensions; ++j) {
            const float r = q[j] - p->code_centroid[j];
            query_residual[j] = r;
            query_squared += r * r;
          }
          const float query_norm = std::sqrt(query_squared);
          float query_peak = 0.0f;
          for (size_t j = 0; j < i->dimensions; ++j)
            query_peak = std::max(query_peak, std::fabs(query_residual[j]));
          // <c,u_q> = int4_dot * (query_unit_peak / 127)
          const float query_scale =
              query_norm > 0.0f ? query_peak / (query_norm * 127.0f) : 0.0f;
          const float inverse = query_peak > 0.0f ? 127.0f / query_peak : 0.0f;
          for (size_t j = 0; j < i->dimensions; ++j)
            query_code[j] = int8_t(std::clamp(
                std::lround(query_residual[j] * inverse), -127l, 127l));
          for (uint32_t slot : p->filled) {
            ++last_search_metrics.physical_candidates;
            if (!p->visible(slot, snap))
              continue;
            if (filter && !p->labels.empty() &&
                !label_matches(p->labels[slot], *filter))
              continue;
            ++last_search_metrics.visible_candidates;
            ++last_search_metrics.sketch_scores;
            const float *scalars = p->record_scalars(slot);
            const float inner =
                float(int4_dot(query_code.data(), p->record_codes(slot),
                               i->dimensions)) *
                query_scale * scalars[1];
            const float record_norm = scalars[0];
            // Estimated ||q-v||^2, monotone in cosine distance too since
            // ||q-v||^2 = 2*(1-<q,v>) for unit vectors.
            consider(query_squared + record_norm * record_norm -
                         2.0f * query_norm * record_norm * inner,
                     page_index, slot);
          }
        } else {
          for (uint32_t slot : p->filled) {
            ++last_search_metrics.physical_candidates;
            if (!p->visible(slot, snap))
              continue;
            if (filter && !p->labels.empty() &&
                !label_matches(p->labels[slot], *filter))
              continue;
            ++last_search_metrics.visible_candidates;
            ++last_search_metrics.sketch_scores;
            // Negated similarity so both paths share one min-is-better heap.
            consider(-float(int8_dot(query_code.data(),
                                     reinterpret_cast<const int8_t *>(
                                         p->record_codes(slot)),
                                     i->dimensions)),
                     page_index, slot);
          }
        }
      }
      auto screening_finished = SearchClock::now();
      last_search_metrics.screening_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              screening_finished - routing_finished)
              .count();
      while (!screened.empty()) {
        auto item = screened.top();
        screened.pop();
        auto &page = images[item.page];
        ++last_search_metrics.exact_scores;
        Candidate candidate{i->distance(q.data(), page->vec(item.slot)),
                            page->ids[item.slot]};
        if (best.size() < k)
          best.push(candidate);
        else if (candidate.distance < best.top().distance) {
          best.pop();
          best.push(candidate);
        }
      }
      auto rerank_finished = SearchClock::now();
      last_search_metrics.rerank_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              rerank_finished - screening_finished)
              .count();
    } else {
      for (uint32_t pi : routed) {
        auto p = dir->pages[pi]->image.load();
        if (filter && !p->labels.empty() &&
            !page_can_match(p->label_union, *filter)) {
          ++last_search_metrics.bound_pruned_pages;
          continue;
        }
        if ((search_flags & CV_SEARCH_ADAPTIVE) && best.size() == k &&
            i->page_lower_bound(q.data(), *p) >= best.top().distance) {
          ++last_search_metrics.bound_pruned_pages;
          continue;
        }
        for (uint32_t s : p->filled) {
          ++last_search_metrics.physical_candidates;
          if (p->visible(s, snap)) {
            if (filter && !p->labels.empty() &&
                !label_matches(p->labels[s], *filter))
              continue;
            ++last_search_metrics.visible_candidates;
            ++last_search_metrics.exact_scores;
            Candidate candidate{i->distance(q.data(), p->vec(s)), p->ids[s]};
            if (best.size() < k)
              best.push(candidate);
            else if (candidate.distance < best.top().distance) {
              best.pop();
              best.push(candidate);
            }
          }
        }
      }
      auto exact_finished = SearchClock::now();
      last_search_metrics.rerank_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(exact_finished -
                                                               routing_finished)
              .count();
    }
    last_search_metrics.mvcc_filtered =
        last_search_metrics.physical_candidates -
        last_search_metrics.visible_candidates;
    auto materialization_started = SearchClock::now();
    size_t count = best.size();
    last_search_metrics.result_count = count;
    for (size_t x = count; x-- > 0;) {
      out_ids[x] = best.top().id;
      out_d[x] = i->metric == CV_L2 ? std::sqrt(best.top().distance)
                                    : best.top().distance;
      best.pop();
    }
    auto finished = SearchClock::now();
    last_search_metrics.result_materialization_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            finished - materialization_started)
            .count();
    last_search_metrics.total_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished -
                                                             total_started)
            .count();
    return count;
  } catch (const std::exception &e) {
    last_error = e.what();
    return 0;
  }
}
// A filtered search cannot use a fixed probe width the way an unfiltered one
// can. Routing ranks pages by distance to the query, which says nothing about
// where the matching records are: when the filtered attribute correlates with
// position in the embedding space -- one tenant, one language, one product
// line, which is the normal case rather than a corner one -- the matching
// records sit in pages that the query's own neighbourhood does not reach. At
// 1% selectivity on SIFT this returned 1.6 of 10 requested neighbours at
// nprobe 32 and needed 512 to fill k, and it did so silently: short results
// look like a fast query, not a wrong one.
//
// So the probe set is widened until k matching records are found or the
// directory is exhausted. This runs only when a filter is present and the
// pass came up short, so an unfiltered search is bit-for-bit what it was, and
// a filtered search that fills k on the first pass pays one comparison. The
// repeated passes redo work rather than resuming, which costs at most about
// twice the final pass because the widths double -- worth it against
// returning the wrong answer, and the alternative is threading a resumable
// routing cursor through a function that four execution flows depend on.
size_t cv_search_impl(cv_index *i, const float *input, size_t k,
                      uint64_t snapshot, size_t probes, uint32_t search_flags,
                      const cv_filter *filter, int64_t *out_ids,
                      float *out_d) {
  // Reset here rather than inside the pass. Selecting the pages a filter
  // admits happens before the pass and records work of its own, and a reset
  // inside the pass wiped it: centroid_scores read zero for every filtered
  // query however many centroids had been ranked.
  last_search_metrics = {};
  size_t width = probes ? probes : (i ? i->default_nprobe : 0);
  if (i && filter) {
    // Routing works on the prepared query, the same as inside a pass.
    const auto filter_started = std::chrono::steady_clock::now();
    auto q_for_route = i->prepare(input);
    // Which pages can hold a match is answerable from the label unions alone,
    // with one bitmask test per page and no distance arithmetic. When that
    // set is small the whole filtered query is answered exactly by scanning
    // it, and the routing step -- which is what truncates the answer -- is
    // never run.
    //
    // This is the case the structure exists for, and it is also the realistic
    // one: an attribute worth filtering on (a tenant, an owner, a language, a
    // product line) correlates with position in the embedding space, so its
    // records occupy few pages. A scattered attribute saturates every page's
    // union, the test rules nothing out, and the code below falls through to
    // routing, which is the right behaviour for it -- there the filter is a
    // cheap per-record rejection ahead of the distance, not a way to skip
    // pages.
    auto dir = i->directory.load();
    if (dir && !dir->pages.empty() &&
        dir->centroids.size() == dir->pages.size() * i->dimensions) {
      std::vector<uint32_t> matching;
      bool labelled = false;
      for (uint32_t index = 0; index < dir->pages.size(); ++index) {
        auto image = dir->pages[index]->image.load();
        if (image->labels.empty())
          continue;
        labelled = true;
        if (page_can_match(image->label_union, *filter))
          matching.push_back(index);
      }
      // Only worth it while the test actually rules pages out. Once most of
      // the directory can match, the filter is not page-selective, ordinary
      // routing is both cheaper and sufficient -- the per-record rejection
      // ahead of the distance is what pays there -- and this path would just
      // be a slower way to reach the same pages.
      if (labelled && !matching.empty() &&
          matching.size() <= dir->pages.size() / 2) {
        size_t reach = std::min(width ? width : matching.size(),
                                matching.size());
        auto chosen = i->route_filtered(dir, q_for_route.data(), reach,
                                        matching);
        // Selecting the eligible pages happens before the pass and would
        // otherwise be invisible: it is real query time, it grows with the
        // directory, and leaving it out of the metrics is how it went
        // unnoticed while it grew.
        const uint64_t selection_ns =
            uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - filter_started)
                         .count());
        size_t hit = cv_search_pass(i, input, k, snapshot, width, search_flags,
                                    filter, out_ids, out_d, &chosen);
        // Widening here means probing more of the matching pages, not more of
        // the directory, so it terminates at the matching set rather than at
        // the index.
        while (hit < k && reach < matching.size()) {
          reach = reach > matching.size() / 2 ? matching.size() : reach * 2;
          chosen = i->route_filtered(dir, q_for_route.data(), reach, matching);
          hit = cv_search_pass(i, input, k, snapshot, width, search_flags,
                               filter, out_ids, out_d, &chosen);
        }
        last_search_metrics.routing_ns += selection_ns;
        last_search_metrics.total_ns += selection_ns;
        return hit;
      }
    }
  }
  size_t found = cv_search_pass(i, input, k, snapshot, width, search_flags,
                                filter, out_ids, out_d);
  if (!filter || found >= k || !i)
    return found;
  auto dir = i->directory.load();
  const size_t pages = dir ? dir->pages.size() : 0;
  if (!width || width >= pages)
    return found;
  // Widening stops at the directory size: past that there is nothing left to
  // route, and a filter matching fewer than k records legitimately returns
  // fewer than k.
  while (width < pages && found < k) {
    width = width > pages / 2 ? pages : width * 2;
    found = cv_search_pass(i, input, k, snapshot, width, search_flags, filter,
                           out_ids, out_d);
  }
  return found;
}

size_t cv_search_with_options(cv_index *i, const float *input, size_t k,
                              uint64_t snapshot, size_t probes,
                              uint32_t search_flags, int64_t *out_ids,
                              float *out_d) {
  return cv_search_impl(i, input, k, snapshot, probes, search_flags, nullptr,
                        out_ids, out_d);
}
size_t cv_search_filtered(cv_index *i, const float *input, size_t k,
                          uint64_t snapshot, size_t probes,
                          uint32_t search_flags, const cv_filter *filter,
                          int64_t *out_ids, float *out_d) {
  return cv_search_impl(i, input, k, snapshot, probes, search_flags, filter,
                        out_ids, out_d);
}
size_t cv_search(cv_index *i, const float *input, size_t k, uint64_t snapshot,
                 size_t probes, int64_t *out_ids, float *out_d) {
  return cv_search_with_options(i, input, k, snapshot, probes, 0, out_ids,
                                out_d);
}
size_t cv_search_batch(cv_index *i, const float *queries, size_t count,
                       size_t k, uint64_t snapshot, size_t nprobe,
                       uint32_t search_flags, int64_t *out_ids,
                       float *out_distances, size_t *out_found) {
  if (!i || (!queries && count) || (!out_ids && count) ||
      (!out_distances && count)) {
    last_error = "null argument";
    return 0;
  }
  // One crossing for the whole set rather than one per query. The queries are
  // answered independently and in order, so a snapshot covers all of them and
  // the batch is a consistent read of one point in time.
  //
  // Deliberately not parallel: readers are lock-free, so a caller that wants
  // cores can run several of these at once, and doing it here would take that
  // choice away and oversubscribe a caller that already has.
  size_t answered = 0;
  for (size_t row = 0; row < count; ++row) {
    const size_t found = cv_search_with_options(
        i, queries + row * i->dimensions, k, snapshot, nprobe, search_flags,
        out_ids + row * k, out_distances + row * k);
    if (out_found)
      out_found[row] = found;
    // Pad the tail so a caller can read the block without consulting
    // out_found: -1 is the empty-slot marker the index already uses.
    for (size_t slot = found; slot < k; ++slot) {
      out_ids[row * k + slot] = -1;
      out_distances[row * k + slot] = std::numeric_limits<float>::infinity();
    }
    answered += found;
  }
  return answered;
}
size_t cv_exact_search_f32(const float *vectors, const int64_t *ids,
                           size_t count, size_t dimensions, int metric,
                           const float *query, size_t k, int64_t *out_ids,
                           float *out_distances) {
  try {
    if (!k)
      return 0;
    if ((!vectors && count) || !query || !out_ids || !out_distances ||
        !dimensions || (metric != CV_COSINE && metric != CV_L2))
      throw std::invalid_argument("invalid exact scan arguments");
    std::priority_queue<Candidate, std::vector<Candidate>, Worse> best;
    for (size_t row = 0; row < count; ++row) {
      const float *vector = vectors + row * dimensions;
      float value = metric == CV_COSINE ? 1.0f - dot(query, vector, dimensions)
                                        : l2sq(query, vector, dimensions);
      Candidate candidate{value, ids ? ids[row] : int64_t(row)};
      if (best.size() < k)
        best.push(candidate);
      else if (value < best.top().distance) {
        best.pop();
        best.push(candidate);
      }
    }
    size_t result_count = best.size();
    for (size_t position = result_count; position-- > 0;) {
      out_ids[position] = best.top().id;
      out_distances[position] = metric == CV_L2 ? std::sqrt(best.top().distance)
                                                : best.top().distance;
      best.pop();
    }
    return result_count;
  } catch (const std::exception &error) {
    last_error = error.what();
    return 0;
  }
}
struct VacuumProfile {
  bool on = false;
  double scan = 0, pages = 0, locations = 0, merges = 0;
  size_t calls = 0, reclaimed = 0;
  VacuumProfile() {
    const char *e = std::getenv("CHRONOVEC_PROFILE_VACUUM");
    on = e && *e && *e != '0';
  }
  ~VacuumProfile() {
    if (!on || !calls)
      return;
    std::fprintf(stderr,
                 "[vacuum] calls=%zu reclaimed=%zu scan=%.1f pages=%.1f "
                 "locations=%.1f merges=%.1f (ms total)\n",
                 calls, reclaimed, scan, pages, locations, merges);
  }
};
static VacuumProfile vacuum_profile;
struct VacuumTimer {
  double *sink;
  std::chrono::steady_clock::time_point t0;
  explicit VacuumTimer(double *s)
      : sink(vacuum_profile.on ? s : nullptr),
        t0(std::chrono::steady_clock::now()) {}
  ~VacuumTimer() {
    if (sink)
      *sink += std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
  }
};
size_t cv_vacuum(cv_index *i, uint64_t oldest, size_t budget) {
  if (!i) {
    last_error = "null index";
    return 0;
  }
  try {
    std::lock_guard lock(i->writer_mutex);
    // Never reclaim past what an in-flight search might still need, however
    // aggressive the caller's `oldest` is (the documented idiom is
    // clock()+1, i.e. "everything eligible right now") -- see
    // ReaderRegistry.
    const uint64_t safe_oldest = i->reader_registry.safe_oldest(oldest);
    size_t count = 0;
    std::unordered_map<Descriptor *,
                       std::pair<std::shared_ptr<Descriptor>,
                                 std::vector<std::pair<uint64_t, uint64_t>>>>
        remove;
    std::optional<VacuumTimer> stage;
    stage.emplace(&vacuum_profile.scan);
    while (!i->retired.empty() && i->retired.front().end < safe_oldest &&
           (!budget || count < budget)) {
      auto r = i->retired.front();
      i->retired.pop_front();
      auto f = i->locations.find(r.id);
      if (f == i->locations.end())
        continue;
      for (auto &l : f->second) {
        auto p = l.page->image.load();
        if (p->occupied[l.slot] && p->begin[l.slot] == r.begin &&
            p->end[l.slot] == r.end) {
          auto &entry = remove[l.page.get()];
          entry.first = l.page;
          entry.second.push_back({r.begin, r.end});
          ++count;
          break;
        }
      }
    }
    stage.emplace(&vacuum_profile.pages);
    for (auto &[raw, entry] : remove) {
      // The descriptor is already held, so no directory scan is needed here.
      const std::shared_ptr<Descriptor> &d = entry.first;
      const auto &versions = entry.second;
      (void)raw;
      if (!d)
        continue;
      auto before = d->image.load();
      auto after = std::make_shared<Page>(*before);
      for (size_t s = 0; s < after->capacity; ++s)
        if (after->occupied[s])
          for (auto [begin, end] : versions)
            if (after->begin[s] == begin && after->end[s] == end) {
              after->occupied[s] = 0;
              after->ids[s] = -1;
              break;
            }
      after->recompute(i->metric);
      i->publish_page(d, before, after);
    }
    stage.emplace(&vacuum_profile.locations);
    if (count) {
      for (auto it = i->locations.begin(); it != i->locations.end();) {
        auto &v = it->second;
        v.erase(std::remove_if(v.begin(), v.end(),
                               [](auto &l) {
                                 return !l.page->image.load()->occupied[l.slot];
                               }),
                v.end());
        if (v.empty())
          it = i->locations.erase(it);
        else
          ++it;
      }
      i->reclaimed += count;
    }
    // Consolidate in proportion to the space just freed. Merging exactly once
    // per call left bulk churn at 4x capacity amplification: pages fill with
    // dead slots, so at a fixed probe budget each probe reaches half as many
    // live records, which surfaces as recall drift across turnovers rather than
    // as a space problem. One merge reclaims roughly half a page of holes.
    stage.emplace(&vacuum_profile.merges);
    vacuum_profile.calls += 1;
    vacuum_profile.reclaimed += count;
    // Consolidation is the same control law the bulk path uses; see
    // cv_index::consolidate.
    i->consolidate(MAX_MERGES_PER_VACUUM);
    return count;
  } catch (const std::exception &error) {
    last_error = error.what();
    return 0;
  }
}
int cv_save_checkpoint(cv_index *i, const char *path) {
  try {
    if (!i || !path || !*path)
      throw std::invalid_argument("invalid checkpoint path");
    std::lock_guard lock(i->writer_mutex);
    auto directory = i->directory.load();
    std::vector<uint8_t> output;
    output.reserve(256 + directory->pages.size() * i->page_capacity *
                             (sizeof(int64_t) + 2 * sizeof(uint64_t) +
                              i->dimensions * sizeof(float)));
    output.insert(output.end(), CHECKPOINT_MAGIC, CHECKPOINT_MAGIC + 8);
    append_value(output, uint64_t(i->dimensions));
    append_value(output, uint64_t(i->metric));
    append_value(output, uint64_t(i->page_capacity));
    append_value(output, uint64_t(i->default_nprobe));
    append_value(
        output, uint64_t((i->screening ? CV_ENABLE_INT8_SCREENING : 0) |
                         (i->adaptive_bounds ? CV_ENABLE_ADAPTIVE_BOUNDS : 0) |
                         (i->labels_enabled ? CV_ENABLE_LABELS : 0) |
                         (i->label_partition ? CV_ENABLE_LABEL_PARTITION : 0)));
    append_value(output, uint64_t(i->rerank_factor));
    append_value(output, i->clock.load(std::memory_order_acquire));
    append_value(output, i->next_page_id);
    append_value(output, i->splits);
    append_value(output, i->merges);
    append_value(output, i->reclaimed);
    append_value(output, i->routing_full_rebuilds);
    append_value(output, i->routing_incremental_updates);
    append_value(output, uint64_t(directory->pages.size()));
    for (size_t page_index = 0; page_index < directory->pages.size();
         ++page_index) {
      const auto &descriptor = directory->pages[page_index];
      auto page = descriptor->image.load();
      append_value(output, descriptor->id);
      append_value(output, uint64_t(page->filled.size()));
      for (uint32_t slot : page->filled) {
        append_value(output, page->ids[slot]);
        append_value(output, page->begin[slot]);
        append_value(output, page->end[slot]);
        const auto *bytes = reinterpret_cast<const uint8_t *>(page->vec(slot));
        output.insert(output.end(), bytes,
                      bytes + i->dimensions * sizeof(float));
        if (i->labels_enabled)
          append_value(output, uint64_t(page->labels[slot]));
      }
      const auto &edges = page_index < directory->neighbors.size()
                              ? directory->neighbors[page_index]
                              : std::vector<uint32_t>{};
      append_value(output, uint64_t(edges.size()));
      for (uint32_t edge : edges)
        append_value(output, edge);
    }
    append_value(output, checkpoint_hash(output.data(), output.size()));

    std::filesystem::path destination(path);
#if defined(_WIN32)
    const int process_id = _getpid();
#else
    const int process_id = getpid();
#endif
    std::filesystem::path temporary =
        destination.string() + ".tmp." + std::to_string(process_id);
#if defined(_WIN32)
    int descriptor = _open(temporary.string().c_str(),
                           _O_CREAT | _O_TRUNC | _O_WRONLY | _O_BINARY,
                           _S_IREAD | _S_IWRITE);
#else
    int descriptor =
        ::open(temporary.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
#endif
    if (descriptor < 0)
      throw std::runtime_error("cannot open checkpoint temporary file");
    try {
      write_all(descriptor, output.data(), output.size());
#if defined(_WIN32)
      if (_commit(descriptor) != 0)
#else
      if (fsync(descriptor) != 0)
#endif
        throw std::runtime_error("checkpoint sync failed");
#if defined(_WIN32)
      _close(descriptor);
#else
      close(descriptor);
#endif
      descriptor = -1;
#if defined(_WIN32)
      if (!MoveFileExW(temporary.c_str(), destination.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("checkpoint atomic replacement failed");
#else
      std::filesystem::rename(temporary, destination);
#endif
#if !defined(_WIN32)
      auto parent = destination.parent_path();
      if (parent.empty())
        parent = ".";
      int directory_fd = ::open(parent.c_str(), O_RDONLY);
      if (directory_fd >= 0) {
        fsync(directory_fd);
        close(directory_fd);
      }
#endif
    } catch (...) {
      if (descriptor >= 0) {
#if defined(_WIN32)
        _close(descriptor);
#else
        close(descriptor);
#endif
      }
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      throw;
    }
    return 0;
  } catch (const std::exception &error) {
    last_error = error.what();
    return -1;
  }
}

cv_index *cv_load_checkpoint(const char *path) {
  try {
    if (!path || !*path)
      throw std::invalid_argument("invalid checkpoint path");
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
      throw std::runtime_error("cannot open checkpoint");
    auto length = stream.tellg();
    if (length < std::streamoff(8 + sizeof(uint64_t)))
      throw std::runtime_error("truncated checkpoint");
    std::vector<uint8_t> input(static_cast<size_t>(length), uint8_t{0});
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char *>(input.data()), length))
      throw std::runtime_error("checkpoint read failed");
    uint64_t stored_hash;
    std::memcpy(&stored_hash, input.data() + input.size() - sizeof(uint64_t),
                sizeof(uint64_t));
    if (stored_hash !=
        checkpoint_hash(input.data(), input.size() - sizeof(uint64_t)))
      throw std::runtime_error("checkpoint checksum mismatch");
    size_t position = 0;
    if (std::memcmp(input.data(), CHECKPOINT_MAGIC, 8) != 0)
      throw std::runtime_error("unsupported checkpoint format");
    position += 8;
    uint64_t dimensions = read_value<uint64_t>(input, position);
    uint64_t metric = read_value<uint64_t>(input, position);
    uint64_t capacity = read_value<uint64_t>(input, position);
    uint64_t probes = read_value<uint64_t>(input, position);
    uint64_t flags = read_value<uint64_t>(input, position);
    uint64_t rerank = read_value<uint64_t>(input, position);
    if (!dimensions || dimensions > CHECKPOINT_MAX_DIMENSIONS || capacity < 8 ||
        capacity > UINT32_MAX || (metric != CV_COSINE && metric != CV_L2))
      throw std::runtime_error("invalid checkpoint configuration");
    auto result = std::make_unique<cv_index>(dimensions, int(metric), capacity,
                                             probes, uint32_t(flags), rerank);
    {
      const uint64_t restored = read_value<uint64_t>(input, position);
      result->clock.store(restored);
      result->committed.store(restored);
    }
    result->next_page_id = read_value<uint64_t>(input, position);
    result->splits = read_value<uint64_t>(input, position);
    result->merges = read_value<uint64_t>(input, position);
    result->reclaimed = read_value<uint64_t>(input, position);
    result->routing_full_rebuilds = read_value<uint64_t>(input, position);
    result->routing_incremental_updates = read_value<uint64_t>(input, position);
    uint64_t page_count = read_value<uint64_t>(input, position);
    if (page_count > CHECKPOINT_MAX_PAGES)
      throw std::runtime_error("invalid checkpoint page count");
    Directory restored;
    restored.pages.reserve(size_t(page_count));
    restored.neighbors.resize(size_t(page_count));
    std::vector<Retired> retired;
    for (size_t page_index = 0; page_index < page_count; ++page_index) {
      uint64_t descriptor_id = read_value<uint64_t>(input, position);
      uint64_t record_count = read_value<uint64_t>(input, position);
      if (record_count > capacity)
        throw std::runtime_error("invalid checkpoint page occupancy");
      auto page = std::make_shared<Page>(
          capacity, dimensions, flags & CV_ENABLE_INT8_SCREENING,
          flags & CV_ENABLE_ADAPTIVE_BOUNDS,
          metric == CV_L2 || dimensions >= 64, flags & CV_ENABLE_LABELS);
      auto descriptor = std::make_shared<Descriptor>(descriptor_id, page);
      for (size_t slot = 0; slot < record_count; ++slot) {
        page->ids[slot] = read_value<int64_t>(input, position);
        page->begin[slot] = read_value<uint64_t>(input, position);
        page->end[slot] = read_value<uint64_t>(input, position);
        size_t vector_bytes = dimensions * sizeof(float);
        if (position > input.size() || input.size() - position < vector_bytes)
          throw std::runtime_error("truncated checkpoint vector");
        std::memcpy(page->vec(slot), input.data() + position, vector_bytes);
        position += vector_bytes;
        if (flags & CV_ENABLE_LABELS)
          page->labels[slot] = read_value<uint64_t>(input, position);
        page->occupied[slot] = 1;
        page->filled.push_back(uint32_t(slot));
        result->locations[page->ids[slot]].push_back({descriptor, slot});
        if (page->end[slot] != MAX_TS)
          retired.push_back(
              {page->end[slot], page->begin[slot], page->ids[slot]});
      }
      page->recompute(int(metric));
      restored.pages.push_back(descriptor);
      uint64_t edge_count = read_value<uint64_t>(input, position);
      if (edge_count > page_count)
        throw std::runtime_error("invalid checkpoint routing degree");
      auto &edges = restored.neighbors[page_index];
      edges.reserve(size_t(edge_count));
      for (size_t edge = 0; edge < edge_count; ++edge) {
        uint32_t target = read_value<uint32_t>(input, position);
        if (target >= page_count)
          throw std::runtime_error("invalid checkpoint routing edge");
        edges.push_back(target);
      }
    }
    if (position != input.size() - sizeof(uint64_t))
      throw std::runtime_error("checkpoint has trailing data");
    std::sort(retired.begin(), retired.end(),
              [](const Retired &left, const Retired &right) {
                return left.end < right.end;
              });
    result->retired.assign(retired.begin(), retired.end());
    result->refresh_directory_centroids(restored);
    result->directory.store(
        std::make_shared<const Directory>(std::move(restored)));
    return result.release();
  } catch (const std::exception &error) {
    last_error = error.what();
    return nullptr;
  }
}

cv_version_t cv_version(void) {
  return {CV_ABI_VERSION_MAJOR, CV_ABI_VERSION_MINOR, CV_ABI_VERSION_PATCH, 0};
}

size_t cv_dimensions(const cv_index *i) { return i ? i->dimensions : 0; }
int cv_metric(const cv_index *i) { return i ? i->metric : -1; }
size_t cv_page_capacity(const cv_index *i) { return i ? i->page_capacity : 0; }
size_t cv_page_label_profile(const cv_index *i, const cv_filter *filter,
                             uint64_t snapshot, uint32_t *out_live,
                             uint32_t *out_matching, size_t max_pages) {
  if (!i)
    return 0;
  auto dir = i->directory.load();
  if (!dir)
    return 0;
  const uint64_t snap =
      snapshot ? snapshot : i->committed.load(std::memory_order_acquire);
  const size_t n = dir->pages.size();
  const size_t limit = std::min(n, max_pages);
  for (size_t index = 0; index < limit; ++index) {
    auto page = dir->pages[index]->image.load();
    uint32_t live = 0, hit = 0;
    for (uint32_t slot : page->filled) {
      if (!page->visible(slot, snap))
        continue;
      ++live;
      if (!filter || page->labels.empty() ||
          label_matches(page->labels[slot], *filter))
        ++hit;
    }
    if (out_live)
      out_live[index] = live;
    if (out_matching)
      out_matching[index] = hit;
  }
  return n;
}
size_t cv_default_nprobe(const cv_index *i) {
  return i ? i->default_nprobe : 0;
}
uint32_t cv_flags(const cv_index *i) {
  return i ? uint32_t((i->screening ? CV_ENABLE_INT8_SCREENING : 0) |
                      (i->adaptive_bounds ? CV_ENABLE_ADAPTIVE_BOUNDS : 0) |
                      (i->labels_enabled ? CV_ENABLE_LABELS : 0) |
                      (i->label_partition ? CV_ENABLE_LABEL_PARTITION : 0))
           : 0;
}
size_t cv_rerank_factor(const cv_index *i) { return i ? i->rerank_factor : 0; }
void cv_set_auto_consolidate(cv_index *i, int enabled) {
  if (i)
    i->auto_consolidate = enabled != 0;
}
void cv_set_threads(cv_index *i, size_t lanes) {
  if (i)
    i->threads = lanes ? lanes : 1;
}
uint64_t cv_clock(const cv_index *i) {
  // The committed clock, because this is what callers turn into snapshots. An
  // allocated-but-uncommitted timestamp is not a point anyone may read at.
  return i ? i->committed.load(std::memory_order_acquire) : 0;
}
int cv_get_stats(const cv_index *i, cv_stats *out) {
  if (!i || !out)
    return -1;
  auto dir = i->directory.load();
  uint64_t live = 0, retained = 0;
  uint64_t edge_count = 0;
  for (auto &d : dir->pages) {
    auto p = d->image.load();
    for (uint32_t s : p->filled) {
      if (p->visible(s, MAX_TS - 1))
        ++live;
      else
        ++retained;
    }
  }
  for (const auto &edges : dir->neighbors)
    edge_count += edges.size();
  *out = {i->committed.load(std::memory_order_acquire),
          dir->pages.size(),
          live,
          retained,
          dir->pages.size() * i->page_capacity,
          i->splits,
          i->reclaimed,
          i->merges,
          i->routing_full_rebuilds,
          i->routing_incremental_updates,
          dir->centroids.size() * sizeof(float),
          edge_count};
  return 0;
}
int cv_get_last_search_metrics(cv_search_metrics *out) {
  if (!out)
    return -1;
  *out = last_search_metrics;
  return 0;
}
const char *cv_last_error() { return last_error.c_str(); }
}
