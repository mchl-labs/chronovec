#ifndef CHRONOVEC_H
#define CHRONOVEC_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define CV_API __declspec(dllexport)
#else
#define CV_API __attribute__((visibility("default")))
#endif

/*
 * Semantic version of this C ABI.
 *
 * MAJOR increments on any breaking change (struct layout, removed symbol,
 * changed semantics). MINOR increments on additive-only changes (new symbols,
 * new flag bits in existing enums). PATCH for bug-fix releases with no ABI
 * change.
 *
 * Callers that need a runtime check:
 *   cv_version_t v = cv_version();
 *   assert(v.major == CV_ABI_VERSION_MAJOR);
 */
#define CV_ABI_VERSION_MAJOR 0
#define CV_ABI_VERSION_MINOR 1
#define CV_ABI_VERSION_PATCH 0
#define CV_ABI_VERSION \
  ((CV_ABI_VERSION_MAJOR << 16) | (CV_ABI_VERSION_MINOR << 8) | CV_ABI_VERSION_PATCH)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cv_index cv_index;

typedef struct cv_version_t {
  uint8_t major;
  uint8_t minor;
  uint8_t patch;
  uint8_t reserved;
} cv_version_t;

/* Returns the ABI version of the loaded shared library. Compare major against
   CV_ABI_VERSION_MAJOR to detect incompatible builds. */
CV_API cv_version_t cv_version(void);

/*
 * Threading contract: searches are lock-free over immutable snapshots.
 * Mutation/checkpoint/vacuum calls are thread-safe but serialize per index.
 * This API does not claim lock-free or parallel multiwriter progress.
 */

typedef struct cv_stats {
  uint64_t clock;
  uint64_t pages;
  uint64_t live_vectors;
  uint64_t retained_versions;
  uint64_t allocated_slots;
  uint64_t splits;
  uint64_t reclaimed_versions;
  uint64_t merges;
  uint64_t routing_full_rebuilds;
  uint64_t routing_incremental_updates;
  uint64_t routing_centroid_bytes;
  uint64_t routing_edge_count;
} cv_stats;

typedef struct cv_search_metrics {
  uint64_t directory_pages;
  uint64_t centroid_scores;
  uint64_t routed_pages;
  uint64_t physical_candidates;
  uint64_t visible_candidates;
  uint64_t mvcc_filtered;
  uint64_t sketch_scores;
  uint64_t exact_scores;
  uint64_t result_count;
  uint64_t bound_pruned_pages;
  uint64_t preparation_ns;
  uint64_t routing_ns;
  uint64_t screening_ns;
  uint64_t rerank_ns;
  uint64_t result_materialization_ns;
  uint64_t total_ns;
} cv_search_metrics;

enum { CV_COSINE = 0, CV_L2 = 1 };
enum {
  CV_ENABLE_INT8_SCREENING = 1,
  CV_ENABLE_ADAPTIVE_BOUNDS = 2,
  /* Reserve a 64-bit label per record so searches can filter on attributes.
     Costs 8 bytes per vector and nothing when no filter is supplied. */
  CV_ENABLE_LABELS = 4,
  /* Keep one label to a page: a page holds records of a single label value,
     and placement takes the nearest page already holding that label rather
     than the nearest page overall. Requires CV_ENABLE_LABELS.

     This is the trade a filtered workload wants and an unfiltered one does
     not. Mixed pages follow the vector geometry exactly, but a selective
     filter then finds its records spread thinly across them: at 1% of a 500k
     corpus the matching records occupied 485 pages where 20 would hold them,
     so a query probed 24x the pages to gather the same candidates. Pure pages
     cost geometry and page count, and buy that back -- measured at 3.4x on
     that workload against an index holding only the matching records, which
     is the ceiling for this. */
  CV_ENABLE_LABEL_PARTITION = 8
};

/* Attribute filter over the 64-bit label attached to each record.
   A record matches when
     (label & require_all) == require_all
     && (require_any == 0 || (label & require_any) != 0)
     && (label & exclude) == 0
   Pages track the union of their labels, so a page that cannot contain a match
   is skipped whole. That page-level pruning is why a partition index filters
   without the recall collapse graph indexes suffer at low selectivity. */
typedef struct cv_filter {
  uint64_t require_all;
  uint64_t require_any;
  uint64_t exclude;
} cv_filter;
enum { CV_SEARCH_ADAPTIVE = 1, CV_SEARCH_LINEAR_ROUTING = 2 };

CV_API cv_index *cv_create(size_t dimensions, int metric, size_t page_capacity,
                           size_t nprobe);
CV_API cv_index *cv_create_with_options(size_t dimensions, int metric,
                                        size_t page_capacity, size_t nprobe,
                                        uint32_t flags, size_t rerank_factor);
/* Keep the authoritative float vectors in a file-backed mapping instead of on
   the heap, so the operating system can evict cold ones. BETA.

   `max_vectors` fixes the mapping size up front; block pointers therefore never
   move. Codes and routing stay resident, and a query reads the mapping for only
   about rerank_factor*k records.

   An insert no longer rewrites the page: clones share the float payload, so a
   single vector is dirtied rather than the whole block. Sustained mutation is
   therefore viable, unlike the first version of this mode. Split, merge and
   vacuum still write whole pages, so heavy structural churn remains I/O bound.
   Not available on Windows. */
/* Create an index backed by a write-ahead log at `wal_path`.
 *
 * If the file exists its records are replayed first, so this is both "create"
 * and "recover". Records are appended before a change is applied; with
 * `sync_on_commit` non-zero each commit is fsynced, which is what makes a
 * power cut survivable and is also the slower setting.
 *
 * A torn tail left by a crash mid-append is detected by CRC, dropped, and the
 * file truncated: a partly written record describes a change that never
 * committed.
 *
 * Pass `arena_path` and `max_vectors` to combine durability with a
 * memory-mapped index. The arena is a spill file rather than a record: it is
 * recreated empty and refilled by the replay, since the log is what says what
 * the index contains. Pass NULL and 0 for a heap-resident index. */
/* Whether a large batch insert consolidates the pages its splits created.
 * On by default. Turning it off makes a bulk load faster and leaves the
 * directory holding every page a split produced until vacuum runs; on uniform
 * high-dimensional data that is roughly fourteen times the memory. */
CV_API void cv_set_auto_consolidate(cv_index *index, int enabled);

/* Lanes used by the batched insert's page-apply phase. One by default.
 * Writers are serialised overall, so this parallelises one phase, not the
 * write path. */
CV_API void cv_set_threads(cv_index *index, size_t lanes);

CV_API cv_index *cv_create_with_wal(size_t dimensions, int metric,
                                    size_t page_capacity, size_t nprobe,
                                    uint32_t flags, size_t rerank_factor,
                                    const char *wal_path, int sync_on_commit,
                                    const char *arena_path,
                                    size_t max_vectors);

CV_API cv_index *cv_create_disk_backed(size_t dimensions, int metric,
                                       size_t page_capacity, size_t nprobe,
                                       uint32_t flags, size_t rerank_factor,
                                       const char *arena_path,
                                       size_t max_vectors);
/* Flush the vector arena so its pages are clean and the kernel may evict them
   under memory pressure. Returns bytes made evictable, or 0 if the index is not
   disk backed. Eviction is the kernel's decision, so resident memory does not
   necessarily fall at once. BETA. */
CV_API size_t cv_flush_vectors(cv_index *index);
CV_API void cv_destroy(cv_index *index);
CV_API int cv_insert(cv_index *index, int64_t id, const float *vector,
                     uint64_t timestamp, uint64_t *committed_timestamp);
CV_API int cv_delete(cv_index *index, int64_t id, uint64_t timestamp,
                     uint64_t *committed_timestamp);
CV_API size_t cv_search(cv_index *index, const float *query, size_t k,
                        uint64_t snapshot, size_t nprobe, int64_t *out_ids,
                        float *out_distances);
CV_API size_t cv_search_with_options(cv_index *index, const float *query,
                                     size_t k, uint64_t snapshot, size_t nprobe,
                                     uint32_t search_flags, int64_t *out_ids,
                                     float *out_distances);
/* Answer `count` queries in one call.
 *
 * `out_ids` and `out_distances` are `count * k` long and each row is padded
 * with -1 and infinity when fewer than k results are found; `out_found`, if
 * given, receives the count per row. Identical results to calling cv_search
 * once per query -- the saving is the per-call crossing, which measured 7.9us
 * of a 61.2us query from Python.
 *
 * Queries are answered independently and in order, so a snapshot applies to
 * all of them and the batch is a consistent read. */
/* Insert many vectors in one call, each carrying an attribute label.
 *
 * `labels` is `count` long, or NULL for unlabelled rows. Requires the index to
 * have been created with CV_ENABLE_LABELS. Labels are what let a filtered
 * query skip a page whole rather than testing its records one at a time. */
CV_API size_t cv_insert_batch_labeled(cv_index *index, const int64_t *ids,
                                      const float *vectors,
                                      const uint64_t *labels, size_t count,
                                      uint64_t *committed);

CV_API size_t cv_search_batch(cv_index *index, const float *queries,
                              size_t count, size_t k, uint64_t snapshot,
                              size_t nprobe, uint32_t search_flags,
                              int64_t *out_ids, float *out_distances,
                              size_t *out_found);

/* Exact scan baseline. Cosine inputs must already be unit normalized. */
CV_API size_t cv_exact_search_f32(const float *vectors, const int64_t *ids,
                                  size_t count, size_t dimensions, int metric,
                                  const float *query, size_t k,
                                  int64_t *out_ids, float *out_distances);
/* Insert many vectors in one call. Each record still receives its own MVCC
   timestamp; this only removes per-call binding overhead, which dominates
   ingestion from Python. `vectors` is `count * dimensions` floats, row major.
   Returns the number inserted; on failure that is fewer than count. */
CV_API size_t cv_insert_batch(cv_index *index, const int64_t *ids,
                              const float *vectors, size_t count,
                              uint64_t *committed_timestamp);
/* Delete many ids in one call. Returns the number deleted. */
CV_API size_t cv_delete_batch(cv_index *index, const int64_t *ids, size_t count,
                              uint64_t *committed_timestamp);
/* Publish deletes and upserts as one MVCC commit. All changes are stamped with
 * one timestamp and become visible together; readers see either the state
 * before this call or the state after it, never a mixed result. `upsert_vectors`
 * is `upsert_count * dimensions` floats in row-major order. */
CV_API int cv_apply_changes(cv_index *index, const int64_t *delete_ids,
                            size_t delete_count, const int64_t *upsert_ids,
                            const float *upsert_vectors, size_t upsert_count,
                            uint64_t *committed_timestamp);
/* Insert with an attribute label. Requires CV_ENABLE_LABELS. */
CV_API int cv_insert_labeled(cv_index *index, int64_t id, const float *vector,
                             uint64_t label, uint64_t timestamp,
                             uint64_t *committed_timestamp);
/* Search restricted to records matching `filter`. A NULL filter behaves
   exactly like cv_search_with_options. */
CV_API size_t cv_search_filtered(cv_index *index, const float *query, size_t k,
                                 uint64_t snapshot, size_t nprobe,
                                 uint32_t search_flags,
                                 const cv_filter *filter, int64_t *out_ids,
                                 float *out_distances);
CV_API size_t cv_vacuum(cv_index *index, uint64_t oldest_snapshot,
                        size_t budget_versions);
/* Atomically replace path with a checksummed, self-contained checkpoint. */
CV_API int cv_save_checkpoint(cv_index *index, const char *path);
CV_API cv_index *cv_load_checkpoint(const char *path);
CV_API size_t cv_dimensions(const cv_index *index);
CV_API int cv_metric(const cv_index *index);
CV_API size_t cv_page_capacity(const cv_index *index);
CV_API size_t cv_default_nprobe(const cv_index *index);
CV_API uint32_t cv_flags(const cv_index *index);
CV_API size_t cv_rerank_factor(const cv_index *index);
CV_API uint64_t cv_clock(const cv_index *index);
CV_API int cv_get_stats(const cv_index *index, cv_stats *out);

// Per-page occupancy and how much of a page a filter actually selects.
//
// How well records sharing a label are grouped decides how much a filtered
// search can skip, and that is not visible from recall or latency: an index
// where every page holds a few matching records answers correctly and reads
// as merely slow. Writes up to `max_pages` entries and returns the number of
// pages, so a null `out_live` with max_pages 0 asks only for the count.
//
// `out_live` receives live records per page, `out_matching` how many of those
// satisfy `filter`. A page with matching == 0 is one the label union lets the
// search skip; a page with 0 < matching < live is one it must scan in full to
// reach a few records.
CV_API size_t cv_page_label_profile(const cv_index *index,
                                    const cv_filter *filter, uint64_t snapshot,
                                    uint32_t *out_live, uint32_t *out_matching,
                                    size_t max_pages);
CV_API int cv_get_last_search_metrics(cv_search_metrics *out);
CV_API const char *cv_last_error(void);

#ifdef __cplusplus
}
#endif
#endif
