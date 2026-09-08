#include "chronovec_sqlite.h"
#include "chronovec.h"

#if defined(CHRONOVEC_SQLITE_LOADABLE) && defined(SQLITE_OMIT_LOAD_EXTENSION)
#undef SQLITE_OMIT_LOAD_EXTENSION
#endif
#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1

#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
std::mutex registry_mutex;
std::unordered_map<std::string,
                   std::unique_ptr<cv_index, decltype(&cv_destroy)>>
    registry;
std::unordered_map<std::string, size_t> registry_dimensions;

cv_index *lookup(sqlite3_context *ctx, sqlite3_value *name) {
  const auto *text = sqlite3_value_text(name);
  if (!text) {
    sqlite3_result_error(ctx, "index name is required", -1);
    return nullptr;
  }
  auto it = registry.find(reinterpret_cast<const char *>(text));
  if (it == registry.end()) {
    sqlite3_result_error(ctx, "ChronoVec index not found", -1);
    return nullptr;
  }
  return it->second.get();
}

void create_fn(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  if (argc < 2) {
    sqlite3_result_error(
        ctx, "chronovec_create(name, dimensions [, metric, capacity, nprobe])",
        -1);
    return;
  }
  std::lock_guard lock(registry_mutex);
  std::string name(reinterpret_cast<const char *>(sqlite3_value_text(argv[0])));
  size_t dimensions = sqlite3_value_int64(argv[1]);
  int metric = (argc > 2 && std::string(reinterpret_cast<const char *>(
                                sqlite3_value_text(argv[2]))) == "l2")
                   ? CV_L2
                   : CV_COSINE;
  size_t capacity = argc > 3 ? sqlite3_value_int64(argv[3]) : 256;
  size_t nprobe = argc > 4 ? sqlite3_value_int64(argv[4]) : 16;
  cv_index *raw = cv_create(dimensions, metric, capacity, nprobe);
  if (!raw) {
    sqlite3_result_error(ctx, cv_last_error(), -1);
    return;
  }
  registry.insert_or_assign(
      name, std::unique_ptr<cv_index, decltype(&cv_destroy)>(raw, cv_destroy));
  registry_dimensions[name] = dimensions;
  sqlite3_result_int(ctx, 1);
}

void insert_fn(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  std::lock_guard lock(registry_mutex);
  cv_index *index = lookup(ctx, argv[0]);
  if (!index)
    return;
  const auto *name_text = sqlite3_value_text(argv[0]);
  size_t dimensions =
      registry_dimensions[reinterpret_cast<const char *>(name_text)];
  const float *value = nullptr;
  if (sqlite3_value_type(argv[2]) != SQLITE_BLOB ||
      sqlite3_value_bytes(argv[2]) !=
          static_cast<int>(dimensions * sizeof(float))) {
    sqlite3_result_error(
        ctx,
        "vector must be a packed float32 BLOB with the configured dimensions",
        -1);
    return;
  }
  value = static_cast<const float *>(sqlite3_value_blob(argv[2]));
  uint64_t committed = 0;
  uint64_t requested = argc > 3 ? sqlite3_value_int64(argv[3]) : 0;
  if (cv_insert(index, sqlite3_value_int64(argv[1]), value, requested,
                &committed))
    sqlite3_result_error(ctx, cv_last_error(), -1);
  else
    sqlite3_result_int64(ctx, committed);
}

void delete_fn(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  std::lock_guard lock(registry_mutex);
  cv_index *index = lookup(ctx, argv[0]);
  if (!index)
    return;
  uint64_t committed = 0;
  uint64_t requested = argc > 2 ? sqlite3_value_int64(argv[2]) : 0;
  if (cv_delete(index, sqlite3_value_int64(argv[1]), requested, &committed))
    sqlite3_result_error(ctx, cv_last_error(), -1);
  else
    sqlite3_result_int64(ctx, committed);
}

void search_fn(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  std::lock_guard lock(registry_mutex);
  cv_index *index = lookup(ctx, argv[0]);
  if (!index)
    return;
  const auto *name_text = sqlite3_value_text(argv[0]);
  size_t dimensions =
      registry_dimensions[reinterpret_cast<const char *>(name_text)];
  if (sqlite3_value_type(argv[1]) != SQLITE_BLOB ||
      sqlite3_value_bytes(argv[1]) !=
          static_cast<int>(dimensions * sizeof(float))) {
    sqlite3_result_error(
        ctx,
        "query must be a packed float32 BLOB with the configured dimensions",
        -1);
    return;
  }
  size_t k = sqlite3_value_int64(argv[2]);
  uint64_t snapshot = argc > 3 ? sqlite3_value_int64(argv[3]) : 0;
  size_t nprobe = argc > 4 ? sqlite3_value_int64(argv[4]) : 0;
  std::vector<int64_t> ids(k);
  std::vector<float> distances(k);
  size_t count =
      cv_search(index, static_cast<const float *>(sqlite3_value_blob(argv[1])),
                k, snapshot, nprobe, ids.data(), distances.data());
  std::ostringstream json;
  json << '[';
  for (size_t i = 0; i < count; ++i) {
    if (i)
      json << ',';
    json << "{\"id\":" << ids[i] << ",\"distance\":" << distances[i] << '}';
  }
  json << ']';
  sqlite3_result_text(ctx, json.str().c_str(), -1, SQLITE_TRANSIENT);
}

void vacuum_fn(sqlite3_context *ctx, int, sqlite3_value **argv) {
  std::lock_guard lock(registry_mutex);
  cv_index *index = lookup(ctx, argv[0]);
  if (!index)
    return;
  sqlite3_result_int64(ctx, cv_vacuum(index, sqlite3_value_int64(argv[1]),
                                      sqlite3_value_int64(argv[2])));
}

void stats_fn(sqlite3_context *ctx, int, sqlite3_value **argv) {
  std::lock_guard lock(registry_mutex);
  cv_index *index = lookup(ctx, argv[0]);
  if (!index)
    return;
  cv_stats s{};
  cv_get_stats(index, &s);
  std::ostringstream json;
  json << "{\"clock\":" << s.clock << ",\"pages\":" << s.pages
       << ",\"live_vectors\":" << s.live_vectors
       << ",\"retained_versions\":" << s.retained_versions
       << ",\"allocated_slots\":" << s.allocated_slots
       << ",\"splits\":" << s.splits
       << ",\"reclaimed_versions\":" << s.reclaimed_versions
       << ",\"merges\":" << s.merges << '}';
  sqlite3_result_text(ctx, json.str().c_str(), -1, SQLITE_TRANSIENT);
}
} // namespace

extern "C" int sqlite3_chronovec_init(sqlite3 *db, char **error,
                                      const sqlite3_api_routines *api) {
  SQLITE_EXTENSION_INIT2(api);
  int module_rc = chronovec_register_vtab(db);
  if (module_rc != SQLITE_OK) {
    if (error)
      *error = sqlite3_mprintf("failed to register chronovec virtual table");
    return module_rc;
  }
  struct Function {
    const char *name;
    int args;
    void (*fn)(sqlite3_context *, int, sqlite3_value **);
  } functions[] = {
      {"chronovec_create", -1, create_fn}, {"chronovec_insert", -1, insert_fn},
      {"chronovec_delete", -1, delete_fn}, {"chronovec_search", -1, search_fn},
      {"chronovec_vacuum", 3, vacuum_fn},  {"chronovec_stats", 1, stats_fn},
  };
  for (auto &f : functions) {
    int rc = sqlite3_create_function(db, f.name, f.args, SQLITE_UTF8, nullptr,
                                     f.fn, nullptr, nullptr);
    if (rc != SQLITE_OK) {
      if (error)
        *error = sqlite3_mprintf("failed to register %s", f.name);
      return rc;
    }
  }
  return SQLITE_OK;
}

extern "C" int sqlite3_extension_init(sqlite3 *db, char **error,
                                      const sqlite3_api_routines *api) {
  return sqlite3_chronovec_init(db, error, api);
}
