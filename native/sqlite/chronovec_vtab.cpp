#include "chronovec.h"
#include "chronovec_sqlite.h"
#if defined(CHRONOVEC_SQLITE_LOADABLE) && defined(SQLITE_OMIT_LOAD_EXTENSION)
#undef SQLITE_OMIT_LOAD_EXTENSION
#endif
#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT3

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {
struct VTab {
  sqlite3_vtab base{};
  sqlite3 *db = nullptr;
  std::string name, log_table;
  size_t dimensions = 0, capacity = 256, nprobe = 16;
  int metric = CV_COSINE;
  cv_index *index = nullptr;
  bool dirty = true;
};
struct Cursor {
  sqlite3_vtab_cursor base{};
  std::vector<int64_t> ids;
  std::vector<float> distances;
  size_t position = 0;
};

std::string quote_identifier(const char *value) {
  std::string result = "\"";
  for (const char *p = value; *p; ++p) {
    if (*p == '\"')
      result += '\"';
    result += *p;
  }
  return result + '\"';
}
void set_error(sqlite3_vtab *tab, const char *message) {
  sqlite3_free(tab->zErrMsg);
  tab->zErrMsg = sqlite3_mprintf("%s", message);
}

bool parse_option(const char *argument, const char *key, std::string &value) {
  std::string item(argument), prefix = std::string(key) + "=";
  if (item.rfind(prefix, 0) != 0)
    return false;
  value = item.substr(prefix.size());
  if (value.size() > 1 && ((value.front() == '\'' && value.back() == '\'') ||
                           (value.front() == '\"' && value.back() == '\"')))
    value = value.substr(1, value.size() - 2);
  return true;
}

int rebuild(VTab *vtab) {
  cv_destroy(vtab->index);
  vtab->index =
      cv_create(vtab->dimensions, vtab->metric, vtab->capacity, vtab->nprobe);
  if (!vtab->index)
    return SQLITE_NOMEM;
  std::string sql =
      "SELECT seq,op,id,vector FROM " + vtab->log_table + " ORDER BY seq";
  sqlite3_stmt *statement = nullptr;
  int rc = sqlite3_prepare_v2(vtab->db, sql.c_str(), -1, &statement, nullptr);
  if (rc != SQLITE_OK)
    return rc;
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    uint64_t seq = sqlite3_column_int64(statement, 0);
    int op = sqlite3_column_int(statement, 1);
    int64_t id = sqlite3_column_int64(statement, 2);
    uint64_t committed = 0;
    if (op == 1) {
      if (sqlite3_column_bytes(statement, 3) !=
          int(vtab->dimensions * sizeof(float))) {
        rc = SQLITE_CORRUPT;
        break;
      }
      if (cv_insert(
              vtab->index, id,
              static_cast<const float *>(sqlite3_column_blob(statement, 3)),
              seq, &committed)) {
        rc = SQLITE_ERROR;
        break;
      }
    } else if (cv_delete(vtab->index, id, seq, &committed)) {
      rc = SQLITE_ERROR;
      break;
    }
  }
  if (rc == SQLITE_DONE)
    rc = SQLITE_OK;
  sqlite3_finalize(statement);
  vtab->dirty = rc != SQLITE_OK;
  return rc;
}

int connect_impl(sqlite3 *db, int argc, const char *const *argv,
                 sqlite3_vtab **out, char **error, bool create) {
  auto *vtab = new (std::nothrow) VTab();
  if (!vtab)
    return SQLITE_NOMEM;
  vtab->db = db;
  vtab->name = argv[2];
  vtab->log_table = quote_identifier((vtab->name + "_chronovec_log").c_str());
  std::string dimensions, metric, capacity, nprobe;
  for (int i = 3; i < argc; ++i) {
    parse_option(argv[i], "dim", dimensions) ||
        parse_option(argv[i], "dimensions", dimensions) ||
        parse_option(argv[i], "metric", metric) ||
        parse_option(argv[i], "capacity", capacity) ||
        parse_option(argv[i], "nprobe", nprobe);
  }
  if (create) {
    if (dimensions.empty()) {
      *error = sqlite3_mprintf("chronovec requires dim=N");
      delete vtab;
      return SQLITE_ERROR;
    }
    try {
      vtab->dimensions = std::stoull(dimensions);
      if (!capacity.empty())
        vtab->capacity = std::stoull(capacity);
      if (!nprobe.empty())
        vtab->nprobe = std::stoull(nprobe);
    } catch (...) {
      *error = sqlite3_mprintf("invalid chronovec numeric option");
      delete vtab;
      return SQLITE_ERROR;
    }
    vtab->metric = metric == "l2" ? CV_L2 : CV_COSINE;
    std::string sql = "CREATE TABLE IF NOT EXISTS " + vtab->log_table +
                      "(seq INTEGER PRIMARY KEY AUTOINCREMENT,op INTEGER NOT "
                      "NULL CHECK(op IN(1,2)),id INTEGER NOT NULL,vector BLOB)";
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, error);
    if (rc != SQLITE_OK) {
      delete vtab;
      return rc;
    }
  } else {
    // Configuration is preserved in sqlite_master's CREATE VIRTUAL TABLE SQL.
    if (dimensions.empty()) {
      *error = sqlite3_mprintf("chronovec schema is missing dim=N");
      delete vtab;
      return SQLITE_CORRUPT;
    }
    try {
      vtab->dimensions = std::stoull(dimensions);
      if (!capacity.empty())
        vtab->capacity = std::stoull(capacity);
      if (!nprobe.empty())
        vtab->nprobe = std::stoull(nprobe);
    } catch (...) {
      delete vtab;
      return SQLITE_CORRUPT;
    }
    vtab->metric = metric == "l2" ? CV_L2 : CV_COSINE;
  }
  int rc = sqlite3_declare_vtab(
      db,
      "CREATE TABLE x(id INTEGER,distance REAL,vector BLOB,query BLOB HIDDEN,k "
      "INTEGER HIDDEN,snapshot INTEGER HIDDEN,nprobe INTEGER HIDDEN)");
  if (rc != SQLITE_OK) {
    delete vtab;
    return rc;
  }
  sqlite3_vtab_config(db, SQLITE_VTAB_CONSTRAINT_SUPPORT, 1);
  rc = rebuild(vtab);
  if (rc != SQLITE_OK) {
    *error =
        sqlite3_mprintf("failed to replay ChronoVec log: %s", cv_last_error());
    cv_destroy(vtab->index);
    delete vtab;
    return rc;
  }
  *out = &vtab->base;
  return SQLITE_OK;
}
int create_fn(sqlite3 *d, void *, int a, const char *const v[],
              sqlite3_vtab **o, char **e) {
  return connect_impl(d, a, v, o, e, true);
}
int connect_fn(sqlite3 *d, void *, int a, const char *const v[],
               sqlite3_vtab **o, char **e) {
  return connect_impl(d, a, v, o, e, false);
}
int disconnect_fn(sqlite3_vtab *base) {
  auto *v = reinterpret_cast<VTab *>(base);
  cv_destroy(v->index);
  delete v;
  return SQLITE_OK;
}
int destroy_fn(sqlite3_vtab *base) {
  auto *v = reinterpret_cast<VTab *>(base);
  std::string sql = "DROP TABLE IF EXISTS " + v->log_table;
  int rc = sqlite3_exec(v->db, sql.c_str(), nullptr, nullptr, nullptr);
  disconnect_fn(base);
  return rc;
}

int best_index(sqlite3_vtab *, sqlite3_index_info *info) {
  int found[5] = {-1, -1, -1, -1, -1};
  bool query = false;
  for (int i = 0; i < info->nConstraint; ++i) {
    auto &c = info->aConstraint[i];
    if (!c.usable || c.op != SQLITE_INDEX_CONSTRAINT_EQ)
      continue;
    if (c.iColumn == 3) {
      found[0] = i;
      info->idxNum |= 1;
      query = true;
    } else if (c.iColumn == 4) {
      found[1] = i;
      info->idxNum |= 2;
    } else if (c.iColumn == 5) {
      found[2] = i;
      info->idxNum |= 4;
    } else if (c.iColumn == 6) {
      found[3] = i;
      info->idxNum |= 8;
    } else if (c.iColumn == 0) {
      found[4] = i;
      info->idxNum |= 16;
    }
  }
  int argument = 1;
  for (int index : found)
    if (index >= 0) {
      info->aConstraintUsage[index].argvIndex = argument++;
      info->aConstraintUsage[index].omit = 1;
    }
  if (info->nOrderBy == 1 && info->aOrderBy[0].iColumn == 1 &&
      !info->aOrderBy[0].desc)
    info->orderByConsumed = 1;
  info->estimatedCost = query ? 100.0 : 1e12;
  info->estimatedRows = query ? 10 : 1000000000;
  return SQLITE_OK;
}
int open_fn(sqlite3_vtab *, sqlite3_vtab_cursor **out) {
  auto *c = new (std::nothrow) Cursor();
  if (!c)
    return SQLITE_NOMEM;
  *out = &c->base;
  return SQLITE_OK;
}
int close_fn(sqlite3_vtab_cursor *base) {
  delete reinterpret_cast<Cursor *>(base);
  return SQLITE_OK;
}
int filter_fn(sqlite3_vtab_cursor *base, int idx, const char *, int argc,
              sqlite3_value **argv) {
  auto *c = reinterpret_cast<Cursor *>(base);
  auto *v = reinterpret_cast<VTab *>(base->pVtab);
  c->ids.clear();
  c->distances.clear();
  c->position = 0;
  if (v->dirty) {
    int rc = rebuild(v);
    if (rc != SQLITE_OK) {
      set_error(&v->base, cv_last_error());
      return rc;
    }
  }
  if (!(idx & 1)) {
    if (idx & 16) {
      c->ids.push_back(sqlite3_value_int64(argv[0]));
      c->distances.push_back(0);
      return SQLITE_OK;
    }
    set_error(&v->base, "ChronoVec scans require WHERE query = float32_blob");
    return SQLITE_CONSTRAINT;
  }
  int arg = 0;
  sqlite3_value *q = argv[arg++];
  size_t k = 10;
  uint64_t snapshot = 0;
  size_t probes = 0;
  if (idx & 2)
    k = sqlite3_value_int64(argv[arg++]);
  if (idx & 4)
    snapshot = sqlite3_value_int64(argv[arg++]);
  if (idx & 8)
    probes = sqlite3_value_int64(argv[arg++]);
  if (idx & 16)
    ++arg;
  if (sqlite3_value_type(q) != SQLITE_BLOB ||
      sqlite3_value_bytes(q) != int(v->dimensions * sizeof(float))) {
    set_error(&v->base, "query must be a packed float32 BLOB of dim floats");
    return SQLITE_MISMATCH;
  }
  c->ids.resize(k);
  c->distances.resize(k);
  size_t count =
      cv_search(v->index, static_cast<const float *>(sqlite3_value_blob(q)), k,
                snapshot, probes, c->ids.data(), c->distances.data());
  c->ids.resize(count);
  c->distances.resize(count);
  return SQLITE_OK;
}
int next_fn(sqlite3_vtab_cursor *b) {
  ++reinterpret_cast<Cursor *>(b)->position;
  return SQLITE_OK;
}
int eof_fn(sqlite3_vtab_cursor *b) {
  auto *c = reinterpret_cast<Cursor *>(b);
  return c->position >= c->ids.size();
}
int column_fn(sqlite3_vtab_cursor *b, sqlite3_context *ctx, int column) {
  auto *c = reinterpret_cast<Cursor *>(b);
  if (column == 0)
    sqlite3_result_int64(ctx, c->ids[c->position]);
  else if (column == 1)
    sqlite3_result_double(ctx, c->distances[c->position]);
  else
    sqlite3_result_null(ctx);
  return SQLITE_OK;
}
sqlite3_int64 rowid_value(Cursor *c) {
  return c->position < c->ids.size() ? c->ids[c->position] : 0;
}
int rowid_fn(sqlite3_vtab_cursor *b, sqlite3_int64 *out) {
  *out = rowid_value(reinterpret_cast<Cursor *>(b));
  return SQLITE_OK;
}
int update_fn(sqlite3_vtab *base, int argc, sqlite3_value **argv,
              sqlite3_int64 *rowid) {
  auto *v = reinterpret_cast<VTab *>(base);
  int op;
  int64_t id;
  const void *blob = nullptr;
  int bytes = 0;
  if (argc == 1) {
    op = 2;
    id = sqlite3_value_int64(argv[0]);
  } else {
    op = 1;
    id = sqlite3_value_type(argv[2]) == SQLITE_NULL
             ? sqlite3_value_int64(argv[1])
             : sqlite3_value_int64(argv[2]);
    blob = sqlite3_value_blob(argv[4]);
    bytes = sqlite3_value_bytes(argv[4]);
    if (bytes != int(v->dimensions * sizeof(float))) {
      set_error(base, "vector must be a packed float32 BLOB of dim floats");
      return SQLITE_MISMATCH;
    }
  }
  std::string sql =
      "INSERT INTO " + v->log_table + "(op,id,vector) VALUES(?,?,?)";
  sqlite3_stmt *statement = nullptr;
  int rc = sqlite3_prepare_v2(v->db, sql.c_str(), -1, &statement, nullptr);
  if (rc != SQLITE_OK)
    return rc;
  sqlite3_bind_int(statement, 1, op);
  sqlite3_bind_int64(statement, 2, id);
  if (op == 1)
    sqlite3_bind_blob(statement, 3, blob, bytes, SQLITE_TRANSIENT);
  else
    sqlite3_bind_null(statement, 3);
  rc = sqlite3_step(statement);
  sqlite3_finalize(statement);
  if (rc != SQLITE_DONE)
    return rc;
  uint64_t seq = sqlite3_last_insert_rowid(v->db), committed = 0;
  if (op == 1)
    rc = cv_insert(v->index, id, static_cast<const float *>(blob), seq,
                   &committed)
             ? SQLITE_ERROR
             : SQLITE_OK;
  else
    rc = cv_delete(v->index, id, seq, &committed) ? SQLITE_ERROR : SQLITE_OK;
  if (rc != SQLITE_OK) {
    v->dirty = true;
    set_error(base, cv_last_error());
    return rc;
  }
  if (rowid)
    *rowid = id;
  return SQLITE_OK;
}
int begin_fn(sqlite3_vtab *) { return SQLITE_OK; }
int sync_fn(sqlite3_vtab *) { return SQLITE_OK; }
int commit_fn(sqlite3_vtab *) { return SQLITE_OK; }
int rollback_fn(sqlite3_vtab *b) {
  reinterpret_cast<VTab *>(b)->dirty = true;
  return SQLITE_OK;
}
// Trailing optional methods (xFindFunction..xIntegrity, added across SQLite
// versions from 3.7 through 3.44) are left unspecified rather than
// null-padded: aggregate init zero-fills whatever the compiling SQLite
// header's struct defines beyond xRollback, so this compiles against any
// SQLite version instead of hardcoding one vintage's exact field count.
sqlite3_module module = {
    3,          create_fn, connect_fn,  best_index, disconnect_fn,
    destroy_fn, open_fn,   close_fn,    filter_fn,  next_fn,
    eof_fn,     column_fn, rowid_fn,    update_fn,  begin_fn,
    sync_fn,    commit_fn, rollback_fn};
} // namespace

extern "C" int chronovec_register_vtab(sqlite3 *db) {
  return sqlite3_create_module_v2(db, "chronovec", &module, nullptr, nullptr);
}
