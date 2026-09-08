#ifndef CHRONOVEC_SQLITE_H
#define CHRONOVEC_SQLITE_H

#include <sqlite3.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Register the persistent ChronoVec virtual table on an existing SQLite
 * connection. This is the static-link entry point for embedded SQLite and a
 * future dqlite server integration; loadable extensions use
 * sqlite3_chronovec_init instead. */
int chronovec_register_vtab(sqlite3 *db);

#ifdef __cplusplus
}
#endif
#endif
