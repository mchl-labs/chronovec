// Package chronovec is a Go binding for the ChronoVec C ABI: an
// approximate-nearest-neighbour index with snapshot isolation and bounded
// deletion.
//
// This is the raw tier -- the same vocabulary as the C ABI, Rust's
// chronovec::Index, and Python's chronovec.Index (Insert/Delete/Search/
// Vacuum/Save/Load/Clock), over int64 ids and bare []float32 vectors. It
// does not (yet) have a Go equivalent of Python's ergonomic Collection
// (string ids, metadata, filtering) or Rust's chronovec::collection::Collection.
//
// # Threading
//
// Searches are lock-free over immutable snapshots; mutations are
// thread-safe but serialise internally. An *Index may be shared across
// goroutines without an external lock -- this is lock-free reads, not
// parallel multiwriter progress, matching the C ABI's own contract.
//
// # Linking
//
// The native library must be built first (cmake -B build && cmake --build
// build from the repo root). By default this package links against
// ../../../build relative to its own source directory; override with the
// standard CGO_LDFLAGS/CGO_CFLAGS environment variables (e.g.
// CGO_LDFLAGS="-L/path/to/build -lchronovec" CGO_CFLAGS="-I/path/to/native/include")
// if the library lives elsewhere -- the same escape hatch CHRONOVEC_LIB_DIR
// gives the Rust binding, expressed through Go's own mechanism instead of a
// custom one.
package chronovec

/*
#cgo CFLAGS: -I${SRCDIR}/../../rust/chronovec-sys/native/include
#cgo LDFLAGS: -L${SRCDIR}/../../../build -lchronovec
#include <chronovec.h>
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

// Metric is the distance function an Index compares vectors with.
type Metric int

const (
	Cosine Metric = Metric(C.CV_COSINE)
	L2     Metric = Metric(C.CV_L2)
)

// Snapshot is a logical point in time. Returned by writes; accepted by
// [Index.SearchAsOf] and [Index.Vacuum].
type Snapshot uint64

// Hit is one search result.
type Hit struct {
	ID       int64
	Distance float32
}

// Stats reports index-wide counters: live vectors, pages, reclaimed
// versions, and so on.
type Stats struct {
	Clock                     uint64
	Pages                     uint64
	LiveVectors               uint64
	RetainedVersions          uint64
	AllocatedSlots            uint64
	Splits                    uint64
	ReclaimedVersions         uint64
	Merges                    uint64
	RoutingFullRebuilds       uint64
	RoutingIncrementalUpdates uint64
	RoutingCentroidBytes      uint64
	RoutingEdgeCount          uint64
}

// Options configures a new [Index]. Adding a field here never breaks
// callers using [DefaultOptions].
type Options struct {
	Metric       Metric
	PageCapacity int
	NProbe       int
	Screening    bool
	RerankFactor int
}

// DefaultOptions matches the C ABI's own defaults: cosine metric, page
// capacity 256, nprobe 16, screening on, rerank factor 4.
func DefaultOptions() Options {
	return Options{
		Metric:       Cosine,
		PageCapacity: 256,
		NProbe:       16,
		Screening:    true,
		RerankFactor: 4,
	}
}

// ErrNative wraps a message from the native layer's cv_last_error().
type ErrNative struct{ Message string }

func (e *ErrNative) Error() string { return e.Message }

func lastError() error {
	raw := C.cv_last_error()
	if raw == nil {
		return &ErrNative{Message: "unknown native error"}
	}
	return &ErrNative{Message: C.GoString(raw)}
}

// ErrDimension is returned when a vector's length does not match the
// index's dimensionality.
var ErrDimension = errors.New("vector length does not match index dimensions")

// Index is a snapshot-isolated approximate-nearest-neighbour index.
type Index struct {
	ptr        *C.cv_index
	dimensions int
}

// New creates an index of the given dimensionality.
func New(dimensions int, opts Options) (*Index, error) {
	flags := C.uint32_t(0)
	if opts.Screening {
		flags |= C.CV_ENABLE_INT8_SCREENING
	}
	ptr := C.cv_create_with_options(
		C.size_t(dimensions),
		C.int(opts.Metric),
		C.size_t(opts.PageCapacity),
		C.size_t(opts.NProbe),
		flags,
		C.size_t(opts.RerankFactor),
	)
	if ptr == nil {
		return nil, lastError()
	}
	idx := &Index{ptr: ptr, dimensions: dimensions}
	// Safety net: Close() is the documented way to release the native
	// handle, but a forgotten Close on a short-lived Index would otherwise
	// leak the C allocation past Go's own GC -- cgo pointers are invisible
	// to it. This finalizer is a backstop, not a substitute for Close(): it
	// runs at an unpredictable time (or not at all before process exit), so
	// a program creating many indices per second should still call Close.
	runtime.SetFinalizer(idx, func(idx *Index) { idx.Close() })
	return idx, nil
}

// Close releases the native resources. Safe to call more than once; safe
// to call on a nil *Index.
func (idx *Index) Close() error {
	if idx == nil || idx.ptr == nil {
		return nil
	}
	C.cv_destroy(idx.ptr)
	idx.ptr = nil
	runtime.SetFinalizer(idx, nil)
	return nil
}

func (idx *Index) checkVector(vector []float32) error {
	if len(vector) != idx.dimensions {
		return ErrDimension
	}
	return nil
}

// Dimensions is the vector width this index was created with.
func (idx *Index) Dimensions() int { return idx.dimensions }

// Clock is the current logical clock. Snapshot(Clock()+1) sees all
// committed writes.
func (idx *Index) Clock() Snapshot {
	return Snapshot(C.cv_clock(idx.ptr))
}

// Insert inserts or replaces id, returning the snapshot at which it became visible.
func (idx *Index) Insert(id int64, vector []float32) (Snapshot, error) {
	if err := idx.checkVector(vector); err != nil {
		return 0, err
	}
	var committed C.uint64_t
	rc := C.cv_insert(
		idx.ptr,
		C.int64_t(id),
		(*C.float)(unsafe.Pointer(&vector[0])),
		0,
		&committed,
	)
	if rc != 0 {
		return 0, lastError()
	}
	return Snapshot(committed), nil
}

// InsertMany inserts many vectors in one call. ids and vectors must be the
// same length; each row of vectors is idx.Dimensions() long.
func (idx *Index) InsertMany(ids []int64, vectors [][]float32) (Snapshot, error) {
	if len(ids) != len(vectors) {
		return 0, errors.New("ids and vectors must be the same length")
	}
	if len(ids) == 0 {
		return idx.Clock(), nil
	}
	flat := make([]float32, 0, len(ids)*idx.dimensions)
	for _, v := range vectors {
		if err := idx.checkVector(v); err != nil {
			return 0, err
		}
		flat = append(flat, v...)
	}
	var committed C.uint64_t
	found := C.cv_insert_batch(
		idx.ptr,
		(*C.int64_t)(unsafe.Pointer(&ids[0])),
		(*C.float)(unsafe.Pointer(&flat[0])),
		C.size_t(len(ids)),
		&committed,
	)
	if int(found) != len(ids) {
		return 0, lastError()
	}
	return Snapshot(committed), nil
}

// Delete marks id expired. Its space is returned by Vacuum.
func (idx *Index) Delete(id int64) (Snapshot, error) {
	var committed C.uint64_t
	rc := C.cv_delete(idx.ptr, C.int64_t(id), 0, &committed)
	if rc != 0 {
		return 0, lastError()
	}
	return Snapshot(committed), nil
}

// Search returns the k nearest neighbours of query in the current state.
func (idx *Index) Search(query []float32, k int) ([]Hit, error) {
	return idx.searchAt(query, k, 0)
}

// SearchAsOf returns the k nearest neighbours as the index existed at snapshot.
func (idx *Index) SearchAsOf(query []float32, k int, snapshot Snapshot) ([]Hit, error) {
	return idx.searchAt(query, k, C.uint64_t(snapshot))
}

func (idx *Index) searchAt(query []float32, k int, snapshot C.uint64_t) ([]Hit, error) {
	if err := idx.checkVector(query); err != nil {
		return nil, err
	}
	if k == 0 {
		return nil, nil
	}
	ids := make([]int64, k)
	distances := make([]float32, k)
	found := C.cv_search(
		idx.ptr,
		(*C.float)(unsafe.Pointer(&query[0])),
		C.size_t(k),
		snapshot,
		0,
		(*C.int64_t)(unsafe.Pointer(&ids[0])),
		(*C.float)(unsafe.Pointer(&distances[0])),
	)
	hits := make([]Hit, int(found))
	for i := range hits {
		hits[i] = Hit{ID: ids[i], Distance: distances[i]}
	}
	return hits, nil
}

// Vacuum physically reclaims at most budget versions no longer visible to
// any snapshot at or after oldest. Bounded: it does not scan the index.
func (idx *Index) Vacuum(oldest Snapshot, budget int) int {
	return int(C.cv_vacuum(idx.ptr, C.uint64_t(oldest), C.size_t(budget)))
}

// Save writes a checksummed checkpoint, replacing path atomically.
func (idx *Index) Save(path string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	if C.cv_save_checkpoint(idx.ptr, cPath) != 0 {
		return lastError()
	}
	return nil
}

// Load restores an index from a checkpoint written by [Index.Save].
func Load(path string) (*Index, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	ptr := C.cv_load_checkpoint(cPath)
	if ptr == nil {
		return nil, lastError()
	}
	idx := &Index{ptr: ptr, dimensions: int(C.cv_dimensions(ptr))}
	runtime.SetFinalizer(idx, func(idx *Index) { idx.Close() })
	return idx, nil
}

// Stats returns index-wide counters.
func (idx *Index) Stats() (Stats, error) {
	var out C.cv_stats
	if C.cv_get_stats(idx.ptr, &out) != 0 {
		return Stats{}, lastError()
	}
	return Stats{
		Clock:                     uint64(out.clock),
		Pages:                     uint64(out.pages),
		LiveVectors:               uint64(out.live_vectors),
		RetainedVersions:          uint64(out.retained_versions),
		AllocatedSlots:            uint64(out.allocated_slots),
		Splits:                    uint64(out.splits),
		ReclaimedVersions:         uint64(out.reclaimed_versions),
		Merges:                    uint64(out.merges),
		RoutingFullRebuilds:       uint64(out.routing_full_rebuilds),
		RoutingIncrementalUpdates: uint64(out.routing_incremental_updates),
		RoutingCentroidBytes:      uint64(out.routing_centroid_bytes),
		RoutingEdgeCount:          uint64(out.routing_edge_count),
	}, nil
}
