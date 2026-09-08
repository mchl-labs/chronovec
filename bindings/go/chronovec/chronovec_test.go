package chronovec

import (
	"testing"
)

func newTestIndex(t *testing.T, dims int) *Index {
	t.Helper()
	idx, err := New(dims, DefaultOptions())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { idx.Close() })
	return idx
}

func TestInsertAndSearch(t *testing.T) {
	idx := newTestIndex(t, 3)
	if _, err := idx.Insert(1, []float32{1, 0, 0}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if _, err := idx.Insert(2, []float32{0, 1, 0}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	hits, err := idx.Search([]float32{1, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != 1 {
		t.Fatalf("expected [id=1], got %+v", hits)
	}
}

func TestInsertDimensionMismatch(t *testing.T) {
	idx := newTestIndex(t, 3)
	_, err := idx.Insert(1, []float32{1, 0})
	if err != ErrDimension {
		t.Fatalf("expected ErrDimension, got %v", err)
	}
}

func TestDeleteRemovesFromSearch(t *testing.T) {
	idx := newTestIndex(t, 3)
	if _, err := idx.Insert(1, []float32{1, 0, 0}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if _, err := idx.Delete(1); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	hits, err := idx.Search([]float32{1, 0, 0}, 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 0 {
		t.Fatalf("expected no hits after delete, got %+v", hits)
	}
}

func TestSearchAsOfSeesThePast(t *testing.T) {
	idx := newTestIndex(t, 3)
	if _, err := idx.Insert(1, []float32{1, 0, 0}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	before := idx.Clock()
	if _, err := idx.Delete(1); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	now, err := idx.Search([]float32{1, 0, 0}, 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(now) != 0 {
		t.Fatalf("expected id 1 gone now, got %+v", now)
	}

	then, err := idx.SearchAsOf([]float32{1, 0, 0}, 5, before)
	if err != nil {
		t.Fatalf("SearchAsOf: %v", err)
	}
	if len(then) != 1 || then[0].ID != 1 {
		t.Fatalf("expected id 1 visible before the delete, got %+v", then)
	}
}

func TestInsertMany(t *testing.T) {
	idx := newTestIndex(t, 3)
	ids := []int64{1, 2, 3}
	vectors := [][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	if _, err := idx.InsertMany(ids, vectors); err != nil {
		t.Fatalf("InsertMany: %v", err)
	}
	stats, err := idx.Stats()
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if stats.LiveVectors != 3 {
		t.Fatalf("expected 3 live vectors, got %d", stats.LiveVectors)
	}
}

func TestInsertManyMismatchedLengths(t *testing.T) {
	idx := newTestIndex(t, 3)
	_, err := idx.InsertMany([]int64{1, 2}, [][]float32{{1, 0, 0}})
	if err == nil {
		t.Fatal("expected an error for mismatched lengths")
	}
}

func TestVacuumReclaims(t *testing.T) {
	idx := newTestIndex(t, 3)
	if _, err := idx.Insert(1, []float32{1, 0, 0}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if _, err := idx.Delete(1); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	horizon := Snapshot(uint64(idx.Clock()) + 1)
	freed := idx.Vacuum(horizon, 64)
	if freed == 0 {
		t.Fatal("expected vacuum to reclaim the deleted version")
	}
}

func TestSaveAndLoadRoundTrip(t *testing.T) {
	idx := newTestIndex(t, 3)
	if _, err := idx.Insert(1, []float32{1, 0, 0}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	path := t.TempDir() + "/index.cvec"
	if err := idx.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	restored, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { restored.Close() })

	if restored.Dimensions() != 3 {
		t.Fatalf("expected dimensions 3, got %d", restored.Dimensions())
	}
	hits, err := restored.Search([]float32{1, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != 1 {
		t.Fatalf("expected restored index to contain id 1, got %+v", hits)
	}
}

func TestCloseIsIdempotent(t *testing.T) {
	idx := newTestIndex(t, 3)
	if err := idx.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := idx.Close(); err != nil {
		t.Fatalf("second Close should be a no-op, got: %v", err)
	}
}

func TestCloseOnNilIndex(t *testing.T) {
	var idx *Index
	if err := idx.Close(); err != nil {
		t.Fatalf("Close on nil *Index should be a no-op, got: %v", err)
	}
}
