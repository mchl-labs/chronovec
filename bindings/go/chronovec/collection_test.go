package chronovec

import (
	"testing"
)

func newTestCollection(t *testing.T, dims int) *Collection {
	t.Helper()
	col, err := NewCollection(dims)
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	t.Cleanup(func() { col.Close() })
	return col
}

func strPtr(s string) *string { return &s }

func TestStableIDMatchesPythonAndRust(t *testing.T) {
	// Same fixtures as bindings/rust/chronovec/src/collection.rs's
	// stable_id_matches_python test -- generated with:
	// python -c "from chronovec.collection import stable_id; print(stable_id('a'))"
	// One fixture per input class (ascii, hyphenated, spaced, empty, unicode)
	// so a hashing bug that only shows up on non-ASCII or empty input is caught.
	cases := []struct {
		input    string
		expected int64
	}{
		{"a", 2340832890917691671},
		{"doc-1", 3132883647696894624},
		{"user-preference-42", 4816184505566365106},
		{"hello world", 4882774824043272360},
		{"", 8238016292129134938},
		{"unicode-café-🎉", 4281077293084237012},
	}
	for _, c := range cases {
		got, err := StableID(c.input)
		if err != nil {
			t.Fatalf("StableID(%q): %v", c.input, err)
		}
		if got != c.expected {
			t.Errorf("StableID(%q) = %d, want %d", c.input, got, c.expected)
		}
	}
}

func TestStableIDIntPassesThrough(t *testing.T) {
	got, err := StableID(int64(42))
	if err != nil || got != 42 {
		t.Fatalf("StableID(42) = %d, %v, want 42, nil", got, err)
	}
	got, err = StableID(42) // plain int
	if err != nil || got != 42 {
		t.Fatalf("StableID(int 42) = %d, %v, want 42, nil", got, err)
	}
}

func TestStableIDRejectsUnsupportedType(t *testing.T) {
	if _, err := StableID(3.14); err == nil {
		t.Fatal("expected an error for a float64 id")
	}
}

func TestAddAndQueryByStringID(t *testing.T) {
	col := newTestCollection(t, 3)
	if _, err := col.Add("a", []float32{1, 0, 0}, nil, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if _, err := col.Add("b", []float32{0, 1, 0}, nil, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	hits, err := col.Query([]float32{1, 0, 0}, 1, QueryOptions{})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != "a" {
		t.Fatalf("expected [id=a], got %+v", hits)
	}
}

func TestIntIDsPassThrough(t *testing.T) {
	col := newTestCollection(t, 3)
	if _, err := col.Add(int64(42), []float32{1, 0, 0}, nil, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	hits, err := col.Query([]float32{1, 0, 0}, 1, QueryOptions{})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if hits[0].ID != int64(42) {
		t.Fatalf("expected id int64(42), got %#v", hits[0].ID)
	}
}

func TestMetadataAndDocumentRoundTrip(t *testing.T) {
	col := newTestCollection(t, 3)
	meta := map[string]any{"lang": "en", "year": 2024}
	if _, err := col.Add("a", []float32{1, 0, 0}, meta, strPtr("hello")); err != nil {
		t.Fatalf("Add: %v", err)
	}
	hits, err := col.Query([]float32{1, 0, 0}, 1, QueryOptions{})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if hits[0].Metadata["lang"] != "en" || hits[0].Metadata["year"] != 2024 {
		t.Fatalf("unexpected metadata: %+v", hits[0].Metadata)
	}
	if hits[0].Document == nil || *hits[0].Document != "hello" {
		t.Fatalf("unexpected document: %v", hits[0].Document)
	}
}

func TestSnapshotIsolation(t *testing.T) {
	col := newTestCollection(t, 3)
	if _, err := col.Add("a", []float32{1, 0, 0}, nil, strPtr("v1")); err != nil {
		t.Fatalf("Add: %v", err)
	}
	before := col.Snapshot()
	if _, err := col.Add("a", []float32{1, 0, 0}, nil, strPtr("v2")); err != nil {
		t.Fatalf("Add: %v", err)
	}

	now, err := col.Query([]float32{1, 0, 0}, 1, QueryOptions{})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if *now[0].Document != "v2" {
		t.Fatalf("expected v2 now, got %v", *now[0].Document)
	}

	then, err := col.Query([]float32{1, 0, 0}, 1, QueryOptions{Snapshot: &before})
	if err != nil {
		t.Fatalf("Query as of before: %v", err)
	}
	if *then[0].Document != "v1" {
		t.Fatalf("expected v1 as of before, got %v", *then[0].Document)
	}
}

func TestWhereFilterEquality(t *testing.T) {
	col := newTestCollection(t, 3)
	col.Add("a", []float32{1, 0, 0}, map[string]any{"lang": "en"}, nil)
	col.Add("b", []float32{1, 0, 0}, map[string]any{"lang": "fr"}, nil)

	hits, err := col.Query([]float32{1, 0, 0}, 5, QueryOptions{
		Where: map[string]any{"lang": "fr"},
	})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != "b" {
		t.Fatalf("expected [id=b], got %+v", hits)
	}
}

func TestWhereFilterComparisonOperators(t *testing.T) {
	col := newTestCollection(t, 3)
	col.Add("a", []float32{1, 0, 0}, map[string]any{"year": 2020}, nil)
	col.Add("b", []float32{1, 0, 0}, map[string]any{"year": 2024}, nil)

	hits, err := col.Query([]float32{1, 0, 0}, 5, QueryOptions{
		Where: map[string]any{"year": map[string]any{"$gte": 2024}},
	})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != "b" {
		t.Fatalf("expected [id=b], got %+v", hits)
	}
}

func TestWhereFilterAndOrComposition(t *testing.T) {
	col := newTestCollection(t, 3)
	col.Add("a", []float32{1, 0, 0}, map[string]any{"lang": "en", "year": 2024}, nil)
	col.Add("b", []float32{1, 0, 0}, map[string]any{"lang": "en", "year": 2020}, nil)
	col.Add("c", []float32{1, 0, 0}, map[string]any{"lang": "fr", "year": 2024}, nil)

	andHits, err := col.Query([]float32{1, 0, 0}, 5, QueryOptions{
		Where: map[string]any{
			"$and": []any{
				map[string]any{"lang": "en"},
				map[string]any{"year": map[string]any{"$gte": 2024}},
			},
		},
	})
	if err != nil {
		t.Fatalf("Query $and: %v", err)
	}
	if len(andHits) != 1 || andHits[0].ID != "a" {
		t.Fatalf("expected [id=a], got %+v", andHits)
	}

	orHits, err := col.Query([]float32{1, 0, 0}, 5, QueryOptions{
		Where: map[string]any{
			"$or": []any{
				map[string]any{"lang": "fr"},
				map[string]any{"year": 2020},
			},
		},
	})
	if err != nil {
		t.Fatalf("Query $or: %v", err)
	}
	if len(orHits) != 2 {
		t.Fatalf("expected 2 hits, got %+v", orHits)
	}
}

func TestWhereFilterInContainsRegex(t *testing.T) {
	col := newTestCollection(t, 3)
	col.Add("a", []float32{1, 0, 0}, map[string]any{"title": "ChronoVec MVCC"}, nil)
	col.Add("b", []float32{1, 0, 0}, map[string]any{"title": "other"}, nil)

	inHits, _ := col.Query([]float32{1, 0, 0}, 5, QueryOptions{
		Where: map[string]any{"title": map[string]any{"$in": []any{"ChronoVec MVCC", "nope"}}},
	})
	if len(inHits) != 1 {
		t.Fatalf("$in: expected 1 hit, got %+v", inHits)
	}

	containsHits, _ := col.Query([]float32{1, 0, 0}, 5, QueryOptions{
		Where: map[string]any{"title": map[string]any{"$contains": "MVCC"}},
	})
	if len(containsHits) != 1 {
		t.Fatalf("$contains: expected 1 hit, got %+v", containsHits)
	}

	regexHits, _ := col.Query([]float32{1, 0, 0}, 5, QueryOptions{
		Where: map[string]any{"title": map[string]any{"$regex": "^Chrono.*"}},
	})
	if len(regexHits) != 1 || regexHits[0].ID != "a" {
		t.Fatalf("$regex: expected [id=a], got %+v", regexHits)
	}
}

func TestDeleteAndDeleteWhere(t *testing.T) {
	col := newTestCollection(t, 3)
	col.Add("a", []float32{1, 0, 0}, map[string]any{"kind": "hyp"}, nil)
	col.Add("b", []float32{0, 1, 0}, map[string]any{"kind": "fact"}, nil)
	if col.Count() != 2 {
		t.Fatalf("expected count 2, got %d", col.Count())
	}

	removed, err := col.DeleteWhere(map[string]any{"kind": "hyp"})
	if err != nil {
		t.Fatalf("DeleteWhere: %v", err)
	}
	if len(removed) != 1 || removed[0] != "a" {
		t.Fatalf("expected [a], got %+v", removed)
	}
	if col.Count() != 1 {
		t.Fatalf("expected count 1, got %d", col.Count())
	}

	ok, err := col.Delete("b")
	if err != nil || !ok {
		t.Fatalf("Delete(b): %v, %v", ok, err)
	}
	ok, err = col.Delete("b")
	if err != nil || ok {
		t.Fatalf("Delete(b) again: expected false, got %v, %v", ok, err)
	}
	if col.Count() != 0 {
		t.Fatalf("expected count 0, got %d", col.Count())
	}
}

func TestGetAndGetWhere(t *testing.T) {
	col := newTestCollection(t, 3)
	col.Add("a", []float32{1, 0, 0}, map[string]any{"lang": "en"}, nil)
	col.Add("b", []float32{0, 1, 0}, map[string]any{"lang": "fr"}, nil)

	one, err := col.Get("a")
	if err != nil || one == nil {
		t.Fatalf("Get(a): %v, %v", one, err)
	}
	if one.Document != nil {
		t.Fatalf("expected nil document, got %v", *one.Document)
	}

	missing, err := col.Get("missing")
	if err != nil || missing != nil {
		t.Fatalf("Get(missing): expected nil, nil, got %v, %v", missing, err)
	}

	many, err := col.GetWhere(map[string]any{"lang": "fr"})
	if err != nil || len(many) != 1 || many[0].ID != "b" {
		t.Fatalf("GetWhere: %+v, %v", many, err)
	}
}

func TestUpdateMetadataMergesAndErrorsOnUnknownID(t *testing.T) {
	col := newTestCollection(t, 3)
	col.Add("a", []float32{1, 0, 0}, map[string]any{"lang": "en"}, strPtr("doc"))
	if err := col.UpdateMetadata("a", map[string]any{"score": 5}); err != nil {
		t.Fatalf("UpdateMetadata: %v", err)
	}
	r, err := col.Get("a")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if r.Metadata["lang"] != "en" || r.Metadata["score"] != 5 {
		t.Fatalf("unexpected merged metadata: %+v", r.Metadata)
	}
	if r.Document == nil || *r.Document != "doc" {
		t.Fatal("document should be unchanged")
	}

	if err := col.UpdateMetadata("missing", map[string]any{"x": 1}); err == nil {
		t.Fatal("expected an error for an unknown id")
	}
}

func TestCollectionVacuumReclaims(t *testing.T) {
	col := newTestCollection(t, 3)
	col.Add("a", []float32{1, 0, 0}, nil, nil)
	col.Delete("a")
	horizon := Snapshot(uint64(col.Snapshot()) + 1)
	freed := col.Vacuum(&horizon)
	if freed == 0 {
		t.Fatal("expected vacuum to reclaim at least one version")
	}
}

func TestAddRejectsUnsupportedMetadataValue(t *testing.T) {
	col := newTestCollection(t, 3)
	_, err := col.Add("a", []float32{1, 0, 0}, map[string]any{"bad": []int{1, 2}}, nil)
	if err == nil {
		t.Fatal("expected an error for an unsupported metadata value type")
	}
}

func TestAddWithNilMetadataAndDocumentCarriesForwardPrevious(t *testing.T) {
	col := newTestCollection(t, 3)
	col.Add("a", []float32{1, 0, 0}, map[string]any{"lang": "en"}, strPtr("v1"))
	// Re-add with a new vector, metadata/document both nil.
	if _, err := col.Add("a", []float32{0, 1, 0}, nil, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	r, err := col.Get("a")
	if err != nil || r == nil {
		t.Fatalf("Get: %v, %v", r, err)
	}
	if r.Metadata["lang"] != "en" {
		t.Fatalf("expected carried-forward metadata, got %+v", r.Metadata)
	}
	if r.Document == nil || *r.Document != "v1" {
		t.Fatalf("expected carried-forward document v1, got %v", r.Document)
	}
}

func TestAddWithEmptyMapReplacesMetadataDistinctFromNil(t *testing.T) {
	col := newTestCollection(t, 3)
	col.Add("a", []float32{1, 0, 0}, map[string]any{"lang": "en"}, nil)
	// An explicit, non-nil empty map is a real replace, not carry-forward.
	if _, err := col.Add("a", []float32{1, 0, 0}, map[string]any{}, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	r, _ := col.Get("a")
	if len(r.Metadata) != 0 {
		t.Fatalf("expected metadata cleared to empty, got %+v", r.Metadata)
	}
}

func TestAddCanChangeMetadataWhileCarryingForwardDocument(t *testing.T) {
	col := newTestCollection(t, 3)
	col.Add("a", []float32{1, 0, 0}, map[string]any{"lang": "en"}, strPtr("original"))
	if _, err := col.Add("a", []float32{1, 0, 0}, map[string]any{"lang": "fr"}, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	r, _ := col.Get("a")
	if r.Metadata["lang"] != "fr" {
		t.Fatalf("expected updated metadata, got %+v", r.Metadata)
	}
	if r.Document == nil || *r.Document != "original" {
		t.Fatalf("expected carried-forward document, got %v", r.Document)
	}
}

func TestAddOnNewIDWithNilMetadataIsEmptyNotAnError(t *testing.T) {
	col := newTestCollection(t, 3)
	if _, err := col.Add("a", []float32{1, 0, 0}, nil, nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	r, _ := col.Get("a")
	if len(r.Metadata) != 0 {
		t.Fatalf("expected empty metadata, got %+v", r.Metadata)
	}
	if r.Document != nil {
		t.Fatalf("expected nil document, got %v", *r.Document)
	}
}
