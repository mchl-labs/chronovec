// Ergonomic Collection type, matching Python's chronovec.Collection and
// Rust's chronovec::collection::Collection vocabulary (Add/Query/Snapshot/
// Vacuum) rather than the raw Index tier's (Insert/Search/Clock) in
// chronovec.go.
//
// Unlike the Node binding, there is no existing Rust logic to wrap here --
// this is a from-scratch implementation of the same semantics: string-or-int
// ids via stable ID hashing, versioned metadata/documents, a filter DSL
// matching the same operator set as Python's where= dict, and
// snapshot-isolated reads. It wraps this package's own Index for the vector
// storage/search, the same way the Rust and Python Collection layers wrap
// their own raw Index.
package chronovec

import (
	"encoding/binary"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"golang.org/x/crypto/blake2b"
)

// StableID maps an id onto the int64 the engine uses.
//
// Must match chronovec.collection.stable_id() (Python) and
// chronovec::collection::stable_id() (Rust) byte-for-byte: BLAKE2b, digest
// size 8, big-endian, right-shifted one bit so the top bit is always clear
// (the engine writes -1 into an empty slot, and this guarantees a hashed id
// never collides with it). A record added from one language and read from
// another must resolve to the same engine key, or the bindings are silently
// looking at different records under the same string id. An int64 id passes
// through unchanged, same as the other bindings.
func StableID(id any) (int64, error) {
	switch v := id.(type) {
	case int64:
		return v, nil
	case int:
		return int64(v), nil
	case string:
		h, err := blake2b.New(8, nil)
		if err != nil {
			// Only possible if the requested digest size is invalid, and 8
			// is always valid for BLAKE2b -- this is unreachable in practice.
			return 0, fmt.Errorf("blake2b: %w", err)
		}
		h.Write([]byte(v))
		digest := h.Sum(nil)
		return int64(binary.BigEndian.Uint64(digest) >> 1), nil
	default:
		return 0, fmt.Errorf("id must be a string or int64, got %T", id)
	}
}

// Record is one result from Query, Get, or GetWhere.
type Record struct {
	// ID is the caller's original id: string or int64, matching whichever
	// was passed to Add.
	ID any
	// Distance is nil for a Record returned by Get/GetWhere -- there was no
	// query vector.
	Distance *float32
	Metadata map[string]any
	Document *string
}

type version struct {
	snapshot Snapshot
	metadata map[string]any
	document *string
}

// QueryOptions configures Collection.Query. The zero value is "now, no
// filter, default overfetch" -- the equivalent of Python's
// col.query(v, k) with no further keyword arguments.
type QueryOptions struct {
	Snapshot *Snapshot
	// Where is a filter object with the same shape as Python's where= dict:
	// {"lang": "en"} for equality, {"year": {"$gte": 2024}} for operators,
	// {"$and": [...]} / {"$or": [...]} for boolean composition. nil means
	// no filter.
	Where map[string]any
	// Overfetch controls how many candidates past K to fetch before
	// applying Where. 0 means the package's own default, not "fetch zero".
	Overfetch int
}

// sidecar holds everything Index does not carry -- metadata, documents, and
// the mapping from a caller's id to the engine's int64 key -- behind one
// lock.
//
// Deliberately one sync.Mutex guarding all four maps rather than one per
// map: the Rust ergonomic Collection (chronovec::collection::Collection)
// originally used four independent mutexes and had a real deadlock hazard
// from it -- Query locked names before history while Vacuum's pruning
// locked history before names, two different orderings over the same two
// structures under concurrent use. One lock has no ordering to get wrong,
// and the critical sections here are small map operations, not the vector
// search itself, which stays lock-free in the underlying Index.
type sidecar struct {
	mu        sync.Mutex
	history   map[int64][]version
	names     map[int64]any
	live      map[int64]struct{}
	deletedAt map[int64]Snapshot
}

// Collection is vectors with string-or-int ids, metadata, filtering, and
// snapshots -- the same vocabulary as Python's chronovec.Collection.
type Collection struct {
	index *Index
	state sidecar
}

// NewCollection creates a collection of the given dimensionality with
// default index options (cosine metric). For custom index options, build an
// *Index with New and pass it to WrapCollection.
func NewCollection(dimensions int) (*Collection, error) {
	idx, err := New(dimensions, DefaultOptions())
	if err != nil {
		return nil, err
	}
	return WrapCollection(idx), nil
}

// WrapCollection adds the ergonomic layer over an already-built *Index.
func WrapCollection(index *Index) *Collection {
	return &Collection{
		index: index,
		state: sidecar{
			history:   make(map[int64][]version),
			names:     make(map[int64]any),
			live:      make(map[int64]struct{}),
			deletedAt: make(map[int64]Snapshot),
		},
	}
}

// Close releases the underlying Index's native resources.
func (c *Collection) Close() error { return c.index.Close() }

func (c *Collection) Dimensions() int { return c.index.Dimensions() }

func (c *Collection) Count() int {
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	return len(c.state.live)
}

// Snapshot is a token for the present, readable later via
// QueryOptions.Snapshot.
func (c *Collection) Snapshot() Snapshot { return c.index.Clock() }

// versionAtLocked returns the metadata/document in force at `at` (or now if
// at is nil) for one key. Must be called with state.mu held.
func versionAtLocked(s *sidecar, key int64, at *Snapshot) (map[string]any, *string) {
	versions, ok := s.history[key]
	if !ok || len(versions) == 0 {
		return map[string]any{}, nil
	}
	if at == nil {
		last := versions[len(versions)-1]
		return last.metadata, last.document
	}
	for i := len(versions) - 1; i >= 0; i-- {
		if versions[i].snapshot <= *at {
			return versions[i].metadata, versions[i].document
		}
	}
	first := versions[0]
	return first.metadata, first.document
}

// Add inserts or replaces one record. metadata/document may be nil, meaning
// "no metadata"/"no document" for this version -- unlike Python's
// Collection, Add always replaces rather than carrying forward the previous
// metadata/document when omitted (same tradeoff the Rust and Node bindings
// made: an explicit, non-optional replacement is less surprising in a
// statically shaped call than a hidden carry-forward).
func (c *Collection) Add(id any, vector []float32, metadata map[string]any, document *string) (Snapshot, error) {
	key, err := StableID(id)
	if err != nil {
		return 0, err
	}
	if err := validateMetadata(metadata); err != nil {
		return 0, err
	}
	stamp, err := c.index.Insert(key, vector)
	if err != nil {
		return 0, err
	}
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	// nil metadata/document means "not provided": carry forward the
	// previous version's value, matching Python's Collection.add(), which
	// carries forward when its metadatas/documents arguments are omitted
	// entirely. Pass an empty, non-nil map[string]any{} to explicitly clear
	// metadata -- Go's nil-vs-empty-map distinction already gives this for
	// free, unlike Rust and Node, which needed an Option/optional wrapper
	// added specifically for it. There is no way to explicitly clear an
	// existing document back to "no document" while keeping the metadata --
	// neither can Python's Collection.add(), whose per-item document value
	// is a plain string, not a nullable one, once the documents argument
	// itself is given.
	if metadata == nil || document == nil {
		carriedMeta, carriedDoc := versionAtLocked(&c.state, key, nil)
		if metadata == nil {
			metadata = carriedMeta
		}
		if document == nil {
			document = carriedDoc
		}
	}
	c.state.history[key] = append(c.state.history[key], version{snapshot: stamp, metadata: metadata, document: document})
	c.state.names[key] = id
	c.state.live[key] = struct{}{}
	delete(c.state.deletedAt, key)
	return stamp, nil
}

// UpdateMetadata changes an existing record's metadata without touching its
// vector, merging into (not replacing) the current metadata. Returns an
// error if id is not currently live.
func (c *Collection) UpdateMetadata(id any, metadata map[string]any) error {
	key, err := StableID(id)
	if err != nil {
		return err
	}
	if err := validateMetadata(metadata); err != nil {
		return err
	}
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	if _, ok := c.state.live[key]; !ok {
		return fmt.Errorf("unknown id: %v", id)
	}
	stamp := c.index.Clock()
	current, document := versionAtLocked(&c.state, key, nil)
	merged := make(map[string]any, len(current)+len(metadata))
	for k, v := range current {
		merged[k] = v
	}
	for k, v := range metadata {
		merged[k] = v
	}
	c.state.history[key] = append(c.state.history[key], version{snapshot: stamp, metadata: merged, document: document})
	return nil
}

// Delete removes one record. Returns whether it was live.
func (c *Collection) Delete(id any) (bool, error) {
	key, err := StableID(id)
	if err != nil {
		return false, err
	}
	c.state.mu.Lock()
	if _, ok := c.state.live[key]; !ok {
		c.state.mu.Unlock()
		return false, nil
	}
	delete(c.state.live, key)
	c.state.mu.Unlock()

	stamp, err := c.index.Delete(key)
	if err != nil {
		return false, err
	}
	c.state.mu.Lock()
	c.state.deletedAt[key] = stamp
	c.state.mu.Unlock()
	return true, nil
}

// DeleteWhere removes every live record matching where. Returns their ids.
func (c *Collection) DeleteWhere(where map[string]any) ([]any, error) {
	filter, err := parseFilter(where)
	if err != nil {
		return nil, err
	}
	type candidate struct {
		key      int64
		metadata map[string]any
	}
	c.state.mu.Lock()
	candidates := make([]candidate, 0, len(c.state.live))
	for key := range c.state.live {
		metadata, _ := versionAtLocked(&c.state, key, nil)
		candidates = append(candidates, candidate{key: key, metadata: metadata})
	}
	c.state.mu.Unlock()

	var removed []any
	for _, cand := range candidates {
		if !filter.matches(cand.metadata) {
			continue
		}
		if _, err := c.index.Delete(cand.key); err != nil {
			return removed, err
		}
		stamp := c.index.Clock()
		c.state.mu.Lock()
		delete(c.state.live, cand.key)
		c.state.deletedAt[cand.key] = stamp
		if id, ok := c.state.names[cand.key]; ok {
			removed = append(removed, id)
		}
		c.state.mu.Unlock()
	}
	return removed, nil
}

// Query returns the nearest records to query, optionally filtered and
// optionally as of a past snapshot.
func (c *Collection) Query(query []float32, k int, opts QueryOptions) ([]Record, error) {
	filter, err := parseFilter(opts.Where)
	if err != nil {
		return nil, err
	}
	overfetch := opts.Overfetch
	if overfetch == 0 {
		overfetch = 8
	}
	want := k
	if filter != nil {
		want = k * overfetch
	}

	var hits []Hit
	if opts.Snapshot != nil {
		hits, err = c.index.SearchAsOf(query, want, *opts.Snapshot)
	} else {
		hits, err = c.index.Search(query, want)
	}
	if err != nil {
		return nil, err
	}

	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	out := make([]Record, 0, k)
	for _, hit := range hits {
		metadata, document := versionAtLocked(&c.state, hit.ID, opts.Snapshot)
		if filter != nil && !filter.matches(metadata) {
			continue
		}
		id, ok := c.state.names[hit.ID]
		if !ok {
			id = hit.ID
		}
		distance := hit.Distance
		out = append(out, Record{ID: id, Distance: &distance, Metadata: metadata, Document: document})
		if len(out) == k {
			break
		}
	}
	return out, nil
}

// Get fetches one live record by id, without a vector search. Returns
// (nil, nil) if id is not live.
func (c *Collection) Get(id any) (*Record, error) {
	key, err := StableID(id)
	if err != nil {
		return nil, err
	}
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	if _, ok := c.state.live[key]; !ok {
		return nil, nil
	}
	metadata, document := versionAtLocked(&c.state, key, nil)
	return &Record{ID: id, Metadata: metadata, Document: document}, nil
}

// GetWhere fetches every live record matching where, without a vector search.
func (c *Collection) GetWhere(where map[string]any) ([]Record, error) {
	filter, err := parseFilter(where)
	if err != nil {
		return nil, err
	}
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	var out []Record
	for key := range c.state.live {
		metadata, document := versionAtLocked(&c.state, key, nil)
		if filter != nil && !filter.matches(metadata) {
			continue
		}
		id, ok := c.state.names[key]
		if !ok {
			id = key
		}
		out = append(out, Record{ID: id, Metadata: metadata, Document: document})
	}
	return out, nil
}

// Vacuum reclaims versions no reader at or after keepSnapshot can still
// reach. nil reclaims everything currently unreachable.
func (c *Collection) Vacuum(keepSnapshot *Snapshot) int {
	horizon := keepSnapshot
	if horizon == nil {
		h := Snapshot(uint64(c.index.Clock()) + 1)
		horizon = &h
	}
	freed := c.index.Vacuum(*horizon, 0)
	c.pruneHistory(*horizon)
	return freed
}

func (c *Collection) pruneHistory(horizon Snapshot) {
	c.state.mu.Lock()
	defer c.state.mu.Unlock()
	for key, versions := range c.state.history {
		keepFrom := 0
		for i, v := range versions {
			if v.snapshot <= horizon {
				keepFrom = i
			} else {
				break
			}
		}
		if keepFrom > 0 {
			c.state.history[key] = versions[keepFrom:]
		}
		if gone, ok := c.state.deletedAt[key]; ok && gone <= horizon {
			delete(c.state.history, key)
			delete(c.state.names, key)
			delete(c.state.deletedAt, key)
		}
	}
}

func validateMetadata(m map[string]any) error {
	for k, v := range m {
		switch v.(type) {
		case string, int, int64, float64, float32, bool:
		default:
			return fmt.Errorf("unsupported metadata value for %q: %T (only string, number, bool allowed)", k, v)
		}
	}
	return nil
}

// -- filter DSL --------------------------------------------------------

// filter mirrors chronovec/collection.py's matches() operator set exactly:
// $eq $ne $gt $gte $lt $lte $in $nin $contains $regex $and $or.
type filter struct {
	and []*filter // implicit AND of all top-level keys, plus $and/$or
	or  []*filter

	// A single leaf condition. field is empty for a pure and/or node.
	field string
	op    string
	value any
}

func parseFilter(where map[string]any) (*filter, error) {
	if where == nil {
		return nil, nil
	}
	parts := make([]*filter, 0, len(where))
	for key, condition := range where {
		switch key {
		case "$and":
			nested, err := parseFilterArray(condition)
			if err != nil {
				return nil, err
			}
			parts = append(parts, &filter{and: nested})
		case "$or":
			nested, err := parseFilterArray(condition)
			if err != nil {
				return nil, err
			}
			parts = append(parts, &filter{or: nested})
		default:
			sub, err := parseCondition(key, condition)
			if err != nil {
				return nil, err
			}
			parts = append(parts, sub)
		}
	}
	if len(parts) == 1 {
		return parts[0], nil
	}
	return &filter{and: parts}, nil
}

func parseFilterArray(v any) ([]*filter, error) {
	items, ok := v.([]any)
	if !ok {
		if maps, ok := v.([]map[string]any); ok {
			items = make([]any, len(maps))
			for i, m := range maps {
				items[i] = m
			}
		} else {
			return nil, errors.New("$and/$or must be an array of filter objects")
		}
	}
	out := make([]*filter, 0, len(items))
	for _, item := range items {
		m, ok := item.(map[string]any)
		if !ok {
			return nil, errors.New("$and/$or entries must be filter objects")
		}
		f, err := parseFilter(m)
		if err != nil {
			return nil, err
		}
		out = append(out, f)
	}
	return out, nil
}

func parseCondition(field string, condition any) (*filter, error) {
	ops, ok := condition.(map[string]any)
	if !ok {
		// Bare scalar: {"lang": "en"} means {"lang": {"$eq": "en"}}.
		return &filter{field: field, op: "$eq", value: condition}, nil
	}
	parts := make([]*filter, 0, len(ops))
	for op, want := range ops {
		switch op {
		case "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$contains", "$regex":
			parts = append(parts, &filter{field: field, op: op, value: want})
		default:
			return nil, fmt.Errorf("unknown operator %q", op)
		}
	}
	if len(parts) == 0 {
		return nil, fmt.Errorf("empty operator object for key %q", field)
	}
	if len(parts) == 1 {
		return parts[0], nil
	}
	return &filter{and: parts}, nil
}

func (f *filter) matches(metadata map[string]any) bool {
	if f == nil {
		return true
	}
	if f.and != nil {
		for _, part := range f.and {
			if !part.matches(metadata) {
				return false
			}
		}
		return true
	}
	if f.or != nil {
		for _, part := range f.or {
			if part.matches(metadata) {
				return true
			}
		}
		return false
	}
	have, present := metadata[f.field]
	switch f.op {
	case "$eq":
		return present && valuesEqual(have, f.value)
	case "$ne":
		return !present || !valuesEqual(have, f.value)
	case "$gt":
		cmp, ok := compareValues(have, f.value)
		return present && ok && cmp > 0
	case "$gte":
		cmp, ok := compareValues(have, f.value)
		return present && ok && cmp >= 0
	case "$lt":
		cmp, ok := compareValues(have, f.value)
		return present && ok && cmp < 0
	case "$lte":
		cmp, ok := compareValues(have, f.value)
		return present && ok && cmp <= 0
	case "$in":
		return present && valueIn(have, f.value)
	case "$nin":
		return !present || !valueIn(have, f.value)
	case "$contains":
		haveStr, ok1 := have.(string)
		wantStr, ok2 := f.value.(string)
		return present && ok1 && ok2 && strings.Contains(haveStr, wantStr)
	case "$regex":
		haveStr, ok1 := have.(string)
		pattern, ok2 := f.value.(string)
		if !present || !ok1 || !ok2 {
			return false
		}
		re, err := regexp.Compile(pattern)
		if err != nil {
			return false
		}
		return re.MatchString(haveStr)
	default:
		return false
	}
}

func valuesEqual(a, b any) bool {
	an, aok := toFloat64(a)
	bn, bok := toFloat64(b)
	if aok && bok {
		return an == bn
	}
	return a == b
}

func valueIn(have any, want any) bool {
	items, ok := want.([]any)
	if !ok {
		return false
	}
	for _, item := range items {
		if valuesEqual(have, item) {
			return true
		}
	}
	return false
}

// compareValues orders two metadata values. Numbers compare numerically
// (int/int64/float64/float32 all coerced), strings lexicographically;
// comparing across those two families, or a bool, reports ok=false --
// the same "doesn't match" a Python operator raising on incompatible types
// would produce for a real caller.
func compareValues(a, b any) (cmp int, ok bool) {
	if an, aok := toFloat64(a); aok {
		if bn, bok := toFloat64(b); bok {
			switch {
			case an < bn:
				return -1, true
			case an > bn:
				return 1, true
			default:
				return 0, true
			}
		}
		return 0, false
	}
	if as, aok := a.(string); aok {
		if bs, bok := b.(string); bok {
			switch {
			case as < bs:
				return -1, true
			case as > bs:
				return 1, true
			default:
				return 0, true
			}
		}
	}
	return 0, false
}

func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case float64:
		return n, true
	case float32:
		return float64(n), true
	default:
		return 0, false
	}
}
