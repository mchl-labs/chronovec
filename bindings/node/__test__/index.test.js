"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const os = require("node:os");
const path = require("node:path");
const fs = require("node:fs");
const { Index } = require("../index.js");

function newIndex(dims = 3, opts = {}) {
  return new Index(dims, opts);
}

test("insert and search finds the nearest vector", () => {
  const idx = newIndex();
  idx.insert(1n, new Float32Array([1, 0, 0]));
  idx.insert(2n, new Float32Array([0, 1, 0]));
  const hits = idx.search(new Float32Array([1, 0, 0]), 1);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].id, 1n);
});

test("delete removes a record from search results", () => {
  const idx = newIndex();
  idx.insert(1n, new Float32Array([1, 0, 0]));
  idx.delete(1n);
  const hits = idx.search(new Float32Array([1, 0, 0]), 5);
  assert.equal(hits.length, 0);
});

test("searchAsOf sees a record that has since been deleted", () => {
  const idx = newIndex();
  idx.insert(1n, new Float32Array([1, 0, 0]));
  const before = idx.clock();
  idx.delete(1n);

  const now = idx.search(new Float32Array([1, 0, 0]), 5);
  assert.equal(now.length, 0);

  const then = idx.searchAsOf(new Float32Array([1, 0, 0]), 5, before);
  assert.equal(then.length, 1);
  assert.equal(then[0].id, 1n);
});

test("insert with wrong vector length throws", () => {
  const idx = newIndex();
  assert.throws(() => idx.insert(1n, new Float32Array([1, 0])));
});

test("ids beyond Number.MAX_SAFE_INTEGER round-trip exactly", () => {
  // The real correctness case: stable_id() (Rust/Python Collection layers)
  // hashes strings well past 2^53. A number-based id boundary would
  // silently corrupt this; BigInt must not.
  const idx = newIndex();
  const bigId = 9007199254740993n; // MAX_SAFE_INTEGER + 2
  idx.insert(bigId, new Float32Array([1, 0, 0]));
  const hits = idx.search(new Float32Array([1, 0, 0]), 1);
  assert.equal(hits[0].id, bigId);
});

test("vacuum reclaims a deleted record's space", () => {
  const idx = newIndex();
  idx.insert(1n, new Float32Array([1, 0, 0]));
  idx.delete(1n);
  const horizon = idx.clock() + 1n;
  const freed = idx.vacuum(horizon, 64);
  assert.ok(freed > 0, "expected vacuum to reclaim at least one version");
});

test("save and load round-trips an index", () => {
  const idx = newIndex();
  idx.insert(1n, new Float32Array([1, 0, 0]));
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "chronovec-node-"));
  const file = path.join(dir, "index.cvec");
  idx.save(file);

  const restored = Index.load(file);
  assert.equal(restored.dimensions(), 3);
  const hits = restored.search(new Float32Array([1, 0, 0]), 1);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].id, 1n);
});

test("stats reports live vector count", () => {
  const idx = newIndex();
  idx.insert(1n, new Float32Array([1, 0, 0]));
  idx.insert(2n, new Float32Array([0, 1, 0]));
  const stats = idx.stats();
  assert.equal(stats.liveVectors, 2);
});

test("an unknown metric string is rejected", () => {
  assert.throws(() => new Index(3, { metric: "manhattan" }));
});

test("l2 metric is accepted and usable", () => {
  const idx = newIndex(3, { metric: "l2" });
  idx.insert(1n, new Float32Array([1, 0, 0]));
  const hits = idx.search(new Float32Array([1, 0, 0]), 1);
  assert.equal(hits.length, 1);
});
