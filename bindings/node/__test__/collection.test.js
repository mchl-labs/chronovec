"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { Collection } = require("../index.js");

test("add and query finds a record by string id", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]));
  col.add("b", new Float32Array([0, 1, 0]));
  const hits = col.query(new Float32Array([1, 0, 0]), 1);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].id, "a");
});

test("int ids pass through as bigint", () => {
  const col = new Collection(3);
  col.add(42n, new Float32Array([1, 0, 0]));
  const hits = col.query(new Float32Array([1, 0, 0]), 1);
  assert.equal(hits[0].id, 42n);
});

test("metadata and document round-trip", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { lang: "en", year: 2024 }, "hello");
  const hits = col.query(new Float32Array([1, 0, 0]), 1);
  assert.deepEqual(hits[0].metadata, { lang: "en", year: 2024 });
  assert.equal(hits[0].document, "hello");
});

test("snapshot isolation: query as of a past snapshot sees the old version", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { status: "draft" }, "v1");
  const before = col.snapshot();
  col.add("a", new Float32Array([1, 0, 0]), { status: "final" }, "v2");

  const now = col.query(new Float32Array([1, 0, 0]), 1);
  assert.equal(now[0].document, "v2");

  const then = col.query(new Float32Array([1, 0, 0]), 1, { snapshot: before });
  assert.equal(then[0].document, "v1");
});

test("where filter: plain equality", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { lang: "en" });
  col.add("b", new Float32Array([1, 0, 0]), { lang: "fr" });
  const hits = col.query(new Float32Array([1, 0, 0]), 5, { where: { lang: "fr" } });
  assert.equal(hits.length, 1);
  assert.equal(hits[0].id, "b");
});

test("where filter: comparison operators", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { year: 2020 });
  col.add("b", new Float32Array([1, 0, 0]), { year: 2024 });
  const hits = col.query(new Float32Array([1, 0, 0]), 5, {
    where: { year: { $gte: 2024 } },
  });
  assert.equal(hits.length, 1);
  assert.equal(hits[0].id, "b");
});

test("where filter: $and / $or composition", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { lang: "en", year: 2024 });
  col.add("b", new Float32Array([1, 0, 0]), { lang: "en", year: 2020 });
  col.add("c", new Float32Array([1, 0, 0]), { lang: "fr", year: 2024 });

  const andHits = col.query(new Float32Array([1, 0, 0]), 5, {
    where: { $and: [{ lang: "en" }, { year: { $gte: 2024 } }] },
  });
  assert.deepEqual(
    andHits.map((h) => h.id),
    ["a"],
  );

  const orHits = col
    .query(new Float32Array([1, 0, 0]), 5, {
      where: { $or: [{ lang: "fr" }, { year: 2020 }] },
    })
    .map((h) => h.id)
    .sort();
  assert.deepEqual(orHits, ["b", "c"]);
});

test("where filter: $in / $contains / $regex", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { title: "ChronoVec MVCC" });
  col.add("b", new Float32Array([1, 0, 0]), { title: "other" });

  const inHits = col.query(new Float32Array([1, 0, 0]), 5, {
    where: { title: { $in: ["ChronoVec MVCC", "nope"] } },
  });
  assert.equal(inHits.length, 1);

  const containsHits = col.query(new Float32Array([1, 0, 0]), 5, {
    where: { title: { $contains: "MVCC" } },
  });
  assert.equal(containsHits.length, 1);

  const regexHits = col.query(new Float32Array([1, 0, 0]), 5, {
    where: { title: { $regex: "^Chrono.*" } },
  });
  assert.equal(regexHits.length, 1);
  assert.equal(regexHits[0].id, "a");
});

test("delete removes a record; deleteWhere removes matching records", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { kind: "hyp" });
  col.add("b", new Float32Array([0, 1, 0]), { kind: "fact" });
  assert.equal(col.count(), 2);

  const removed = col.deleteWhere({ kind: "hyp" });
  assert.deepEqual(removed, ["a"]);
  assert.equal(col.count(), 1);

  assert.equal(col.delete("b"), true);
  assert.equal(col.delete("b"), false);
  assert.equal(col.count(), 0);
});

test("get and getWhere fetch without a vector search", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { lang: "en" });
  col.add("b", new Float32Array([0, 1, 0]), { lang: "fr" });

  const one = col.get("a");
  assert.equal(one.document, undefined);
  assert.equal(one.id, "a");
  assert.equal(col.get("missing"), null);

  const many = col.getWhere({ lang: "fr" });
  assert.equal(many.length, 1);
  assert.equal(many[0].id, "b");
});

test("updateMetadata merges and throws on unknown id", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { lang: "en" }, "doc");
  col.updateMetadata("a", { score: 5 });
  const r = col.get("a");
  assert.deepEqual(r.metadata, { lang: "en", score: 5 });
  assert.equal(r.document, "doc");

  assert.throws(() => col.updateMetadata("missing", { x: 1 }));
});

test("vacuum reclaims a deleted record's space", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]));
  col.delete("a");
  const horizon = col.snapshot() + 1n;
  const freed = col.vacuum(horizon);
  assert.ok(freed > 0, "expected vacuum to reclaim at least one version");
});

test("ids beyond Number.MAX_SAFE_INTEGER round-trip exactly", () => {
  const col = new Collection(3);
  const bigId = 9007199254740993n;
  col.add(bigId, new Float32Array([1, 0, 0]));
  const hits = col.query(new Float32Array([1, 0, 0]), 1);
  assert.equal(hits[0].id, bigId);
});

test("add with omitted metadata/document carries forward the previous version", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { lang: "en" }, "v1");
  // Re-add with a new vector, metadata/document omitted entirely.
  col.add("a", new Float32Array([0, 1, 0]));
  const r = col.get("a");
  assert.deepEqual(r.metadata, { lang: "en" });
  assert.equal(r.document, "v1");
});

test("add with an explicit empty object replaces metadata, distinct from omitting it", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { lang: "en" });
  col.add("a", new Float32Array([1, 0, 0]), {}); // explicit replace, not carry-forward
  const r = col.get("a");
  assert.deepEqual(r.metadata, {});
});

test("add can change metadata while carrying forward the document", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]), { lang: "en" }, "original");
  col.add("a", new Float32Array([1, 0, 0]), { lang: "fr" }); // document omitted
  const r = col.get("a");
  assert.deepEqual(r.metadata, { lang: "fr" });
  assert.equal(r.document, "original");
});

test("add on a brand-new id with everything omitted is empty, not an error", () => {
  const col = new Collection(3);
  col.add("a", new Float32Array([1, 0, 0]));
  const r = col.get("a");
  assert.deepEqual(r.metadata, {});
  assert.equal(r.document, undefined);
});
