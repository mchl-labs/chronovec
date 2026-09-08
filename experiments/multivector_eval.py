"""Quality and cost of MUVERA-style FDE retrieval against exact Chamfer.

The task must be well-posed before the encoder can be judged: exact Chamfer has
to rank the intended document first, or "recall" measures tie-breaking noise
rather than retrieval. Documents therefore have real internal structure -- tokens
concentrated around a document topic, mixed with a shared vocabulary that every
document draws from, which is roughly how contextual token embeddings behave.

Reported: whether exact Chamfer finds the target (task validity), whether the
FDE shortlist contains it (the part MUVERA is responsible for), and whether the
full pipeline ranks it first (what a user sees).
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from chronovec.multivector import MultiVectorIndex, FixedDimensionalEncoder, chamfer


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def corpus(rng, n_docs, dim, tokens, shared_fraction=0.3, spread=0.25):
    shared = unit(rng.standard_normal((64, dim)).astype(np.float32))
    docs, topics = {}, unit(rng.standard_normal((n_docs, dim)).astype(np.float32))
    for i in range(n_docs):
        n_shared = int(tokens * shared_fraction)
        specific = topics[i] + spread * rng.standard_normal((tokens - n_shared, dim)).astype(np.float32)
        picked = shared[rng.choice(len(shared), n_shared, replace=False)]
        docs[i] = unit(np.vstack([specific, picked]).astype(np.float32))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=5000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--query-tokens", type=int, default=8)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--output", default="")
    a = ap.parse_args()

    rng = np.random.default_rng(3)
    docs = corpus(rng, a.docs, a.dim, a.tokens)
    targets = rng.choice(a.docs, a.queries, replace=False)
    queries = [unit(docs[t][rng.choice(a.tokens, a.query_tokens, replace=False)]
                    + a.noise * rng.standard_normal((a.query_tokens, a.dim)).astype(np.float32))
               for t in targets]

    keys = list(docs)
    matrix = [docs[i] for i in keys]
    exact_rank, brute_ms = [], 0.0
    t0 = time.perf_counter()
    for q, target in zip(queries, targets):
        scores = np.array([chamfer(q, d) for d in matrix])
        order = np.argsort(-scores)
        exact_rank.append(int(np.nonzero(np.array(keys)[order] == target)[0][0]))
    brute_ms = (time.perf_counter() - t0) / a.queries * 1000
    top1 = sum(r == 0 for r in exact_rank)
    print(f"task validity: exact Chamfer ranks the target first in {top1}/{a.queries} "
          f"queries ({brute_ms:.2f} ms/query brute force over {a.docs} docs)")
    if top1 < 0.8 * a.queries:
        print("  WARNING: the task is not well-posed; encoder numbers below are not meaningful")

    print(f"\n{'k_sim':>5} {'reps':>5} {'proj':>5} {'fde_dim':>8} {'shortlist@50':>13} "
          f"{'shortlist@200':>14} {'top1':>6} {'ms':>7} {'speedup':>8}")
    results = []
    for k_sim, reps, proj in ((3, 4, 8), (4, 8, 8), (4, 8, 16), (5, 8, 16), (4, 16, 16)):
        enc = FixedDimensionalEncoder(a.dim, k_sim=k_sim, repetitions=reps,
                                      projection_dim=proj, seed=1)
        ix = MultiVectorIndex(a.dim, encoder=enc, nprobe=96)
        for i, toks in docs.items():
            ix.add(i, toks)
        found = {}
        for cand in (50, 200):
            found[cand] = sum(
                target in {h.doc_id for h in ix.search(q, k=cand, candidates=cand)}
                for q, target in zip(queries, targets)) / a.queries
        t0 = time.perf_counter()
        top1_hits = sum(ix.search(q, k=1, candidates=200)[0].doc_id == target
                        for q, target in zip(queries, targets))
        ms = (time.perf_counter() - t0) / a.queries * 1000
        print(f"{k_sim:>5} {reps:>5} {proj:>5} {enc.fde_dimensions:>8} {found[50]:>13.3f} "
              f"{found[200]:>14.3f} {top1_hits/a.queries:>6.3f} {ms:>7.3f} {brute_ms/ms:>7.1f}x")
        results.append({"k_sim": k_sim, "repetitions": reps, "projection_dim": proj,
                        "fde_dimensions": enc.fde_dimensions,
                        "target_in_shortlist_50": found[50],
                        "target_in_shortlist_200": found[200],
                        "pipeline_top1": top1_hits / a.queries,
                        "ms_per_query": ms, "speedup_vs_brute": brute_ms / ms})
        ix.close()
    if a.output:
        json.dump({"config": vars(a), "exact_top1": top1 / a.queries,
                   "brute_ms": brute_ms, "variants": results},
                  open(a.output, "w"), indent=1)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
