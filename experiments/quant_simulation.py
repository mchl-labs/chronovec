"""End-to-end simulation of the proposed ChronoVec v2 search path.

Pipeline per query: rank pages by centroid distance -> gather candidates from
the top-P pages -> estimate distances from compressed codes -> rerank the best
`budget` candidates with exact float32 -> Recall@10.

This measures the only thing that matters for the v2 bet: whether compressed
codes preserve enough ranking fidelity that a *small* exact-rerank budget
recovers the true neighbours. Candidate-set recall is the ceiling; the question
is how much of that ceiling the codes give back, and at how many bytes.

Codes modelled
  fp32   : uncompressed control (4*d bytes)
  rabitq : RaBitQ 1-bit (SIGMOD 2024) -- random rotation of the unit residual
           against the cell centroid, sign bits, plus per-vector residual norm
           and <x_bar, u'> correction factor. d/8 + 8 bytes.
  sqB    : B-bit uniform scalar quantisation of the rotated unit residual, a
           stand-in for Extended RaBitQ at B bits/dim. d*B/8 + 8 bytes.
           Approximates the multi-bit regime; it is not the exact codebook.
Estimators are evaluated in float; this isolates *accuracy*. Throughput comes
from the byte counts, which are reported alongside.
"""

import argparse, json, time
import numpy as np
import h5py


def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def sqdist(a, b, angular):
    if angular:
        return 2.0 - 2.0 * (a @ b.T)
    return (np.sum(a * a, axis=1)[:, None] - 2.0 * (a @ b.T)
            + np.sum(b * b, axis=1)[None, :])


def load(path, limit, queries, k, angular, seed=42):
    with h5py.File(path, "r") as h:
        full_n = h["train"].shape[0]
        train = np.asarray(h["train"][:limit] if limit else h["train"][:], dtype=np.float32)
        test = np.asarray(h["test"][:], dtype=np.float32)
        truth = np.asarray(h["neighbors"][:], dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(test.shape[0], size=min(queries, test.shape[0]), replace=False)
    idx.sort()
    test, truth = test[idx], truth[idx]
    if angular:
        train, test = normalize(train), normalize(test)
    if limit and limit < full_n:
        truth = np.empty((test.shape[0], k), dtype=np.int64)
        for s in range(0, test.shape[0], 256):
            e = min(s + 256, test.shape[0])
            dd = sqdist(test[s:e], train, angular)
            cols = np.argpartition(dd, k, axis=1)[:, :k]
            rows = np.arange(e - s)[:, None]
            truth[s:e] = cols[rows, np.argsort(dd[rows, cols], axis=1)]
        print(f"  recomputed exact truth for {limit}-vector subset", flush=True)
    return train, test, truth[:, :k]


def kmeans(train, k, angular, iters, seed=42, chunk=8192):
    rng = np.random.default_rng(seed)
    cents = train[rng.choice(train.shape[0], size=k, replace=False)].copy()
    if angular:
        cents = normalize(cents)
    labels = None
    for it in range(iters):
        labels = np.empty(train.shape[0], dtype=np.int32)
        for s in range(0, train.shape[0], chunk):
            e = min(s + chunk, train.shape[0])
            labels[s:e] = np.argmin(sqdist(train[s:e], cents, angular), axis=1)
        sums = np.zeros((k, train.shape[1]), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)
        np.add.at(counts, labels, 1)
        for s in range(0, train.shape[0], chunk):
            e = min(s + chunk, train.shape[0])
            np.add.at(sums, labels[s:e], train[s:e].astype(np.float64))
        empty = counts == 0
        counts[empty] = 1
        cents = (sums / counts[:, None]).astype(np.float32)
        if empty.any():
            cents[empty] = train[rng.choice(train.shape[0], size=int(empty.sum()))]
        if angular:
            cents = normalize(cents)
        print(f"    kmeans iter {it+1}/{iters}", flush=True)
    for s in range(0, train.shape[0], chunk):
        e = min(s + chunk, train.shape[0])
        labels[s:e] = np.argmin(sqdist(train[s:e], cents, angular), axis=1)
    return cents, labels


def random_rotation(d, seed=7):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


def encode(train, cents, labels, rot, mode, bits):
    """Per-vector code data. Returns dict of arrays plus bytes/vector."""
    d = train.shape[1]
    resid = train - cents[labels]
    rn = np.linalg.norm(resid, axis=1).astype(np.float32)
    safe = np.maximum(rn, 1e-12)
    u = (resid / safe[:, None]).astype(np.float32)
    up = u @ rot.T                                  # rotated unit residual
    out = {"resid_norm": rn}
    if mode == "fp32":
        out["vec"] = up
        return out, 4 * d
    if mode == "rabitq":
        sign = np.where(up >= 0, 1.0, -1.0).astype(np.float32)
        xbar = sign / np.sqrt(d)
        out["code"] = xbar
        out["factor"] = np.einsum('ij,ij->i', xbar, up).astype(np.float32)
        return out, d // 8 + 8
    if mode == "sq":
        lo = up.min(axis=1, keepdims=True)
        hi = up.max(axis=1, keepdims=True)
        step = np.maximum((hi - lo) / (2 ** bits - 1), 1e-12)
        q = np.rint((up - lo) / step)
        deq = (q * step + lo).astype(np.float32)
        nrm = np.linalg.norm(deq, axis=1)
        deq = deq / np.maximum(nrm, 1e-12)[:, None]
        out["code"] = deq
        out["factor"] = np.einsum('ij,ij->i', deq, up).astype(np.float32)
        return out, (d * bits) // 8 + 8
    raise ValueError(mode)


def estimate(codes, mode, idx, qres_unit, qn, page_of):
    """Estimated squared distance from query to each candidate.

    Each candidate's residual is taken against its own page centroid, so the
    query residual is gathered per candidate via page_of."""
    rn = codes["resid_norm"][idx]
    qru = qres_unit[page_of]                     # (m, d) unit query residual
    qrn = qn[page_of]                            # (m,)   query residual norm
    if mode == "fp32":
        ip = np.einsum('ij,ij->i', codes["vec"][idx], qru)
    else:
        ip = np.einsum('ij,ij->i', codes["code"][idx], qru) / np.maximum(
            codes["factor"][idx], 1e-12)
    return rn * rn + qrn * qrn - 2.0 * rn * qrn * ip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--page-capacity", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--angular", action="store_true")
    ap.add_argument("--kmeans-iters", type=int, default=10)
    ap.add_argument("--probes", type=int, nargs="+", default=[32, 64, 128, 192])
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[50, 100, 200, 400, 800, 1600])
    ap.add_argument("--modes", nargs="+", default=["fp32", "rabitq", "sq3", "sq5"])
    ap.add_argument("--no-rotation", action="store_true",
                    help="use identity instead of a random rotation")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    train, queries, truth = load(a.dataset, a.limit, a.queries, a.k, a.angular)
    n, d = train.shape
    print(f"train={train.shape} queries={queries.shape}", flush=True)
    npages = max(1, n // a.page_capacity)
    t0 = time.time()
    cents, labels = kmeans(train, npages, a.angular, a.kmeans_iters)
    print(f"pages={npages} kmeans {time.time()-t0:.1f}s", flush=True)

    members = [[] for _ in range(npages)]
    for i, p in enumerate(labels):
        members[int(p)].append(i)
    members = [np.asarray(m, dtype=np.int64) for m in members]

    rot = np.eye(d, dtype=np.float32) if a.no_rotation else random_rotation(d)
    print(f'rotation: {"identity" if a.no_rotation else "random orthogonal"}', flush=True)
    encoded = {}
    for mode in a.modes:
        base, bits = ("sq", int(mode[2:])) if mode.startswith("sq") else (mode, 0)
        t1 = time.time()
        codes, nbytes = encode(train, cents, labels, rot, base, bits)
        encoded[mode] = (codes, nbytes, base, bits)
        print(f"  encoded {mode}: {nbytes} B/vector in {time.time()-t1:.1f}s", flush=True)

    results = []
    for P in a.probes:
        stats = {m: {b: 0.0 for b in a.budgets} for m in a.modes}
        ceiling = 0.0
        cand_total = 0
        for qi in range(queries.shape[0]):
            q = queries[qi]
            cd = sqdist(q[None, :], cents, a.angular)[0]
            top = np.argsort(cd)[:P]
            idx = np.concatenate([members[p] for p in top]) if len(top) else np.empty(0, np.int64)
            if idx.size == 0:
                continue
            cand_total += idx.size
            page_of_local = np.concatenate(
                [np.full(members[p].shape[0], j, dtype=np.int64) for j, p in enumerate(top)])
            tset = set(int(x) for x in truth[qi])
            ceiling += len(tset & set(idx.tolist())) / a.k
            qres = q[None, :] - cents[top]
            qn = np.linalg.norm(qres, axis=1).astype(np.float32)
            qru = ((qres / np.maximum(qn, 1e-12)[:, None]) @ rot.T).astype(np.float32)
            for mode in a.modes:
                codes, _, base, _ = encoded[mode]
                est = estimate(codes, base, idx, qru, qn, page_of_local)
                order = np.argsort(est)
                for b in a.budgets:
                    sel = idx[order[:b]]
                    exact = sqdist(q[None, :], train[sel], a.angular)[0]
                    keep = sel[np.argsort(exact)[:a.k]]
                    stats[mode][b] += len(tset & set(keep.tolist())) / a.k
        nq = queries.shape[0]
        row = {"probes": P, "mean_candidates": cand_total / nq,
               "candidate_ceiling_recall": ceiling / nq,
               "modes": {m: {"bytes_per_vector": encoded[m][1],
                             "recall": {str(b): stats[m][b] / nq for b in a.budgets}}
                         for m in a.modes}}
        results.append(row)
        print(f"P={P:4d} cands={row['mean_candidates']:8.0f} "
              f"ceiling={row['candidate_ceiling_recall']:.4f}", flush=True)
        for m in a.modes:
            rr = row["modes"][m]["recall"]
            print(f"    {m:7s} {encoded[m][1]:4d}B  " +
                  "  ".join(f"b{b}={rr[str(b)]:.4f}" for b in a.budgets), flush=True)

    with open(a.output, "w") as f:
        json.dump({"dataset": a.dataset, "vectors": int(n), "dimensions": int(d),
                   "queries": int(queries.shape[0]), "pages": npages,
                   "page_capacity": a.page_capacity, "k": a.k,
                   "angular": a.angular, "budgets": a.budgets,
                   "results": results}, f, indent=1)
    print(f"wrote {a.output}", flush=True)


if __name__ == "__main__":
    main()
