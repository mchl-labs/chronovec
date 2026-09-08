"""Offline falsification of the ChronoVec v2 overlap thesis.

Question: how many candidate vectors must be scanned to reach a target
Recall@10, as a function of (a) partition quality and (b) bounded overlap?

Method. With exact reranking, a true neighbour is recovered if and only if it
is present in the probed candidate set. Recall@k therefore reduces to set
membership, so the whole curve can be computed without any distance work at
rerank time. For each query we rank pages by centroid distance, and for each
true neighbour record the best (lowest) rank of any page holding a copy of it.
Recall at P probes is then the fraction of neighbours whose best rank is < P,
and candidates at P is the summed size of the top-P pages, counting replicas
because a replica costs a scan slot even though it dedupes at rerank.
"""

import argparse, json, time
import numpy as np
import h5py


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
    if limit and limit < full_n:
        # published neighbours index the full corpus; recompute exact truth
        tr = normalize(train) if angular else train
        te = normalize(test) if angular else test
        truth = np.empty((te.shape[0], k), dtype=np.int64)
        for s0 in range(0, te.shape[0], 256):
            e0 = min(s0 + 256, te.shape[0])
            d = distances(te[s0:e0], tr, angular)
            truth[s0:e0] = np.argpartition(d, k, axis=1)[:, :k]
            rows = np.arange(e0 - s0)[:, None]
            order = np.argsort(d[rows, truth[s0:e0]], axis=1)
            truth[s0:e0] = truth[s0:e0][rows, order]
        print(f"  recomputed exact ground truth for {limit}-vector subset", flush=True)
    return train, test, truth[:, :k]


def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def distances(a, b, angular):
    """Pairwise distance, chunked. a:(n,d) b:(m,d) -> (n,m)."""
    if angular:
        return 1.0 - a @ b.T
    return (np.sum(a * a, axis=1)[:, None] - 2.0 * (a @ b.T)
            + np.sum(b * b, axis=1)[None, :])


def assign_nearest(train, centroids, angular, chunk=8192):
    out = np.empty(train.shape[0], dtype=np.int32)
    for s in range(0, train.shape[0], chunk):
        e = min(s + chunk, train.shape[0])
        out[s:e] = np.argmin(distances(train[s:e], centroids, angular), axis=1)
    return out


def kmeans(train, k, angular, iters, seed=42, chunk=8192):
    rng = np.random.default_rng(seed)
    centroids = train[rng.choice(train.shape[0], size=k, replace=False)].copy()
    if angular:
        centroids = normalize(centroids)
    for it in range(iters):
        labels = assign_nearest(train, centroids, angular, chunk)
        sums = np.zeros((k, train.shape[1]), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)
        np.add.at(counts, labels, 1)
        for s in range(0, train.shape[0], chunk):
            e = min(s + chunk, train.shape[0])
            np.add.at(sums, labels[s:e], train[s:e].astype(np.float64))
        empty = counts == 0
        counts[empty] = 1
        centroids = (sums / counts[:, None]).astype(np.float32)
        if empty.any():
            centroids[empty] = train[rng.choice(train.shape[0], size=int(empty.sum()))]
        if angular:
            centroids = normalize(centroids)
        print(f"    kmeans iter {it+1}/{iters}", flush=True)
    return centroids, assign_nearest(train, centroids, angular, chunk)


def greedy_incremental(train, capacity, angular, split_iters=4):
    """Faithful replica of ChronoVec choose_page + local two-way split."""
    d = train.shape[1]
    cents = np.zeros((1, d), dtype=np.float32)
    members = [[]]
    sums = [np.zeros(d, dtype=np.float64)]
    n = train.shape[0]
    for i in range(n):
        v = train[i]
        if angular:
            dist = 1.0 - cents @ v
        else:
            dist = np.sum((cents - v) ** 2, axis=1)
        p = int(np.argmin(dist))
        if len(members[p]) >= capacity:
            # local two-way split, seeded by farthest point from member[0]
            idxs = np.asarray(members[p], dtype=np.int64)
            pts = train[idxs]
            c0 = pts[0].astype(np.float32).copy()
            dd = (1.0 - pts @ c0) if angular else np.sum((pts - c0) ** 2, axis=1)
            c1 = pts[int(np.argmax(dd))].astype(np.float32).copy()
            for _ in range(split_iters):
                d0 = (1.0 - pts @ c0) if angular else np.sum((pts - c0) ** 2, axis=1)
                d1 = (1.0 - pts @ c1) if angular else np.sum((pts - c1) ** 2, axis=1)
                lab = d1 < d0
                if lab.all():
                    lab[0] = False
                if not lab.any():
                    lab[-1] = True
                c0 = pts[~lab].mean(axis=0).astype(np.float32)
                c1 = pts[lab].mean(axis=0).astype(np.float32)
                if angular:
                    c0 /= max(np.linalg.norm(c0), 1e-12)
                    c1 /= max(np.linalg.norm(c1), 1e-12)
            left = idxs[~lab].tolist()
            right = idxs[lab].tolist()
            members[p] = left
            sums[p] = train[np.asarray(left)].astype(np.float64).sum(axis=0)
            cents[p] = c0
            members.append(right)
            sums.append(train[np.asarray(right)].astype(np.float64).sum(axis=0))
            cents = np.vstack([cents, c1[None, :]])
            # re-route the current vector after the split
            if angular:
                dist = 1.0 - cents @ v
            else:
                dist = np.sum((cents - v) ** 2, axis=1)
            p = int(np.argmin(dist))
        members[p].append(i)
        sums[p] += v
        c = (sums[p] / len(members[p])).astype(np.float32)
        if angular:
            c /= max(np.linalg.norm(c), 1e-12)
        cents[p] = c
        if (i + 1) % 100000 == 0:
            print(f"    greedy inserted {i+1}/{n}, pages={len(members)}", flush=True)
    labels = np.empty(n, dtype=np.int32)
    for pi, ms in enumerate(members):
        labels[np.asarray(ms, dtype=np.int64)] = pi
    return cents, labels


def boundary_scores(train, centroids, angular, max_replicas, chunk=4096):
    """For each vector, its top-R candidate pages and the distance from the
    vector to each Voronoi bisector separating that page from its primary.

    For the bisector between primary c0 and rival cp,
        dist_to_plane = (||v-cp||^2 - ||v-c0||^2) / (2 * ||cp-c0||)
    which is a true length in the embedding space and therefore does not
    collapse under the distance concentration that makes ratio margins useless
    in high dimension. Unit-normalised angular data uses the same formula since
    ||v-c||^2 = 2 - 2<v,c>.
    """
    n, k = train.shape[0], centroids.shape[0]
    take = min(max_replicas, k)
    cand = np.empty((n, take), dtype=np.int32)
    plane = np.empty((n, take), dtype=np.float32)
    cn = np.sum(centroids * centroids, axis=1)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sq = distances(train[s:e], centroids, angular)  # monotone in ||v-c||^2
        part = np.argpartition(sq, take - 1, axis=1)[:, :take]
        rows = np.arange(e - s)[:, None]
        order = np.argsort(sq[rows, part], axis=1)
        part = part[rows, order]
        cand[s:e] = part
        d2 = sq[rows, part]
        if angular:
            d2 = 2.0 * d2  # 1-dot -> 2-2dot
        prim = centroids[part[:, 0]]
        sep = np.sqrt(np.maximum(
            np.sum(prim * prim, axis=1)[:, None] - 2.0 * np.einsum('ij,ikj->ik', prim, centroids[part])
            + cn[part], 1e-12))
        plane[s:e] = (d2 - d2[:, :1]) / (2.0 * sep)
        plane[s:e, 0] = 0.0
    return cand, plane


def pages_at_overlap(cand, plane, n_pages, target_overlap):
    """Admit the globally most boundary-proximate replicas until the overlap
    budget is spent. Threshold is a quantile over pooled plane distances, so the
    budget is a hard resource rather than a tuning artefact."""
    n = cand.shape[0]
    pages = [[] for _ in range(n_pages)]
    if target_overlap <= 1.0 or cand.shape[1] < 2:
        for i in range(n):
            pages[int(cand[i, 0])].append(i)
        return pages, 1.0, float('nan')
    extra = plane[:, 1:].ravel()
    budget = int(round((target_overlap - 1.0) * n))
    budget = min(budget, extra.size)
    thresh = float(np.partition(extra, budget - 1)[budget - 1]) if budget > 0 else -1.0
    keep = plane <= thresh
    keep[:, 0] = True
    total = 0
    rows, cols = np.nonzero(keep)
    for r, c in zip(rows.tolist(), cols.tolist()):
        pages[int(cand[r, c])].append(r)
        total += 1
    return pages, total / float(n), thresh


def evaluate(queries, centroids, pages, truth, angular, k, probe_grid):
    """Best page-rank per true neighbour -> (candidates, recall) curve."""
    n_pages = centroids.shape[0]
    # only the true neighbours of the evaluated queries need owner lists
    needed = set(int(x) for x in truth[:, :k].ravel())
    owner = {}
    for pi, ms in enumerate(pages):
        for m in ms:
            if m in needed:
                owner.setdefault(m, []).append(pi)
    sizes = np.array([len(m) for m in pages], dtype=np.int64)
    max_probe = max(probe_grid)
    recalls = np.zeros(len(probe_grid))
    cands = np.zeros(len(probe_grid))
    for qi in range(queries.shape[0]):
        dist = distances(queries[qi:qi + 1], centroids, angular)[0]
        order = np.argsort(dist)[:max_probe]
        rank_of = np.full(n_pages, np.iinfo(np.int32).max, dtype=np.int64)
        rank_of[order] = np.arange(order.shape[0])
        cum = np.cumsum(sizes[order])
        best = []
        for nb in truth[qi, :k]:
            pr = owner.get(int(nb))
            best.append(min((rank_of[p] for p in pr), default=np.iinfo(np.int32).max)
                        if pr else np.iinfo(np.int32).max)
        best = np.asarray(best)
        for j, P in enumerate(probe_grid):
            recalls[j] += float(np.count_nonzero(best < P)) / k
            cands[j] += float(cum[min(P, len(cum)) - 1])
    q = queries.shape[0]
    return recalls / q, cands / q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--page-capacity", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--angular", action="store_true")
    ap.add_argument("--partition", choices=["kmeans", "greedy"], default="kmeans")
    ap.add_argument("--kmeans-iters", type=int, default=10)
    ap.add_argument("--overlaps", type=float, nargs="+",
                    default=[1.0, 1.15, 1.25, 1.4, 1.75, 2.0])
    ap.add_argument("--max-replicas", type=int, default=4)
    ap.add_argument("--probes", type=int, nargs="+",
                    default=[1,2,4,8,12,16,24,32,48,64,96,128,192,256,384,512])
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    t0 = time.time()
    train, queries, truth = load(a.dataset, a.limit, a.queries, a.k, a.angular)
    if a.angular:
        train, queries = normalize(train), normalize(queries)
    print(f"loaded train={train.shape} queries={queries.shape} in {time.time()-t0:.1f}s", flush=True)

    n_pages = max(1, train.shape[0] // a.page_capacity)
    t1 = time.time()
    if a.partition == "kmeans":
        centroids, labels = kmeans(train, n_pages, a.angular, a.kmeans_iters)
    else:
        centroids, labels = greedy_incremental(train, a.page_capacity, a.angular)
        n_pages = centroids.shape[0]
    build_s = time.time() - t1
    print(f"partition={a.partition} pages={centroids.shape[0]} in {build_s:.1f}s", flush=True)

    t2 = time.time()
    cand_pages, plane = boundary_scores(train, centroids, a.angular, a.max_replicas)
    print(f"boundary scores in {time.time()-t2:.1f}s", flush=True)

    results = []
    for target in a.overlaps:
        t3 = time.time()
        pages, overlap, thresh = pages_at_overlap(cand_pages, plane,
                                                  centroids.shape[0], target)
        rec, cand = evaluate(queries, centroids, pages, truth, a.angular, a.k, a.probes)
        results.append({
            "target_overlap": target, "overlap_factor": overlap,
            "plane_threshold": thresh, "probes": a.probes,
            "recall": rec.tolist(), "candidates": cand.tolist(),
            "seconds": time.time() - t3,
        })
        hit = [(p, r, c) for p, r, c in zip(a.probes, rec, cand) if r >= 0.98]
        line = f"overlap={overlap:.3f}x"
        if hit:
            line += f"  -> Recall@10>=0.98 at P={hit[0][0]:>4}, candidates={hit[0][2]:>8.0f}"
        else:
            line += f"  -> max recall {rec.max():.4f} at {cand[int(np.argmax(rec))]:.0f} candidates"
        print(line, flush=True)

    out = {
        "dataset": a.dataset, "vectors": int(train.shape[0]), "queries": int(queries.shape[0]),
        "dimensions": int(train.shape[1]), "page_capacity": a.page_capacity,
        "pages": int(centroids.shape[0]), "partition": a.partition,
        "angular": a.angular, "max_replicas": a.max_replicas, "k": a.k,
        "partition_build_s": build_s, "results": results,
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.output}", flush=True)


if __name__ == "__main__":
    main()
