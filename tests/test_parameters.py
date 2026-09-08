"""Tests for thread pool size and rerank_factor parameters.

These parameters are documented and accepted by the constructor but are not
exercised in existing tests. These checks verify:
- thread count does not change result correctness
- rerank_factor monotonically improves recall (or at minimum does not harm it)
"""

import numpy as np

from chronovec import NativeChronoVecIndex


def clustered(n, dim, seed=42):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(8, dim)).astype(np.float32)
    return (centres[rng.integers(0, 8, n)] + rng.normal(scale=0.3, size=(n, dim))).astype(
        np.float32
    )


def recall_at_k(results, ground_truth_ids, k):
    result_ids = {r.id for r in results[:k]}
    return len(result_ids & set(ground_truth_ids[:k])) / k


def test_threads_do_not_change_results():
    """Results with threads=1 and threads=4 must be identical."""
    dim = 32
    n = 500
    vectors = clustered(n, dim)
    query = clustered(1, dim, seed=99)[0]

    index1 = NativeChronoVecIndex(dim, page_capacity=64, nprobe=16, threads=1)
    index4 = NativeChronoVecIndex(dim, page_capacity=64, nprobe=16, threads=4)

    ids = np.arange(n, dtype=np.int64)
    index1.insert_many(ids, vectors)
    index4.insert_many(ids, vectors)

    r1 = [r.id for r in index1.search(query, k=10)]
    r4 = [r.id for r in index4.search(query, k=10)]

    # Results should be identical: same data, same nprobe, same algorithm
    assert r1 == r4, f"thread=1 and thread=4 returned different results: {r1} vs {r4}"

    index1.close()
    index4.close()


def test_rerank_factor_monotonically_improves_recall():
    """Higher rerank_factor must not decrease recall versus lower."""
    dim = 32
    n = 1000
    k = 10
    vectors = clustered(n, dim)
    queries = clustered(20, dim, seed=7)

    # Exact nearest neighbours (ground truth)
    from chronovec import exact_search

    def avg_recall(factor):
        index = NativeChronoVecIndex(dim, page_capacity=64, nprobe=16, rerank_factor=factor)
        index.insert_many(np.arange(n, dtype=np.int64), vectors)
        total = 0.0
        for q in queries:
            exact = [r.id for r in exact_search(vectors, q[None], k=k)[0]]
            approx = [r.id for r in index.search(q, k=k)]
            total += len(set(exact) & set(approx)) / k
        index.close()
        return total / len(queries)

    recall_1 = avg_recall(1)
    recall_4 = avg_recall(4)
    recall_8 = avg_recall(8)

    # Recall must not decrease as rerank_factor increases
    assert recall_4 >= recall_1 - 0.01, (
        f"rerank_factor=4 ({recall_4:.3f}) should not be worse than 1 ({recall_1:.3f})"
    )
    assert recall_8 >= recall_4 - 0.01, (
        f"rerank_factor=8 ({recall_8:.3f}) should not be worse than 4 ({recall_4:.3f})"
    )
