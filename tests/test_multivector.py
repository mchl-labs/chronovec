import numpy as np
import pytest

from chronovec.multivector import FixedDimensionalEncoder, MultiVectorIndex, chamfer


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def tokens(rng, n, d=32):
    return unit(rng.standard_normal((n, d)).astype(np.float32))


def test_chamfer_matches_definition():
    rng = np.random.default_rng(0)
    q, d = tokens(rng, 4), tokens(rng, 6)
    expected = sum(max(float(qi @ dj) for dj in d) for qi in q)
    assert chamfer(q, d) == pytest.approx(expected, rel=1e-5)


def test_chamfer_handles_empty():
    rng = np.random.default_rng(0)
    assert chamfer(tokens(rng, 3), np.empty((0, 32), dtype=np.float32)) == 0.0


def test_exact_document_is_retrieved():
    rng = np.random.default_rng(1)
    with MultiVectorIndex(32, strategy="tokens", nprobe=256) as index:
        docs = {i: tokens(rng, 12) for i in range(60)}
        for i, t in docs.items():
            index.add(i, t)
        target = 17
        query = unit(docs[target][:5] + 0.05 * rng.standard_normal((5, 32)).astype(np.float32))
        assert index.search(query, k=1, candidates=60)[0].doc_id == target


def test_ranking_agrees_with_exact_chamfer():
    rng = np.random.default_rng(2)
    with MultiVectorIndex(32, strategy="tokens", nprobe=256) as index:
        docs = {i: tokens(rng, 10) for i in range(40)}
        for i, t in docs.items():
            index.add(i, t)
        query = tokens(rng, 6)
        hits = index.search(query, k=5, candidates=40)
        exact = sorted(((chamfer(query, d), i) for i, d in docs.items()), reverse=True)[:5]
        assert [h.doc_id for h in hits] == [i for _, i in exact]
        assert hits[0].score == pytest.approx(exact[0][0], rel=1e-5)


def test_delete_removes_every_token():
    rng = np.random.default_rng(3)
    with MultiVectorIndex(32, strategy="tokens", nprobe=256) as index:
        doc = tokens(rng, 8)
        index.add(1, doc)
        index.add(2, tokens(rng, 8))
        index.delete(1)
        assert {h.doc_id for h in index.search(doc, k=5, candidates=20)} == {2}


def test_payload_round_trips():
    rng = np.random.default_rng(4)
    with MultiVectorIndex(32, strategy="tokens", nprobe=256) as index:
        doc = tokens(rng, 8)
        index.add(9, doc, title="paper")
        assert index.search(doc, k=1, candidates=10)[0].payload == {"title": "paper"}


def test_shape_validation():
    rng = np.random.default_rng(5)
    with MultiVectorIndex(32, nprobe=64) as index:
        with pytest.raises(ValueError, match=r"shape \(n, 32\)"):
            index.add(1, rng.standard_normal((4, 8)).astype(np.float32))
        index.add(1, tokens(rng, 4))
        with pytest.raises(ValueError, match=r"shape \(n, 32\)"):
            index.search(rng.standard_normal((2, 8)).astype(np.float32))


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="tokens.*fde"):
        MultiVectorIndex(32, strategy="magic")


def test_fde_encoder_shapes_and_determinism():
    rng = np.random.default_rng(6)
    enc = FixedDimensionalEncoder(32, k_sim=3, repetitions=2, projection_dim=8, seed=5)
    assert enc.fde_dimensions == 2 * 8 * 8
    doc = tokens(rng, 10)
    assert enc.encode_document(doc).shape == (enc.fde_dimensions,)
    assert np.array_equal(enc.encode_document(doc), enc.encode_document(doc))
    assert enc.encode_query(doc).shape == (enc.fde_dimensions,)


def test_fde_strategy_still_functions():
    rng = np.random.default_rng(7)
    with MultiVectorIndex(32, strategy="fde", k_sim=3, repetitions=4, nprobe=64) as index:
        docs = {i: tokens(rng, 10) for i in range(30)}
        for i, t in docs.items():
            index.add(i, t)
        hits = index.search(docs[3], k=3, candidates=30)
        assert hits and hits[0].score == pytest.approx(
            chamfer(docs[3], docs[hits[0].doc_id]), rel=1e-5
        )
