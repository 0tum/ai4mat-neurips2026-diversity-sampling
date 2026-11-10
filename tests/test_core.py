import numpy as np
import pytest

from divsample import distance, metrics, selection, splits

rng = np.random.default_rng(0)
ECFP = rng.random((300, 64)) < 0.2


def brute_force_distance(ecfp, pool, queries, k):
    a = ecfp[queries].astype(float)
    b = ecfp[pool].astype(float)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return (1 - np.sort(sim, axis=1)[:, ::-1][:, :k]).mean(axis=1)


@pytest.mark.parametrize("pool_chunk,query_chunk", [(7, 5), (1000, 1000)])
def test_knn_distance_matches_brute_force(pool_chunk, query_chunk):
    pool, queries = np.arange(0, 200), np.arange(200, 300)
    got = distance.knn_distance(ECFP, pool, queries, k=10, pool_chunk=pool_chunk, query_chunk=query_chunk)
    np.testing.assert_allclose(got, brute_force_distance(ECFP, pool, queries, 10), rtol=1e-6)


def test_knn_distance_does_not_depend_on_chunks():
    pool, queries = np.arange(0, 200), np.arange(200, 300)
    a = distance.knn_distance(ECFP, pool, queries, k=10, pool_chunk=7, query_chunk=5)
    b = distance.knn_distance(ECFP, pool, queries, k=10)
    assert np.array_equal(a, b)


def test_random_selection_is_seeded_sorted_subset():
    pool = np.arange(10, 110)
    a = selection.random_selection(pool, 20, sample_seed=3)
    assert np.array_equal(a, selection.random_selection(pool, 20, sample_seed=3))
    assert not np.array_equal(a, selection.random_selection(pool, 20, sample_seed=4))
    assert np.all(np.diff(a) > 0) and set(a) <= set(pool)


def test_maxmin_picks_far_apart_molecules():
    # three tight groups; three picks must come from three different groups
    ecfp = np.zeros((30, 30), dtype=bool)
    for g in range(3):
        ecfp[g * 10:(g + 1) * 10, g * 10:(g + 1) * 10 - 1] = True
        ecfp[g * 10 + np.arange(10), g * 10 + 9] = np.arange(10) % 2 == 0
    picks = selection.maxmin_selection(np.arange(30), ecfp, 3, sample_seed=0)
    assert sorted(picks // 10) == [0, 1, 2]


def test_m_over_n_interpolates_and_censors():
    grid = [100, 300, 1000, 3000, 10000]
    curve = np.array([1.0, 0.8, 0.6, 0.5, 0.4])
    value, side = metrics.log2_m_over_n(curve, 0.6, 1000, grid)
    assert side == "within" and value == pytest.approx(0.0)
    assert metrics.log2_m_over_n(curve, 0.3, 1000, grid)[1] == "high"
    assert metrics.log2_m_over_n(curve, 1.2, 1000, grid)[1] == "low"
    # a curve that rises again is replaced by its running minimum
    rising = np.array([1.0, 0.6, 0.7, 0.5, 0.4])
    assert metrics.log2_m_over_n(rising, 0.6, 1000, grid)[0] < 0


def test_deciles_have_equal_counts_and_follow_distance():
    d = rng.random(1000)
    decile = metrics.distance_deciles(d)
    assert np.bincount(decile).tolist() == [100] * 10
    assert d[decile == 0].max() < d[decile == 9].min()


def test_cluster_split_sizes():
    labels = np.repeat(np.arange(10), 50)
    split = splits.cluster_split(labels, leave_out=0, seed=0)
    assert len(split["ood_test"]) == 50 and len(split["iid_test"]) == 50
    assert len(split["train_pool"]) == 400
    assert set(labels[split["ood_test"]]) == {0}


def test_pcqm_property_split_takes_the_tails():
    y = np.concatenate([rng.normal(0, 1, 5000), [8.0, 9.0, -9.0]])
    split = splits.property_split_pcqm(y, n_iid=100, n_ood=3)
    assert sorted(split["ood_test"]) == [5000, 5001, 5002]
    assert len(split["iid_test"]) == 100 and len(split["train_pool"]) == 4900
