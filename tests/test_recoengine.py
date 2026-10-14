"""Tests on a small synthetic rating set (no dataset download required)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    train_test_split_per_user,
)
from src.recommenders import ContentBased, Dataset, Hybrid, ItemCF, SVDRecommender


@pytest.fixture
def ratings() -> pd.DataFrame:
    # Two taste clusters: users 1-3 like items 1-3, users 4-6 like items 4-6.
    rows = []
    for u in (1, 2, 3):
        for m in (1, 2, 3):
            rows.append((u, m, 5))
        rows.append((u, 7, 3))            # shared mediocre item
    for u in (4, 5, 6):
        for m in (4, 5, 6):
            rows.append((u, m, 5))
        rows.append((u, 7, 3))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])


@pytest.fixture
def movies() -> pd.DataFrame:
    from src.data import GENRES
    rows = []
    for m in range(1, 8):
        genres = {g: 0 for g in GENRES}
        # items 1-3 are "Action", items 4-6 are "Comedy", item 7 both.
        if m <= 3:
            genres["Action"] = 1
        elif m <= 6:
            genres["Comedy"] = 1
        else:
            genres["Action"] = genres["Comedy"] = 1
        rows.append({"item_id": m, "title": f"Movie {m}", **genres})
    return pd.DataFrame(rows)


# ---- metrics ----

def test_precision_recall_ndcg():
    rec = [10, 20, 30, 40]
    rel = {20, 40, 99}
    assert precision_at_k(rec, rel, 4) == 0.5
    assert recall_at_k(rec, rel, 4) == pytest.approx(2 / 3)
    # NDCG: hits at ranks 2 and 4.
    assert 0 < ndcg_at_k(rec, rel, 4) < 1

def test_ndcg_perfect_is_one():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)


# ---- dataset ----

def test_dataset_matrix_shape(ratings):
    d = Dataset(ratings)
    assert d.matrix.shape == (6, 7)
    assert d.matrix[d.u_index[1], d.i_index[1]] == 5

def test_rated_items(ratings):
    d = Dataset(ratings)
    assert d.rated_items(1) == {1, 2, 3, 7}


# ---- recommenders learn the cluster structure ----

def test_itemcf_recommends_in_cluster(ratings):
    d = Dataset(ratings)
    model = ItemCF(d).fit()
    # User 1 rated 1,2,3,7; the only unrated in-cluster-ish item is 7 already
    # rated, so recommend should surface items, not crash, and exclude rated.
    recs = model.recommend(1, k=3)
    assert all(m not in d.rated_items(1) for m in recs)

def test_svd_excludes_rated(ratings):
    d = Dataset(ratings)
    model = SVDRecommender(d, n_factors=3).fit()
    recs = model.recommend(1, k=3)
    assert all(m not in d.rated_items(1) for m in recs)

def test_content_based_matches_taste(ratings, movies):
    d = Dataset(ratings)
    model = ContentBased(d, movies).fit()
    # User 1 likes Action (items 1-3). Among unrated, the Action-flavored item 7
    # should outrank pure-Comedy items 4-6.
    scores = model._scores(1)
    action_item = d.i_index[7]
    comedy_items = [d.i_index[m] for m in (4, 5, 6)]
    assert scores[action_item] >= max(scores[c] for c in comedy_items)

def test_hybrid_runs(ratings, movies):
    d = Dataset(ratings)
    model = Hybrid(d, movies)
    recs = model.recommend(1, k=3)
    assert isinstance(recs, list)


# ---- split ----

def test_split_keeps_all_ratings(ratings):
    train, test = train_test_split_per_user(ratings, test_frac=0.3, min_ratings=4)
    assert len(train) + len(test) == len(ratings)
