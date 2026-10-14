"""Offline evaluation with ranking metrics (precision@k, recall@k, NDCG@k).

Protocol: for each user, hold out a random fraction of their ratings. Train on
the rest. A held-out movie is *relevant* if the user rated it >= 4. We ask the
model for its top-k recommendations (excluding training-seen movies) and score
them against the held-out relevant set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .recommenders import (
    ContentBased,
    Dataset,
    Hybrid,
    ItemCF,
    SVDRecommender,
)


def train_test_split_per_user(ratings: pd.DataFrame, test_frac: float = 0.2,
                              seed: int = 42, min_ratings: int = 10):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, grp in ratings.groupby("user_id"):
        if len(grp) < min_ratings:
            train_parts.append(grp)
            continue
        idx = rng.permutation(len(grp))
        n_test = max(1, int(len(grp) * test_frac))
        test_parts.append(grp.iloc[idx[:n_test]])
        train_parts.append(grp.iloc[idx[n_test:]])
    return (pd.concat(train_parts).reset_index(drop=True),
            pd.concat(test_parts).reset_index(drop=True))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for m in recommended[:k] if m in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for m in recommended[:k] if m in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = sum((1.0 / np.log2(i + 2)) for i, m in enumerate(recommended[:k])
              if m in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def build_model(name: str, train: pd.DataFrame, movies: pd.DataFrame):
    data = Dataset(train)
    if name == "itemcf":
        return ItemCF(data).fit(), data
    if name == "svd":
        return SVDRecommender(data).fit(), data
    if name == "content":
        return ContentBased(data, movies).fit(), data
    if name == "hybrid":
        return Hybrid(data, movies), data
    raise ValueError(f"unknown model: {name}")


def evaluate(name: str, train: pd.DataFrame, test: pd.DataFrame,
             movies: pd.DataFrame, k: int = 10, n_users: int | None = 200,
             seed: int = 42) -> dict:
    model, data = build_model(name, train, movies)

    # Relevant held-out items per user (rating >= 4).
    rel = (test[test["rating"] >= 4]
           .groupby("user_id")["item_id"].apply(set).to_dict())
    eval_users = [u for u in rel if u in data.u_index]
    rng = np.random.default_rng(seed)
    if n_users and len(eval_users) > n_users:
        eval_users = list(rng.choice(eval_users, n_users, replace=False))

    precs, recs, ndcgs = [], [], []
    for u in eval_users:
        recommended = model.recommend(int(u), k)
        relevant = rel[u]
        precs.append(precision_at_k(recommended, relevant, k))
        recs.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))

    return {
        "model": name,
        "k": k,
        "n_users_evaluated": len(eval_users),
        "precision@k": round(float(np.mean(precs)), 4),
        "recall@k": round(float(np.mean(recs)), 4),
        "ndcg@k": round(float(np.mean(ndcgs)), 4),
    }
