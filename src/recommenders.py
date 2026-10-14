"""Recommender models for MovieLens.

Three complementary approaches plus a hybrid, all sharing a common
``recommend(user_id, k)`` interface:

  * ItemCF        — item-item collaborative filtering (cosine similarity).
  * SVDRecommender — latent-factor matrix factorization (truncated SVD).
  * ContentBased  — genre-profile matching (no ratings co-occurrence needed,
                    so it works for cold-start items).
  * Hybrid        — normalized blend of SVD + content scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from .data import GENRES


class Dataset:
    """Dense user-item rating matrix with id<->index maps."""

    def __init__(self, ratings: pd.DataFrame):
        self.user_ids = np.sort(ratings["user_id"].unique())
        self.item_ids = np.sort(ratings["item_id"].unique())
        self.u_index = {u: i for i, u in enumerate(self.user_ids)}
        self.i_index = {m: j for j, m in enumerate(self.item_ids)}
        self.matrix = np.zeros((len(self.user_ids), len(self.item_ids)), dtype=np.float32)
        for u, m, r in zip(ratings["user_id"], ratings["item_id"], ratings["rating"]):
            self.matrix[self.u_index[u], self.i_index[m]] = r

    def rated_items(self, user_id: int) -> set[int]:
        row = self.matrix[self.u_index[user_id]]
        return {self.item_ids[j] for j in np.nonzero(row)[0]}


class _Base:
    def __init__(self, data: Dataset):
        self.data = data

    def _top_unrated(self, user_id: int, scores: np.ndarray, k: int,
                     exclude: set[int] | None = None) -> list[int]:
        exclude = exclude or self.data.rated_items(user_id)
        ex_idx = [self.data.i_index[m] for m in exclude if m in self.data.i_index]
        scores = scores.copy()
        scores[ex_idx] = -np.inf
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [int(self.data.item_ids[j]) for j in top if np.isfinite(scores[j])]


class ItemCF(_Base):
    """Item-item collaborative filtering via cosine similarity.

    Raw cosine lets obscure items with a single co-rating reach similarity 1.0
    and dominate recommendations. We apply **significance weighting**: shrink
    each pair's similarity by co_ratings / (co_ratings + shrinkage), so pairs
    supported by few users count for little.
    """

    def __init__(self, data: Dataset, shrinkage: float = 25.0):
        super().__init__(data)
        self.shrinkage = shrinkage

    def fit(self) -> "ItemCF":
        sim = cosine_similarity(self.data.matrix.T)        # items x items
        binary = (self.data.matrix > 0).astype(np.float32)
        co_ratings = binary.T @ binary                     # #users co-rating each pair
        sim *= co_ratings / (co_ratings + self.shrinkage)  # significance weighting
        np.fill_diagonal(sim, 0.0)
        self.sim = sim
        return self

    def _scores(self, user_id: int) -> np.ndarray:
        user_row = self.data.matrix[self.data.u_index[user_id]]
        # For top-N ranking we use the raw similarity-weighted sum rather than a
        # normalized predicted rating: dividing by the similarity mass would
        # cancel significance weighting and let a single cold neighbor score as
        # high as a well-supported one. The raw sum rewards items with many
        # strong, well-supported neighbors among what the user liked.
        return self.sim @ user_row

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        return self._top_unrated(user_id, self._scores(user_id), k)


class SVDRecommender(_Base):
    """Latent-factor model via truncated SVD on the mean-centered matrix."""

    def __init__(self, data: Dataset, n_factors: int = 40, seed: int = 42):
        super().__init__(data)
        self.n_factors = n_factors
        self.seed = seed

    def fit(self) -> "SVDRecommender":
        R = self.data.matrix
        mask = R > 0
        self.global_mean = R[mask].mean() if mask.any() else 0.0
        centered = np.where(mask, R - self.global_mean, 0.0)
        # TruncatedSVD needs n_components < n_features; clamp for small matrices.
        n_comp = max(1, min(self.n_factors, min(R.shape) - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
        user_factors = svd.fit_transform(centered)          # users x f
        self.pred = user_factors @ svd.components_ + self.global_mean
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        return self._top_unrated(user_id, self.pred[self.data.u_index[user_id]], k)


class ContentBased(_Base):
    """Genre-profile recommender.

    Builds a user taste vector from the genres of movies they rated highly,
    then ranks unseen movies by cosine similarity to that profile.
    """

    def __init__(self, data: Dataset, movies: pd.DataFrame, like_threshold: float = 4.0):
        super().__init__(data)
        self.like_threshold = like_threshold
        aligned = movies.set_index("item_id").reindex(self.data.item_ids)
        g = aligned[GENRES].fillna(0).to_numpy(dtype=np.float32)
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        self.item_genres = np.divide(g, norms, out=np.zeros_like(g), where=norms > 0)

    def fit(self) -> "ContentBased":
        return self

    def _profile(self, user_id: int) -> np.ndarray:
        row = self.data.matrix[self.data.u_index[user_id]]
        liked = row >= self.like_threshold
        if not liked.any():
            liked = row > 0
        weights = row[liked][:, None]
        profile = (self.item_genres[liked] * weights).sum(axis=0)
        n = np.linalg.norm(profile)
        return profile / n if n > 0 else profile

    def _scores(self, user_id: int) -> np.ndarray:
        return self.item_genres @ self._profile(user_id)

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        return self._top_unrated(user_id, self._scores(user_id), k)


class Hybrid(_Base):
    """Normalized blend of SVD (behavioral) and content (genre) scores."""

    def __init__(self, data: Dataset, movies: pd.DataFrame, alpha: float = 0.8):
        super().__init__(data)
        self.alpha = alpha
        self.svd = SVDRecommender(data).fit()
        self.content = ContentBased(data, movies).fit()

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        lo, hi = np.min(x), np.max(x)
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        s = self._norm(self.svd.pred[self.data.u_index[user_id]])
        c = self._norm(self.content._scores(user_id))
        return self._top_unrated(user_id, self.alpha * s + (1 - self.alpha) * c, k)
