"""
Collaborative Filtering recommender using Alternating Least Squares (ALS).

Trains an implicit-feedback ALS model on the full ratings table, with the
500 evaluation heldout pairs removed to prevent answer leakage.

Ratings are binarized to implicit feedback:
    confidence = 1 + alpha * rating
so higher-rated movies still get more weight, but the model treats all
observed ratings as "the user engaged with this movie."

Usage (standalone smoke test):
    python cf_recommender.py

Plug into run_eval.py:
    from cf_recommender import CFRecommender
    rec = CFRecommender.build(ratings_df, splits_df)
    # rec.recommend(query, history_ids, k) -> list[int]
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


# --------------------------------------------------------------------------- #
# recommender
# --------------------------------------------------------------------------- #
class CFRecommender:
    """
    Collaborative Filtering via ALS (implicit library).

    At recommendation time we synthesise a temporary user vector by folding
    the history items into the learned item factors via one ALS step. This
    lets us handle eval users who may not have been in training (and avoids
    storing 6 K user vectors we mostly don't need).

    Args:
        model:       fitted ALS model
        movie_ids:   array of movieIds, length M (the CF item universe)
        item_factors: (M, F) float32 item embeddings from ALS
        alpha:       confidence scaling (must match training value)
    """

    name = "cf_als"

    def __init__(
        self,
        model: AlternatingLeastSquares,
        movie_ids: np.ndarray,
        item_factors: np.ndarray,
        alpha: float,
    ):
        self._model = model
        self._ids = movie_ids                          # (M,)
        self._factors = item_factors                   # (M, F)
        self._alpha = alpha
        self._id2idx: dict[int, int] = {mid: i for i, mid in enumerate(movie_ids)}

    # ---------------------------------------------------------------------- #
    # factory
    # ---------------------------------------------------------------------- #
    @classmethod
    def build(
        cls,
        ratings: pd.DataFrame,
        splits: pd.DataFrame,
        factors: int = 64,
        iterations: int = 20,
        regularization: float = 0.01,
        alpha: float = 40.0,
        random_state: int = 42,
    ) -> "CFRecommender":
        """
        Fit ALS on ratings minus the heldout pairs, return a ready recommender.

        The heldout (userId, movieId) pairs from splits are dropped so the
        model never sees the ground-truth answers.
        """
        # --- remove heldout pairs ---
        heldout = set(zip(splits["userId"], splits["heldout_movieId"]))
        mask = ~ratings.apply(
            lambda r: (int(r.userId), int(r.movieId)) in heldout, axis=1
        )
        clean = ratings[mask].copy()
        print(f"  ratings: {len(ratings):,} -> {len(clean):,} after removing {mask.sum()} heldout pairs")

        # --- build item universe from cleaned ratings ---
        movie_ids = clean["movieId"].unique()
        movie_ids.sort()
        mid2idx = {mid: i for i, mid in enumerate(movie_ids)}
        uid2idx = {uid: i for i, uid in enumerate(clean["userId"].unique())}

        n_users = len(uid2idx)
        n_items = len(mid2idx)

        rows = clean["userId"].map(uid2idx).to_numpy()
        cols = clean["movieId"].map(mid2idx).to_numpy()
        # confidence = 1 + alpha * raw_rating
        data = (1.0 + alpha * clean["rating"].to_numpy()).astype(np.float32)

        # implicit expects (users, items) CSR for fit()
        user_item = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        print(f"  matrix: {n_users:,} users x {n_items:,} items  (nnz={user_item.nnz:,})")

        model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=random_state,
            use_gpu=False,
        )
        print(f"  training ALS (factors={factors}, iterations={iterations})…")
        model.fit(user_item, show_progress=True)

        return cls(
            model=model,
            movie_ids=movie_ids,
            item_factors=model.item_factors,   # (M, F)
            alpha=alpha,
        )

    # ---------------------------------------------------------------------- #
    # recommend
    # ---------------------------------------------------------------------- #
    def _fold_in(self, history_ids: list[int], history_ratings: list[float] | None = None) -> np.ndarray:
        """
        Compute a user latent vector by solving one ALS step:
            u = (Y^T C_u Y + λI)^{-1} Y^T C_u p_u

        where Y = item factors, C_u = diag(confidence), p_u = 1.

        history_ratings: if provided, used as raw rating values for confidence
                         weighting; otherwise all treated as rating=4 (positive).
        """
        idxs = [self._id2idx[mid] for mid in history_ids if mid in self._id2idx]
        if not idxs:
            return np.zeros(self._factors.shape[1], dtype=np.float32)

        Y = self._factors[idxs]                 # (H, F)
        if history_ratings is not None:
            ratings_arr = np.array([history_ratings[i] for i in range(len(idxs))], dtype=np.float32)
        else:
            ratings_arr = np.full(len(idxs), 4.0, dtype=np.float32)

        c = (1.0 + self._alpha * ratings_arr).astype(np.float32)  # confidence weights

        lam = self._model.regularization
        F = Y.shape[1]

        # YtCY + λI
        YtCY = (Y * c[:, None]).T @ Y + lam * np.eye(F, dtype=np.float32)
        # YtCp  (p=1 for all observed items)
        YtCp = (Y * c[:, None]).sum(axis=0)

        user_vec = np.linalg.solve(YtCY, YtCp)
        return user_vec.astype(np.float32)

    def recommend(self, query: str, history_ids: list[int], k: int) -> list[int]:
        """
        Return up to k movieIds ranked by inner product with the fold-in user vector.
        Movies in history are excluded.
        """
        seen = set(history_ids)
        user_vec = self._fold_in(history_ids)   # (F,)

        # scores: inner product with all item factors
        scores = self._factors @ user_vec       # (M,)

        # mask seen items
        for mid in seen:
            idx = self._id2idx.get(mid)
            if idx is not None:
                scores[idx] = -np.inf

        top_idxs = np.argpartition(scores, -k)[-k:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]
        return self._ids[top_idxs].tolist()


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    RATINGS_PATH = "processed/ratings.parquet"
    SPLITS_PATH  = "processed/splits.parquet"

    ratings = pd.read_parquet(RATINGS_PATH)
    splits  = pd.read_parquet(SPLITS_PATH)

    print("Building CF (ALS) model…")
    t0 = time.time()
    rec = CFRecommender.build(ratings, splits)
    print(f"  done in {time.time()-t0:.1f}s")

    from metrics import score_all
    all_scores = []
    for row in splits.head(50).itertuples(index=False):
        ranked = rec.recommend("", list(row.history_movieIds), k=20)
        all_scores.append(score_all(ranked, int(row.heldout_movieId)))

    agg = pd.DataFrame(all_scores).mean()
    print("\nCF-ALS (n=50 users, k=20):")
    for metric, val in agg.items():
        print(f"  {metric:<10s} {val:.4f}")
