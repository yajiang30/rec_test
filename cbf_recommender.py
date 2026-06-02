"""
Content-Based Filtering recommender.

Builds a TF-IDF document for every movie by concatenating its metadata
fields (genres, keywords, cast, director, overview), then scores unseen
movies against a user's taste profile (mean of their history vectors).

Usage (standalone smoke test):
    python cbf_recommender.py

Plug into run_eval.py:
    from cbf_recommender import CBFRecommender
    rec = CBFRecommender.build(movies_df)
    # rec.recommend(query, history_ids, k) -> list[int]
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_tokens(cell, sep=" ") -> str:
    """Convert a list/array cell to a whitespace-joined token string."""
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell
    try:
        items = [str(x).replace(" ", "_") for x in cell if x]
        return sep.join(items)
    except TypeError:
        return ""


def _build_doc(row: pd.Series) -> str:
    """Combine all metadata fields into one weighted text document.

    Fields are repeated to up-weight them relative to the overview prose:
      genres   x3  (strong signal, clean vocabulary)
      keywords x2
      cast     x2  (underscore-joined so "Tom_Hanks" is one token)
      director x3  (auteur signal)
      overview x1
    """
    genres   = (_to_tokens(row.get("genres"))   + " ") * 3
    keywords = (_to_tokens(row.get("keywords")) + " ") * 2
    cast     = (_to_tokens(row.get("cast"))     + " ") * 2
    director = (_to_tokens(row.get("director")) + " ") * 3 if row.get("director") else ""
    overview = str(row.get("overview") or "")
    return f"{genres}{keywords}{cast}{director} {overview}".strip()


# --------------------------------------------------------------------------- #
# recommender
# --------------------------------------------------------------------------- #
class CBFRecommender:
    """
    Content-Based Filtering via TF-IDF cosine similarity.

    Candidate retrieval: score all unseen movies against the user's
    aggregate taste profile and return the top-k.

    Args:
        movie_ids:   array of movieIds, length N
        tfidf_mat:   sparse (N, vocab) TF-IDF matrix
        vectorizer:  fitted TfidfVectorizer (kept for inspection / future use)
    """

    name = "cbf"

    def __init__(
        self,
        movie_ids: np.ndarray,
        tfidf_mat: csr_matrix,
        vectorizer: TfidfVectorizer,
    ):
        self._ids = movie_ids                        # shape (N,)
        self._mat = tfidf_mat                        # shape (N, vocab)
        self._vec = vectorizer
        # fast lookup: movieId -> row index
        self._id2idx: dict[int, int] = {mid: i for i, mid in enumerate(movie_ids)}

    # ---------------------------------------------------------------------- #
    # factory
    # ---------------------------------------------------------------------- #
    @classmethod
    def build(
        cls,
        movies: pd.DataFrame,
        max_features: int = 20_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
    ) -> "CBFRecommender":
        """Fit TF-IDF on the movie corpus and return a ready recommender."""
        docs = movies.apply(_build_doc, axis=1).tolist()
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,       # log(1+tf) dampens very frequent terms
        )
        mat = vectorizer.fit_transform(docs)
        return cls(
            movie_ids=movies["movieId"].to_numpy(),
            tfidf_mat=mat,
            vectorizer=vectorizer,
        )

    # ---------------------------------------------------------------------- #
    # recommend
    # ---------------------------------------------------------------------- #
    def recommend(self, query: str, history_ids: list[int], k: int) -> list[int]:
        """
        Return up to k movieIds ranked by cosine similarity to the user profile.

        User profile = mean TF-IDF vector of their history movies.
        Movies already in history are excluded from results.
        """
        seen = set(history_ids)

        # collect history vectors that we actually have in the index
        history_idxs = [self._id2idx[mid] for mid in history_ids if mid in self._id2idx]
        if not history_idxs:
            # cold-start fallback: return highest vote_average movies (caller has no history)
            unseen_mask = np.array([mid not in seen for mid in self._ids])
            unseen_ids = self._ids[unseen_mask]
            return unseen_ids[:k].tolist()

        # mean profile vector (dense, shape (1, vocab))
        history_mat = self._mat[history_idxs]           # (H, vocab) sparse
        profile = np.asarray(history_mat.mean(axis=0))  # (1, vocab)

        # cosine similarity against full corpus
        sims = cosine_similarity(profile, self._mat).flatten()  # (N,)

        # mask out seen movies
        for mid in seen:
            idx = self._id2idx.get(mid)
            if idx is not None:
                sims[idx] = -1.0

        # top-k by descending similarity
        top_idxs = np.argpartition(sims, -k)[-k:]
        top_idxs = top_idxs[np.argsort(sims[top_idxs])[::-1]]
        return self._ids[top_idxs].tolist()


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time

    MOVIES_PATH = "processed/movies_enriched.parquet"
    SPLITS_PATH = "processed/splits.parquet"

    movies = pd.read_parquet(MOVIES_PATH)
    splits = pd.read_parquet(SPLITS_PATH)

    print("Building CBF index…")
    t0 = time.time()
    rec = CBFRecommender.build(movies)
    print(f"  done in {time.time()-t0:.2f}s  |  vocab size: {len(rec._vec.vocabulary_):,}")

    # quick eval over first 50 users
    from metrics import score_all
    all_scores = []
    for row in splits.head(50).itertuples(index=False):
        ranked = rec.recommend("", list(row.history_movieIds), k=20)
        all_scores.append(score_all(ranked, int(row.heldout_movieId)))

    agg = pd.DataFrame(all_scores).mean()
    print("\nCBF (n=50 users, k=20):")
    for metric, val in agg.items():
        print(f"  {metric:<10s} {val:.4f}")
