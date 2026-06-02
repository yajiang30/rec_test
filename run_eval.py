"""
Step 2D: End-to-end evaluation harness.

For each user in eval_set.parquet:
    1. ask a recommender to rank K candidates
    2. score the ranking against the user's heldout positive
    3. average the metrics across all users -> one row of the results table

This file ships with a `RandomRecommender` as the wiring smoke test. When
its NDCG comes out near zero (~0.005 for K=20 over ~3800 movies) you know
the pipeline is wired correctly; real recommenders plug in by implementing
the same `recommend(...)` signature.

Run:
    python run_eval.py --method random                     # smoke-test floor
    python run_eval.py --method cf --ok-only               # CF baseline
    python run_eval.py --method cbf --ok-only              # CBF baseline
    python run_eval.py --method pipeline --ok-only         # full hybrid
    python run_eval.py --method pipeline --ok-only --limit 10   # quick check
"""

import argparse
import os
import random
from typing import Protocol

import numpy as np
import pandas as pd

from metrics import score_all

EVAL_SET_PATH = "processed/eval_set.parquet"
MOVIES_PATH = "processed/movies_enriched.parquet"
RATINGS_PATH = "processed/ratings.parquet"
OUT_PATH = "processed/eval_results.parquet"

DEFAULT_K = 20
KS = (5, 10, 20)
SEED = 42
HIGH_RATING = 4         # match split.py + make_queries.py
MIN_LIKED = 3           # if a user has fewer liked movies, fall back to full history


class Recommender(Protocol):
    name: str

    def recommend(self, query: str, history_ids: list[int], k: int) -> list[int]:
        """Return a ranked list of up to k movieIds."""
        ...


class RandomRecommender:
    """Smoke-test baseline: shuffle the universe, drop history, take top-k.

    Expected scores on ML-1M (~3800 movies, 1 positive):
        HR@20  ~ 20 / 3800 = 0.005
        MRR    ~ harmonic of random rank, basically 0
    """

    name = "random"

    def __init__(self, all_movie_ids: np.ndarray, seed: int = SEED):
        self.all_movie_ids = all_movie_ids
        self.rng = random.Random(seed)

    def recommend(self, query: str, history_ids: list[int], k: int) -> list[int]:
        seen = set(history_ids)
        pool = [m for m in self.all_movie_ids if m not in seen]
        self.rng.shuffle(pool)
        return pool[:k]


def filter_to_liked(history_ids: list[int], liked_ids: set[int]) -> list[int]:
    """Keep only movies the user rated >= HIGH_RATING. Fall back to full
    history if too few liked items remain (rare given split.py's eligibility
    filter, but safe).
    """
    liked = [m for m in history_ids if m in liked_ids]
    return liked if len(liked) >= MIN_LIKED else history_ids


def evaluate(rec: Recommender, eval_df: pd.DataFrame,
             user_likes: dict[int, set[int]], k: int = DEFAULT_K) -> pd.DataFrame:
    rows = []
    for r in eval_df.itertuples(index=False):
        liked_history = filter_to_liked(list(r.history_movieIds),
                                        user_likes.get(int(r.userId), set()))
        ranked = rec.recommend(
            query=getattr(r, "query", "") or "",
            history_ids=liked_history,
            k=k,
        )
        scores = score_all(ranked, int(r.heldout_movieId), ks=KS)
        scores["userId"] = int(r.userId)
        rows.append(scores)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, method: str) -> dict:
    metric_cols = [c for c in results.columns if c != "userId"]
    summary = {"method": method, "n_users": len(results)}
    for c in metric_cols:
        summary[c] = float(results[c].mean())
    return summary


def print_summary(summary: dict) -> None:
    method = summary.pop("method")
    n = summary.pop("n_users")
    print("=" * 60)
    print(f"  {method}  (n={n} users)")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<10s} {v:.4f}")
    summary["method"] = method
    summary["n_users"] = n


def build_recommender(name: str, movies, ratings, splits, seed: int):
    if name == "random":
        return RandomRecommender(movies["movieId"].to_numpy(), seed=seed)
    if name == "cf":
        from cf_recommender import CFRecommender
        return CFRecommender.build(ratings, splits)
    if name == "cbf":
        from cbf_recommender import CBFRecommender
        return CBFRecommender.build(movies)
    if name == "pipeline":
        from pipeline_recommender import PipelineRecommender
        return PipelineRecommender.build(ratings, splits, movies)
    raise ValueError(f"unknown method: {name!r}  (choices: random, cf, cbf, pipeline)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="random",
                    choices=["random", "cf", "cbf", "pipeline"],
                    help="which recommender to evaluate")
    ap.add_argument("--eval-set", default=EVAL_SET_PATH)
    ap.add_argument("--movies", default=MOVIES_PATH)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--limit", type=int, default=None, help="cap users for a quick run")
    ap.add_argument("--ok-only", action="store_true",
                    help="only score users whose query status == 'ok'")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None,
                    help="output parquet path (default: processed/eval_results_<method>.parquet)")
    args = ap.parse_args()

    eval_df = pd.read_parquet(args.eval_set)
    movies = pd.read_parquet(args.movies)
    ratings = pd.read_parquet(RATINGS_PATH)
    splits = pd.read_parquet("processed/splits.parquet")

    # per-user set of high-rated movies, used to filter the taste profile
    # passed to recommenders. Mirrors the discipline in make_queries.py and
    # fixes the CF "averaging across loved + hated" bug.
    high = ratings[ratings.rating >= HIGH_RATING]
    user_likes = high.groupby("userId")["movieId"].apply(set).to_dict()

    if args.ok_only and "status" in eval_df.columns:
        before = len(eval_df)
        eval_df = eval_df[eval_df["status"] == "ok"]
        print(f"filtered to status=='ok': {len(eval_df)} / {before} users")

    if args.limit:
        eval_df = eval_df.head(args.limit)

    print(f"building recommender: {args.method!r}")
    rec = build_recommender(args.method, movies, ratings, splits, seed=args.seed)

    print(f"evaluating {rec.name!r} on {len(eval_df)} users (k={args.k}, |M|={len(movies):,})")
    results = evaluate(rec, eval_df, user_likes, k=args.k)

    summary = summarize(results, method=rec.name)
    print_summary(summary)

    out_path = args.out or f"processed/eval_results_{args.method}.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results.to_parquet(out_path, index=False)
    print(f"\nper-user results -> {out_path}")


if __name__ == "__main__":
    main()
