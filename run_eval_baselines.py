"""
Baseline evaluation driver.

Runs CBFRecommender and CFRecommender through the existing run_eval harness
and prints a side-by-side comparison table.

Prereqs:
    processed/movies_enriched.parquet
    processed/ratings.parquet
    processed/splits.parquet
    processed/eval_set.parquet   <- only needed if --use-queries is set

Run:
    python run_eval_baselines.py                  # both baselines, all users
    python run_eval_baselines.py --limit 50       # quick sanity check
    python run_eval_baselines.py --models cbf     # CBF only
    python run_eval_baselines.py --models cf      # CF only
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from metrics import score_all
from cbf_recommender import CBFRecommender
from cf_recommender import CFRecommender
from llm_recommender import LLMRecommender

MOVIES_PATH = "processed/movies_enriched.parquet"
RATINGS_PATH = "processed/ratings.parquet"
SPLITS_PATH = "processed/splits.parquet"
EVAL_SET_PATH = "processed/eval_set.parquet"
OUT_PATH = "processed/baseline_results.parquet"

KS = (5, 10, 20)
DEFAULT_K = 20


def evaluate(rec, eval_df: pd.DataFrame, k: int = DEFAULT_K) -> pd.DataFrame:
    rows = []
    for r in eval_df.itertuples(index=False):
        query = getattr(r, "query", "") or ""
        ranked = rec.recommend(query, list(r.history_movieIds), k=k)
        scores = score_all(ranked, int(r.heldout_movieId), ks=KS)
        scores["userId"] = int(r.userId)
        rows.append(scores)
    return pd.DataFrame(rows)


def print_table(summaries: list[dict]) -> None:
    metrics = [c for c in summaries[0] if c not in ("method", "n_users")]
    header = f"{'Method':<14}" + "".join(f"{m:>10}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for s in summaries:
        row = f"{s['method']:<14}" + "".join(f"{s[m]:>10.4f}" for m in metrics)
        print(row)
    print("=" * len(header))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=["cbf", "cf", "llm"], default=["cbf", "cf", "llm"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--use-queries", action="store_true",
                    help="load queries from eval_set.parquet instead of splits.parquet")
    ap.add_argument("--ok-only", action="store_true",
                    help="filter to status=='ok' rows (requires --use-queries)")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    movies  = pd.read_parquet(MOVIES_PATH)
    ratings = pd.read_parquet(RATINGS_PATH)
    splits  = pd.read_parquet(SPLITS_PATH)

    # eval dataframe: splits has history+heldout; eval_set also has LLM queries
    if args.use_queries and os.path.exists(EVAL_SET_PATH):
        eval_df = pd.read_parquet(EVAL_SET_PATH)
        if args.ok_only and "status" in eval_df.columns:
            before = len(eval_df)
            eval_df = eval_df[eval_df["status"] == "ok"]
            print(f"filtered to status=='ok': {len(eval_df)}/{before}")
    else:
        # splits already has history_movieIds and heldout_movieId
        eval_df = splits.copy()
        eval_df["query"] = ""   # baselines don't use the query

    if args.limit:
        eval_df = eval_df.head(args.limit)

    print(f"evaluating on {len(eval_df)} users (k={args.k})\n")

    summaries = []
    all_results = []

    if "cbf" in args.models:
        print("── Building CBF index…")
        t0 = time.time()
        cbf = CBFRecommender.build(movies)
        print(f"   done in {time.time()-t0:.1f}s  |  vocab={len(cbf._vec.vocabulary_):,}\n")
        results = evaluate(cbf, eval_df, k=args.k)
        results["method"] = "cbf"
        all_results.append(results)
        metric_cols = [c for c in results.columns if c not in ("userId", "method")]
        summaries.append({"method": "cbf", "n_users": len(results),
                          **{c: float(results[c].mean()) for c in metric_cols}})

    if "cf" in args.models:
        print("── Building CF-ALS model…")
        t0 = time.time()
        cf = CFRecommender.build(ratings, splits)
        print(f"   done in {time.time()-t0:.1f}s\n")
        results = evaluate(cf, eval_df, k=args.k)
        results["method"] = "cf_als"
        all_results.append(results)
        metric_cols = [c for c in results.columns if c not in ("userId", "method")]
        summaries.append({"method": "cf_als", "n_users": len(results),
                          **{c: float(results[c].mean()) for c in metric_cols}})

    if "llm" in args.models:
        print("── Building LLM-pure recommender…")
        llm = LLMRecommender(movies)
        print("   ready (calls Ollama at inference time)\n")
        results = evaluate(llm, eval_df, k=args.k)
        results["method"] = "llm_pure"
        all_results.append(results)
        metric_cols = [c for c in results.columns if c not in ("userId", "method")]
        summaries.append({"method": "llm_pure", "n_users": len(results),
                          **{c: float(results[c].mean()) for c in metric_cols}})

    print_table(summaries)

    combined = pd.concat(all_results, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combined.to_parquet(args.out, index=False)
    print(f"\nper-user results -> {args.out}")


if __name__ == "__main__":
    main()
