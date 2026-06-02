"""
Step 2A: Leave-one-out evaluation split.

For each sampled user, hold out their most recent high-rated movie as the
ground-truth positive; everything else is their "history" (the input the
recommender / query generator gets to see). The held-out (user, movie) pair
must be removed from CF training to prevent answer leakage.

Outputs a single DataFrame, one row per evaluated user:
    userId | history_movieIds (list[int]) | heldout_movieId | heldout_timestamp

Run:  python split.py
"""

import argparse
import os

import numpy as np
import pandas as pd

RATINGS_PATH = "processed/ratings.parquet"
OUT_PATH = "processed/splits.parquet"

MIN_RATINGS = 20         # users below this have too thin a history
MIN_HIGH_RATED = 5       # need enough positives to talk about taste
HIGH_RATING = 4          # >= this counts as a "positive"
DEFAULT_N_USERS = 500    # sample size; LLM eval is the cost driver
SEED = 42


def make_splits(ratings: pd.DataFrame, n_users: int, seed: int) -> pd.DataFrame:
    # users with enough overall history AND enough positives
    user_counts = ratings.groupby("userId").size()
    high_counts = ratings[ratings.rating >= HIGH_RATING].groupby("userId").size()
    eligible = user_counts[user_counts >= MIN_RATINGS].index.intersection(
        high_counts[high_counts >= MIN_HIGH_RATED].index
    )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible.to_numpy(), size=min(n_users, len(eligible)), replace=False)

    sub = ratings[ratings.userId.isin(chosen)].copy()
    # held-out = most recent rating >=4 per user (next-item-prediction framing)
    pos = sub[sub.rating >= HIGH_RATING]
    heldout_idx = pos.sort_values("timestamp").groupby("userId").tail(1).set_index("userId")

    rows = []
    for uid, grp in sub.groupby("userId"):
        if uid not in heldout_idx.index:
            continue
        ho = heldout_idx.loc[uid]
        history = grp[grp.movieId != ho.movieId].sort_values("timestamp")
        rows.append({
            "userId": int(uid),
            "history_movieIds": history.movieId.astype(int).tolist(),
            "heldout_movieId": int(ho.movieId),
            "heldout_timestamp": int(ho.timestamp),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=DEFAULT_N_USERS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    ratings = pd.read_parquet(RATINGS_PATH)
    splits = make_splits(ratings, args.n_users, args.seed)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    splits.to_parquet(OUT_PATH, index=False)

    hist_lens = splits.history_movieIds.map(len)
    print("=" * 60)
    print(f"STEP 2A COMPLETE  ->  {OUT_PATH}")
    print("=" * 60)
    print(f"evaluated users        : {len(splits):,}")
    print(f"history length per user: min={hist_lens.min()}, median={int(hist_lens.median())}, max={hist_lens.max()}")
    print(f"heldout movies (unique): {splits.heldout_movieId.nunique():,}")
    print(f"seed                   : {args.seed}")
    print("\nfirst 3 rows:")
    print(splits.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
