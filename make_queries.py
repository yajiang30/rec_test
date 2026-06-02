"""
Step 2C: Generate one natural-language query per user via a local Ollama model.

For each row in processed/splits.parquet, sample up to N high-rated movies
from that user's history (the held-out movie is already excluded by 2A),
ask the LLM to write a realistic single-turn chat request, run sanity
checks, and save the result.

Output: processed/eval_set.parquet  --  one row per user, schema:
    userId | query | history_movieIds | heldout_movieId

Prereqs:  brew install ollama && ollama serve && ollama pull llama3.1:8b

Quick smoke test (no API needed):
    python make_queries.py --dry-run --limit 3
"""

import argparse
import os
import random
import re
import sys
import time
from typing import Optional

import pandas as pd
import requests

SPLITS_PATH = "processed/splits.parquet"
MOVIES_PATH = "processed/movies_enriched.parquet"
RATINGS_PATH = "processed/ratings.parquet"
OUT_PATH = "processed/eval_set.parquet"

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"
SAMPLE_HISTORY = 8           # how many history movies to show the LLM
HIGH_RATING = 4              # match split.py


SYSTEM_PROMPT = """You are simulating a real moviegoer sending a single chat \
message to a movie recommendation assistant. Write ONE short, natural-sounding \
request (1-3 sentences) for a movie to watch tonight.

The request MUST:
- Sound like a chat message, not a survey response (contractions, casual tone)
- Include AT LEAST TWO of these constraint types, naturally woven in:
    * SIMILARITY  - "like X" / "in the vein of Y"
    * RECENCY     - "recent" / "from the last few years" / "older"
    * TONE/MOOD   - "funny but not childish" / "uplifting" / "dark"
    * EXCLUSION   - "not too violent" / "nothing slow"
    * GENRE       - "sci-fi" / "psychological thriller"
- Reference 1 to 3 movies from the taste history as anchors

The request MUST NOT:
- Name a specific movie as the recommendation (the assistant gives those)
- Mention more than 3 of the listed movies
- Reveal it was generated from a list (no "based on the movies provided...")
- Start with "As an AI" or similar

Output ONLY the chat message text, nothing else."""


def build_user_prompt(history_rows: pd.DataFrame) -> str:
    lines = ["Here are some movies this user has rated highly:"]
    for _, m in history_rows.iterrows():
        g = m["genres"]
        g_list = list(g) if g is not None and hasattr(g, "__iter__") and not isinstance(g, str) else []
        genres = ", ".join(g_list[:3]) if g_list else "Unknown"
        year = int(m["year"]) if pd.notna(m["year"]) else "?"
        clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(m["title"])).strip()
        lines.append(f"- {clean_title} ({year}) [{genres}]")
    lines.append("\nWrite the chat message.")
    return "\n".join(lines)


def call_ollama(model: str, system: str, user: str, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {"temperature": 0.8, "num_predict": 200},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def sanity_check(query: str, history_titles: list[str], heldout_title: str) -> Optional[str]:
    """Return a failure reason if the query is bad, else None."""
    q = query.lower()
    if not q:
        return "empty"
    if not (15 <= len(query) <= 600):
        return f"bad length ({len(query)})"
    # leakage: held-out movie title (or its core words) literally appears
    ho_core = re.sub(r"\s*\(\d{4}\)\s*$", "", heldout_title).lower()
    if ho_core and ho_core in q and len(ho_core) >= 4:
        return f"leaks heldout title: {ho_core!r}"
    if q.startswith(("as an ai", "based on the movies", "i am an ai")):
        return "robotic preamble"
    # require at least one history anchor
    anchors = sum(1 for t in history_titles if re.sub(r"\s*\(\d{4}\)\s*$", "", t).lower() in q)
    if anchors == 0:
        return "no history anchor mentioned"
    return None


def generate_for_user(uid: int, history_ids: list[int], heldout_id: int,
                      movies: pd.DataFrame, high_rated_ids: set[int], model: str,
                      rng: random.Random, dry_run: bool) -> dict:
    # restrict to movies this user actually liked (>=4 stars); fall back to all if too few
    liked = [m for m in history_ids if m in high_rated_ids]
    pool = liked if len(liked) >= 3 else history_ids
    sampled_ids = rng.sample(pool, min(SAMPLE_HISTORY, len(pool)))
    sampled = movies[movies.movieId.isin(sampled_ids)]
    user_prompt = build_user_prompt(sampled)

    heldout_title = movies.loc[movies.movieId == heldout_id, "title"].iloc[0]
    history_titles = sampled["title"].tolist()

    if dry_run:
        return {"userId": uid, "prompt": user_prompt, "query": None, "status": "DRY"}

    query = call_ollama(model, SYSTEM_PROMPT, user_prompt)
    reason = sanity_check(query, history_titles, heldout_title)
    return {"userId": uid, "query": query, "status": "ok" if reason is None else f"FAIL: {reason}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="cap users for a quick run")
    ap.add_argument("--dry-run", action="store_true", help="build prompts but don't call the LLM")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    splits = pd.read_parquet(SPLITS_PATH)
    movies = pd.read_parquet(MOVIES_PATH)
    ratings = pd.read_parquet(RATINGS_PATH)
    if args.limit:
        splits = splits.head(args.limit)

    # per-user set of movies they rated >=4 (for query anchoring)
    high = ratings[ratings.rating >= HIGH_RATING]
    user_likes = high.groupby("userId")["movieId"].apply(set).to_dict()

    rng = random.Random(args.seed)
    results, oks, fails = [], 0, 0
    t0 = time.time()

    for i, row in enumerate(splits.itertuples(index=False), start=1):
        out = generate_for_user(
            row.userId, list(row.history_movieIds), int(row.heldout_movieId),
            movies, user_likes.get(row.userId, set()),
            args.model, rng, args.dry_run,
        )
        results.append(out)
        if args.dry_run:
            print(f"\n--- user {row.userId} (DRY) ---\n{out['prompt']}")
        else:
            ok = out["status"] == "ok"
            oks += int(ok); fails += int(not ok)
            tag = "OK  " if ok else "FAIL"
            print(f"[{i:>3}/{len(splits)}] {tag} u={row.userId}: {out['query'][:110]!r}  ({out['status']})")

    if args.dry_run:
        return

    df = pd.DataFrame(results)
    df = df.merge(splits[["userId", "history_movieIds", "heldout_movieId"]], on="userId")
    df = df[["userId", "query", "status", "history_movieIds", "heldout_movieId"]]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved -> {OUT_PATH}  ({oks} ok, {fails} fail, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        sys.exit("ERROR: Ollama daemon not running. Start it with: ollama serve")
