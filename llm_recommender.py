"""
Pure LLM recommender baseline.

Given a user's natural-language query and their watch history, asks the LLM
to recommend K movies directly — no retrieval, no candidate set. The LLM
must generate movie titles from memory, which we then fuzzy-match back to
movieIds in the ML-1M catalog.

This baseline answers the question: "how much does retrieval actually help?"
If LLM-only scores comparably to CF+LLM-reranking, retrieval isn't adding
much. If it's much lower, retrieval is doing real work.

Prereqs: ollama serve && ollama pull llama3.1:8b

Usage (standalone smoke test):
    python llm_recommender.py --limit 10

Plug into run_eval_baselines.py:
    from llm_recommender import LLMRecommender
    rec = LLMRecommender(movies_df)
    # rec.recommend(query, history_ids, k) -> list[int]
"""

from __future__ import annotations

import argparse
import re
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are a movie recommendation assistant. Given a user's \
watch history and their request, recommend movies they haven't seen.

Rules:
- Output ONLY a numbered list of movie titles, one per line, like:
  1. Movie Title (Year)
  2. Another Movie (Year)
- Include the release year in parentheses when you know it
- Do NOT include any explanation, preamble, or commentary
- Do NOT recommend movies already in the user's watch history"""


def _build_prompt(query: str, history_titles: list[str], k: int) -> str:
    history_block = "\n".join(f"- {t}" for t in history_titles[:15])
    return (
        f"User's watch history (recent favorites):\n{history_block}\n\n"
        f"User's request: {query}\n\n"
        f"Recommend {k} movies they haven't seen."
    )


def _parse_titles(response: str) -> list[str]:
    """Extract movie titles from a numbered list response."""
    titles = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # strip leading "1." / "1)" / "-" / "*"
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        # strip trailing year "(1999)" or "[1999]"
        line = re.sub(r"[\(\[]\d{4}[\)\]]$", "", line).strip()
        # strip trailing notes after " - " or ":"
        line = re.sub(r"\s*[-:–]\s.*$", "", line).strip()
        if line:
            titles.append(line)
    return titles


def _fuzzy_match(title: str, catalog: pd.DataFrame, threshold: float = 0.7) -> int | None:
    """
    Match a generated title string to a movieId in the catalog.

    Strategy:
    1. Exact match (after lowercasing + stripping year)
    2. SequenceMatcher ratio >= threshold as fallback
    """
    query_clean = re.sub(r"\s*\(\d{4}\)\s*$", "", title).lower().strip()

    # pre-computed clean titles on first call (stored on the dataframe)
    if "_title_clean" not in catalog.columns:
        catalog["_title_clean"] = (
            catalog["title"]
            .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
            .str.lower()
            .str.strip()
        )

    # exact
    exact = catalog[catalog["_title_clean"] == query_clean]
    if not exact.empty:
        return int(exact.iloc[0]["movieId"])

    # fuzzy
    best_score, best_id = 0.0, None
    for _, row in catalog.iterrows():
        score = SequenceMatcher(None, query_clean, row["_title_clean"]).ratio()
        if score > best_score:
            best_score, best_id = score, int(row["movieId"])

    return best_id if best_score >= threshold else None


# --------------------------------------------------------------------------- #
# recommender
# --------------------------------------------------------------------------- #
class LLMRecommender:
    """
    Pure LLM baseline: the model recommends from memory, no retrieval.

    Args:
        movies:  movies_enriched DataFrame (needs movieId, title columns)
        model:   Ollama model tag
        temperature: generation temperature (higher = more diverse but noisier)
        fuzzy_threshold: minimum SequenceMatcher ratio to accept a title match
    """

    name = "llm_pure"

    def __init__(
        self,
        movies: pd.DataFrame,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,   # lower than query-gen; we want consistent titles
        fuzzy_threshold: float = 0.7,
    ):
        self._movies = movies.copy()
        self._model = model
        self._temperature = temperature
        self._threshold = fuzzy_threshold
        # build a title -> movieId lookup once
        self._id2title: dict[int, str] = dict(zip(movies["movieId"], movies["title"]))

    def _history_titles(self, history_ids: list[int]) -> list[str]:
        return [
            re.sub(r"\s*\(\d{4}\)\s*$", "", self._id2title[mid]).strip()
            for mid in history_ids
            if mid in self._id2title
        ]

    def recommend(self, query: str, history_ids: list[int], k: int) -> list[int]:
        """
        Ask the LLM for k recommendations, parse titles, fuzzy-match to movieIds.
        Returns however many could be matched (may be fewer than k).
        """
        seen = set(history_ids)
        history_titles = self._history_titles(history_ids)

        prompt = _build_prompt(query or "Recommend movies I'd enjoy.", history_titles, k)

        payload = {
            "model": self._model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._temperature, "num_predict": 300},
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        raw = r.json().get("response", "").strip()

        titles = _parse_titles(raw)

        ranked = []
        for title in titles:
            mid = _fuzzy_match(title, self._movies, threshold=self._threshold)
            if mid is not None and mid not in seen and mid not in ranked:
                ranked.append(mid)
            if len(ranked) >= k:
                break

        return ranked


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    MOVIES_PATH = "processed/movies_enriched.parquet"
    EVAL_SET_PATH = "processed/eval_set.parquet"
    SPLITS_PATH = "processed/splits.parquet"

    movies = pd.read_parquet(MOVIES_PATH)
    rec = LLMRecommender(movies, model=args.model)

    # use eval_set if available (has queries), else fall back to splits
    import os
    if os.path.exists(EVAL_SET_PATH):
        eval_df = pd.read_parquet(EVAL_SET_PATH)
        if "status" in eval_df.columns:
            eval_df = eval_df[eval_df["status"] == "ok"]
    else:
        eval_df = pd.read_parquet(SPLITS_PATH)
        eval_df["query"] = ""

    eval_df = eval_df.head(args.limit)

    from metrics import score_all
    all_scores, match_rates = [], []
    t0 = time.time()

    for i, row in enumerate(eval_df.itertuples(index=False), 1):
        query = getattr(row, "query", "") or ""
        ranked = rec.recommend(query, list(row.history_movieIds), k=args.k)
        match_rates.append(len(ranked) / args.k)
        scores = score_all(ranked, int(row.heldout_movieId))
        all_scores.append(scores)
        hit = "HIT " if int(row.heldout_movieId) in ranked else "miss"
        print(f"[{i:>2}/{args.limit}] {hit}  matched {len(ranked)}/{args.k} titles  "
              f"MRR={scores['MRR']:.3f}  u={row.userId}")

    agg = pd.DataFrame(all_scores).mean()
    print(f"\nLLM-pure (n={args.limit}, k={args.k})  "
          f"avg title match rate: {np.mean(match_rates):.2f}")
    for m, v in agg.items():
        print(f"  {m:<10s} {v:.4f}")
    print(f"\n({time.time()-t0:.0f}s total)")
