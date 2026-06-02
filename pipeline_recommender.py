"""
Pipeline Recommender: CF + CBF retrieval -> LLM rerank.

Two-stage cascade hybrid (Burke 2002):

  Stage 1 - Retrieval (rank fusion)
    CF  retrieves 50 candidates from collaborative signal (ALS)
    CBF retrieves 50 candidates from content similarity (TF-IDF)
    Reciprocal Rank Fusion (RRF) merges both lists into ~60 candidates

  Stage 2 - Query-aware pre-filter
    The query is projected into TF-IDF space and used to trim candidates
    to the top N most semantically relevant before the LLM sees them.
    This keeps the candidate list short enough for small models to handle.

  Stage 3 - LLM Rerank
    Local Llama (via Ollama) reads the user's natural-language request,
    a taste-profile summary, plus the pre-filtered candidate list and
    returns the top-k IDs that best satisfy the constraints (mood, genre,
    similarity, exclusions, recency).

Improvements over v1:
  - RRF fusion replaces naive interleave (better candidate diversity)
  - Query-aware TF-IDF pre-filter trims candidates to top 30 before LLM
  - Taste profile (liked movie titles) injected into the rerank prompt
  - Enriched candidate descriptions include director + top cast
  - Default model bumped to llama3.1:8b for better instruction following

Closed-vocab IDs: candidates are presented as "[12] Title (year) - ..."
and the model returns just "12, 47, 891, ..." so output is unambiguous
and never hallucinates a movie outside the catalog.

A JSON cache keyed on (model, k, query_hash, candidates_hash) sits over
the LLM call so reruns of run_eval.py are instant after the first pass.

Plug into run_eval.py:
    from pipeline_recommender import PipelineRecommender
    rec = PipelineRecommender.build(ratings, splits, movies)

Smoke test (needs Ollama running with llama3.1:8b):
    python pipeline_recommender.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

from cf_recommender import CFRecommender
from cbf_recommender import CBFRecommender


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a movie recommendation reranker. The user will \
provide a natural-language request, a short taste profile showing movies \
they have enjoyed, and a numbered list of candidate movies. \
Pick the candidates that best satisfy the request, considering:

- SIMILARITY  : movies or styles the user references
- MOOD/TONE   : dark, uplifting, intense, lighthearted, etc.
- GENRE       : explicit genre constraints
- EXCLUSIONS  : things the user wants to avoid (gory, slow, etc.)
- RECENCY     : era or year preferences
- TASTE       : align with the user's demonstrated taste profile

Output rules:
- Return ONLY comma-separated IDs in ranked order. Example: 47, 12, 891, 234
- All IDs must come from the candidate list provided.
- No commentary, no explanation, no preamble.
"""

# Taste profile + enriched candidate descriptions are now part of the prompt
USER_PROMPT_TEMPLATE = """Taste profile (movies this user has enjoyed):
{liked_titles}

Request: "{query}"

Candidates:
{candidates}

Return the top {k} IDs in ranked order, comma-separated. Output IDs only."""


# --------------------------------------------------------------------------- #
# recommender
# --------------------------------------------------------------------------- #
class PipelineRecommender:
    """CF + CBF retrieve (RRF fused) -> query pre-filter -> Llama rerank."""

    name = "pipeline"

    def __init__(
        self,
        cf: CFRecommender,
        cbf: CBFRecommender,
        movies: pd.DataFrame,
        model: str = "llama3.1:8b",          # bumped from 3.2:3b
        n_per_retriever: int = 50,
        n_prefilter: int = 30,                # NEW: max candidates sent to LLM
        cache_path: str = "processed/rerank_cache.json",
        ollama_url: str = "http://localhost:11434/api/generate",
        timeout: int = 180,
    ):
        self.cf = cf
        self.cbf = cbf
        # set movieId as index for fast row lookup during prompt building
        self.movies = movies.set_index("movieId", drop=False)
        self.model = model
        self.n_per_retriever = n_per_retriever
        self.n_prefilter = n_prefilter
        self.cache_path = cache_path
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.cache = self._load_cache()

    # ---------------------------------------------------------------------- #
    # factory
    # ---------------------------------------------------------------------- #
    @classmethod
    def build(
        cls,
        ratings: pd.DataFrame,
        splits: pd.DataFrame,
        movies: pd.DataFrame,
        model: str = "llama3.1:8b",
        **kwargs,
    ) -> "PipelineRecommender":
        print("  building CF retriever...")
        cf = CFRecommender.build(ratings, splits)
        print("  building CBF retriever...")
        cbf = CBFRecommender.build(movies)
        return cls(cf=cf, cbf=cbf, movies=movies, model=model, **kwargs)

    # ---------------------------------------------------------------------- #
    # public interface (matches Recommender protocol in run_eval.py)
    # ---------------------------------------------------------------------- #
    def recommend(
        self,
        query: str,
        history_ids: list[int],
        k: int,
        liked_ids: list[int] | None = None,
    ) -> list[int]:
        """
        Return up to k movieIds.

        Args:
            query:       natural-language user request
            history_ids: full watch history (already filtered to liked by caller)
            k:           number of results to return
            liked_ids:   subset of history to use for the taste profile blurb
                         (defaults to history_ids[:5] if not provided)
        """
        candidates = self._retrieve(history_ids)
        candidates = self._query_prefilter(query, candidates)
        taste_titles = self._taste_blurb(liked_ids or history_ids)
        return self._rerank(query, candidates, taste_titles, k=k)

    # ---------------------------------------------------------------------- #
    # stage 1: retrieve + reciprocal rank fusion
    # ---------------------------------------------------------------------- #
    def _retrieve(self, history_ids: list[int]) -> list[int]:
        """
        Retrieve candidates from CF and CBF, then merge with Reciprocal Rank
        Fusion (RRF). RRF score = sum of 1/(rrf_k + rank) across retrievers,
        so items appearing high in *both* lists float to the top without
        naive concatenation bias toward the first retriever.
        """
        cf_top  = self.cf.recommend("", history_ids, k=self.n_per_retriever)
        cbf_top = self.cbf.recommend("", history_ids, k=self.n_per_retriever)

        RRF_K = 60  # standard constant; higher = less aggressive rank fusion
        scores: dict[int, float] = {}
        for rank, mid in enumerate(cf_top, 1):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (RRF_K + rank)
        for rank, mid in enumerate(cbf_top, 1):
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (RRF_K + rank)

        return sorted(scores, key=scores.__getitem__, reverse=True)

    # ---------------------------------------------------------------------- #
    # stage 2: query-aware TF-IDF pre-filter
    # ---------------------------------------------------------------------- #
    def _query_prefilter(self, query: str, candidates: list[int]) -> list[int]:
        """
        Project the query into the CBF TF-IDF space and re-score candidates
        by cosine similarity. This surfaces candidates that lexically match
        the user's mood/genre/similarity language, so the LLM sees a tight,
        relevant slate rather than a noisy 90-item dump.

        Falls back to the original RRF order if the query is empty or no
        candidates are in the CBF index.
        """
        if not query or not query.strip():
            return candidates[:self.n_prefilter]

        # vectorize the query with the already-fitted TF-IDF vocabulary
        q_vec = self.cbf._vec.transform([query])   # (1, vocab) sparse

        # gather indices for candidates that exist in the CBF index
        idxs_in_cbf = [
            (i, self.cbf._id2idx[mid])
            for i, mid in enumerate(candidates)
            if mid in self.cbf._id2idx
        ]
        if not idxs_in_cbf:
            return candidates[:self.n_prefilter]

        cand_positions, cbf_idxs = zip(*idxs_in_cbf)
        item_mat = self.cbf._mat[list(cbf_idxs)]          # (C, vocab) sparse
        sims = cosine_similarity(q_vec, item_mat).flatten()  # (C,)

        # sort by similarity descending, map back to original candidate list
        order = np.argsort(sims)[::-1]
        reordered = [candidates[cand_positions[i]] for i in order]

        # append any candidates missing from the CBF index at the end
        in_result = set(reordered)
        extras = [mid for mid in candidates if mid not in in_result]
        return (reordered + extras)[:self.n_prefilter]

    # ---------------------------------------------------------------------- #
    # taste blurb helper
    # ---------------------------------------------------------------------- #
    def _taste_blurb(self, history_ids: list[int], n: int = 5) -> str:
        """
        Build a short bulleted list of liked movie titles to inject into the
        rerank prompt so the LLM has taste context beyond the query text.
        """
        lines = []
        for mid in history_ids[:n]:
            if mid not in self.movies.index:
                continue
            row = self.movies.loc[mid]
            title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(row["title"])).strip()
            year  = int(row["year"]) if pd.notna(row.get("year")) else "?"
            lines.append(f"- {title} ({year})")
        return "\n".join(lines) if lines else "- (no history available)"

    # ---------------------------------------------------------------------- #
    # stage 3: LLM rerank
    # ---------------------------------------------------------------------- #
    def _rerank(
        self,
        query: str,
        candidates: list[int],
        taste_titles: str,
        k: int,
    ) -> list[int]:
        cache_key = self._cache_key(query, candidates, k)
        if cache_key in self.cache:
            resp = self.cache[cache_key]
        else:
            resp = self._call_ollama(query, candidates, taste_titles, k)
            self.cache[cache_key] = resp
            self._save_cache()
        return self._parse(resp, candidate_order=candidates, k=k)

    def _format_candidates(self, candidate_ids: list[int]) -> str:
        """
        Build the candidate list string for the prompt.
        Includes director and top-2 cast members for stronger reranking signal
        (auteur and star names often map directly to mood/genre constraints).
        """
        lines = []
        for mid in candidate_ids:
            if mid not in self.movies.index:
                continue
            row = self.movies.loc[mid]

            title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(row["title"])).strip()
            year  = int(row["year"]) if pd.notna(row.get("year")) else "?"

            # genres
            g = row.get("genres")
            g_list = list(g) if g is not None and hasattr(g, "__iter__") and not isinstance(g, str) else []
            genres = ", ".join(g_list[:3]) if g_list else "Unknown"

            # director (NEW)
            director = str(row.get("director") or "").strip()
            dir_str  = f"Dir: {director}" if director else ""

            # top-2 cast (NEW)
            cast = row.get("cast")
            cast_list = list(cast) if cast is not None and hasattr(cast, "__iter__") and not isinstance(cast, str) else []
            cast_str  = "Cast: " + ", ".join(cast_list[:2]) if cast_list else ""

            # overview
            overview = str(row.get("overview") or "").strip().replace("\n", " ")
            if len(overview) > 100:
                overview = overview[:97] + "..."

            meta_parts = [p for p in [genres, dir_str, cast_str] if p]
            meta = " | ".join(meta_parts)
            lines.append(f"[{mid}] {title} ({year}) - {meta} - {overview}")
        return "\n".join(lines)

    def _call_ollama(
        self,
        query: str,
        candidates: list[int],
        taste_titles: str,
        k: int,
    ) -> str:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            liked_titles=taste_titles,
            query=query,
            candidates=self._format_candidates(candidates),
            k=k,
        )
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            # low temperature: this is ranking, not creative writing
            "options": {"temperature": 0.2, "num_predict": 500},
        }
        r = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def _parse(self, resp: str, candidate_order: list[int], k: int) -> list[int]:
        """Extract IDs from the response, filter to valid candidates, dedup,
        and pad with retrieval order if the LLM returned fewer than k.
        """
        candidate_set = set(candidate_order)
        ids = [int(x) for x in re.findall(r"\d+", resp)]
        seen: set[int] = set()
        ranked: list[int] = []
        for mid in ids:
            if mid in candidate_set and mid not in seen:
                ranked.append(mid)
                seen.add(mid)
            if len(ranked) >= k:
                break
        # fallback: pad with leftover retrieval candidates in RRF order
        if len(ranked) < k:
            for mid in candidate_order:
                if mid not in seen:
                    ranked.append(mid)
                    seen.add(mid)
                if len(ranked) >= k:
                    break
        return ranked

    # ---------------------------------------------------------------------- #
    # cache
    # ---------------------------------------------------------------------- #
    def _cache_key(self, query: str, candidates: list[int], k: int) -> str:
        h_q = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
        h_c = hashlib.sha1(",".join(map(str, candidates)).encode("utf-8")).hexdigest()[:12]
        return f"{self.model}|k={k}|{h_q}|{h_c}"

    def _load_cache(self) -> dict:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
def _smoke_test() -> None:
    RATINGS_PATH  = "processed/ratings.parquet"
    SPLITS_PATH   = "processed/splits.parquet"
    MOVIES_PATH   = "processed/movies_enriched.parquet"
    EVAL_SET_PATH = "processed/eval_set.parquet"

    HIGH_RATING = 4
    MIN_LIKED = 3
    N_USERS = 3

    print("Loading data...")
    ratings  = pd.read_parquet(RATINGS_PATH)
    splits   = pd.read_parquet(SPLITS_PATH)
    movies   = pd.read_parquet(MOVIES_PATH)
    eval_set = pd.read_parquet(EVAL_SET_PATH)
    eval_set = eval_set[eval_set["status"] == "ok"].head(N_USERS)
    if eval_set.empty:
        raise SystemExit("no 'ok' rows in eval_set.parquet -- run make_queries first")

    high = ratings[ratings.rating >= HIGH_RATING]
    user_likes = high.groupby("userId")["movieId"].apply(set).to_dict()

    print(f"\nBuilding pipeline (this also fits CF + CBF)...")
    t0 = time.time()
    rec = PipelineRecommender.build(ratings, splits, movies)
    print(f"  done in {time.time()-t0:.1f}s")

    movie_title = movies.set_index("movieId")["title"].to_dict()

    print(f"\nReranking {N_USERS} users...\n" + "=" * 70)
    for r in eval_set.itertuples(index=False):
        liked = [m for m in r.history_movieIds if m in user_likes.get(r.userId, set())]
        history = liked if len(liked) >= MIN_LIKED else list(r.history_movieIds)

        t0 = time.time()
        ranked = rec.recommend(r.query, history, k=20, liked_ids=history[:5])
        dt = time.time() - t0

        heldout_title = movie_title.get(int(r.heldout_movieId), "???")
        in_top20 = int(r.heldout_movieId) in ranked
        rank = ranked.index(int(r.heldout_movieId)) + 1 if in_top20 else None

        print(f"\nuser {r.userId}  ({dt:.1f}s)")
        print(f"  query   : {r.query!r}")
        print(f"  heldout : {heldout_title}  -> {'rank ' + str(rank) if in_top20 else 'NOT in top 20'}")
        print(f"  top 5   :")
        for i, mid in enumerate(ranked[:5], start=1):
            print(f"    {i}. [{mid}] {movie_title.get(mid, '???')}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        _smoke_test()
    except requests.exceptions.ConnectionError:
        raise SystemExit("ERROR: Ollama daemon not running. Start it with: ollama serve")
