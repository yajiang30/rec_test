"""
Step 1: Unified data layer for the conversational movie recommender.

Joins MovieLens 1M ratings to rich TMDB metadata via GroupLens' links.csv:

    ml-1m movieId  --links.csv-->  tmdbId  --movies_metadata/keywords/credits-->  metadata

Produces two artifacts so nothing has to be re-parsed on every run:
    processed/movies_enriched.(parquet|pkl)  - one row per ML-1M movie + metadata
    processed/ratings.(parquet|pkl)          - the full 1M ratings table

Run:  python build_data.py
"""

import ast
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

ML_DIR = "movielen1m"
TMDB_DIR = "themoviesdataset"
OUT_DIR = "processed"

TOP_CAST_N = 5  # how many lead actors to keep per movie


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_names(cell, limit=None):
    """Parse a TMDB list-of-dicts string like "[{'name': 'Comedy', ...}]" -> ['Comedy', ...]."""
    if not isinstance(cell, str) or not cell.startswith("["):
        return []
    try:
        items = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    names = [d.get("name") for d in items if isinstance(d, dict) and d.get("name")]
    return names[:limit] if limit else names


def parse_director(crew_cell):
    """Pull the director name out of a TMDB crew list-of-dicts string."""
    if not isinstance(crew_cell, str) or not crew_cell.startswith("["):
        return None
    try:
        crew = ast.literal_eval(crew_cell)
    except (ValueError, SyntaxError):
        return None
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name")
    return None


def save(df, name):
    """Write parquet if an engine is available, otherwise fall back to pickle."""
    path = os.path.join(OUT_DIR, f"{name}.parquet")
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = os.path.join(OUT_DIR, f"{name}.pkl")
        df.to_pickle(path)
        return path


# --------------------------------------------------------------------------- #
# load MovieLens 1M
# --------------------------------------------------------------------------- #
def load_movielens():
    movies = pd.read_csv(
        os.path.join(ML_DIR, "movies.dat"),
        sep="::", engine="python", encoding="latin-1",
        names=["movieId", "title_ml", "genres_ml"],
    )
    # split "Toy Story (1995)" -> title + year
    movies["year"] = movies["title_ml"].str.extract(r"\((\d{4})\)").astype("Int64")
    movies["genres_ml"] = movies["genres_ml"].str.split("|")

    ratings = pd.read_csv(
        os.path.join(ML_DIR, "ratings.dat"),
        sep="::", engine="python", encoding="latin-1",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    return movies, ratings


# --------------------------------------------------------------------------- #
# load + flatten TMDB metadata keyed by tmdbId
# --------------------------------------------------------------------------- #
def load_tmdb():
    meta = pd.read_csv(os.path.join(TMDB_DIR, "movies_metadata.csv"), low_memory=False)
    meta = meta[pd.to_numeric(meta["id"], errors="coerce").notna()].copy()
    meta["tmdbId"] = meta["id"].astype(int)
    meta = meta.drop_duplicates("tmdbId")
    meta["genres"] = meta["genres"].apply(parse_names)
    meta = meta[["tmdbId", "overview", "genres", "vote_average", "runtime"]]

    keywords = pd.read_csv(os.path.join(TMDB_DIR, "keywords.csv"))
    keywords["tmdbId"] = keywords["id"].astype(int)
    keywords = keywords.drop_duplicates("tmdbId")
    keywords["keywords"] = keywords["keywords"].apply(parse_names)
    keywords = keywords[["tmdbId", "keywords"]]

    credits = pd.read_csv(os.path.join(TMDB_DIR, "credits.csv"))
    credits["tmdbId"] = credits["id"].astype(int)
    credits = credits.drop_duplicates("tmdbId")
    credits["cast"] = credits["cast"].apply(lambda c: parse_names(c, limit=TOP_CAST_N))
    credits["director"] = credits["crew"].apply(parse_director)
    credits = credits[["tmdbId", "cast", "director"]]

    return meta.merge(keywords, on="tmdbId", how="left").merge(credits, on="tmdbId", how="left")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    movies, ratings = load_movielens()
    links = pd.read_csv(os.path.join(TMDB_DIR, "links.csv"))[["movieId", "tmdbId"]]
    tmdb = load_tmdb()

    # movieId -> tmdbId -> metadata
    enriched = (
        movies.merge(links, on="movieId", how="left")
        .merge(tmdb, on="tmdbId", how="left")
    )
    # prefer MovieLens genres (clean pipe-delimited) but keep TMDB as fallback
    enriched["genres"] = [
        ml if isinstance(ml, list) and ml != ["(no genres listed)"] else (tm if isinstance(tm, list) else [])
        for ml, tm in zip(enriched["genres_ml"], enriched["genres"])
    ]
    enriched = enriched.rename(columns={"title_ml": "title"}).drop(columns=["genres_ml"])
    enriched = enriched[
        ["movieId", "tmdbId", "title", "year", "genres",
         "overview", "keywords", "cast", "director", "vote_average", "runtime"]
    ]

    p_movies = save(enriched, "movies_enriched")
    p_ratings = save(ratings, "ratings")

    # ---- coverage report ----
    has_meta = enriched["overview"].notna()
    rated_with_meta = ratings["movieId"].isin(set(enriched.loc[has_meta, "movieId"]))
    print("=" * 60)
    print("STEP 1 COMPLETE — data layer built")
    print("=" * 60)
    print(f"movies            : {len(enriched):,}")
    print(f"  with metadata   : {has_meta.sum():,} ({has_meta.mean()*100:.1f}%)")
    print(f"ratings           : {len(ratings):,}")
    print(f"  with metadata   : {rated_with_meta.sum():,} ({rated_with_meta.mean()*100:.1f}%)")
    print(f"users             : {ratings['userId'].nunique():,}")
    print(f"\nsaved -> {p_movies}\n         {p_ratings}")
    print("\nsample enriched rows:")
    with pd.option_context("display.max_colwidth", 28, "display.width", 120):
        print(enriched[["movieId", "title", "year", "genres", "cast", "director"]].head(4).to_string(index=False))


if __name__ == "__main__":
    main()
