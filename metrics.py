"""
Step 2B: Ranking metrics for single-held-out-positive evaluation.

Every metric here takes a ranked list of movieIds and the one ground-truth
heldout id; nothing else. With a single positive, the standard formulas
collapse to simple closed forms:

    HR@K    = 1 if heldout in top K, else 0
    MRR     = 1 / rank if found, else 0          (rank is 1-indexed)
    NDCG@K  = 1 / log2(rank + 1) if rank <= K, else 0

Run `python metrics.py` for self-tests.
"""

from __future__ import annotations

import math
from typing import Iterable


def _rank(ranked: Iterable[int], target: int) -> int | None:
    """1-indexed rank of `target` in `ranked`, or None if absent."""
    for i, mid in enumerate(ranked, start=1):
        if mid == target:
            return i
    return None


def hit_rate(ranked: Iterable[int], heldout: int, k: int) -> float:
    r = _rank(list(ranked)[:k], heldout)
    return 1.0 if r is not None else 0.0


def mrr(ranked: Iterable[int], heldout: int) -> float:
    r = _rank(ranked, heldout)
    return 1.0 / r if r is not None else 0.0


def ndcg(ranked: Iterable[int], heldout: int, k: int) -> float:
    r = _rank(list(ranked)[:k], heldout)
    return 1.0 / math.log2(r + 1) if r is not None else 0.0


def score_all(ranked: Iterable[int], heldout: int, ks=(5, 10, 20)) -> dict:
    """Convenience: compute the full set of metrics in one shot."""
    ranked = list(ranked)
    out = {"MRR": mrr(ranked, heldout)}
    for k in ks:
        out[f"HR@{k}"] = hit_rate(ranked, heldout, k)
        out[f"NDCG@{k}"] = ndcg(ranked, heldout, k)
    return out


# --------------------------------------------------------------------------- #
# self-tests
# --------------------------------------------------------------------------- #
def _approx(a, b, eps=1e-9):
    assert abs(a - b) < eps, f"expected {b}, got {a}"


def _run_tests():
    ranked = [10, 20, 30, 40, 50]  # heldout=30 is at rank 3

    _approx(hit_rate(ranked, 30, k=5), 1.0)
    _approx(hit_rate(ranked, 30, k=2), 0.0)         # not in top 2
    _approx(hit_rate(ranked, 999, k=5), 0.0)        # missing entirely

    _approx(mrr(ranked, 30), 1 / 3)
    _approx(mrr(ranked, 10), 1.0)
    _approx(mrr(ranked, 999), 0.0)

    _approx(ndcg(ranked, 10, k=5), 1.0)             # rank 1 -> log2(2)=1
    _approx(ndcg(ranked, 30, k=5), 1 / math.log2(4))
    _approx(ndcg(ranked, 30, k=2), 0.0)             # outside top K
    _approx(ndcg(ranked, 999, k=5), 0.0)

    bundle = score_all(ranked, 30, ks=(5, 10))
    _approx(bundle["MRR"], 1 / 3)
    _approx(bundle["HR@5"], 1.0)
    _approx(bundle["NDCG@5"], 1 / math.log2(4))
    print("metrics.py: all tests passed")


if __name__ == "__main__":
    _run_tests()
