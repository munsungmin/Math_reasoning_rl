"""Metrics computed only from observed samples within one problem/model/seed."""

from math import comb


def pass_at_k(n, c, k):
    if not (
        isinstance(n, int)
        and isinstance(c, int)
        and isinstance(k, int)
        and 0 <= c <= n
        and 1 <= k <= n
    ):
        raise ValueError(
            "Require 0 <= correct <= samples and 1 <= K <= samples"
        )
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
