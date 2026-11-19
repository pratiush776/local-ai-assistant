"""Proportion stats reused from Unit 2 (promoted from experiments/project-2).

wilson   - 95% confidence interval for a pass rate, honest about small n.
mcnemar  - paired A-vs-B test for the 5.5 reranker on/off comparison.
"""

import math


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Point estimate and Wilson 95% CI for k successes out of n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, center - half, center + half


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Paired comparison: b_only = A right & B wrong, c_only = A wrong & B right."""
    b_only = sum(x and not y for x, y in zip(a, b))
    c_only = sum((not x) and y for x, y in zip(a, b))
    n = b_only + c_only
    chi = ((abs(b_only - c_only) - 1) ** 2) / n if n else 0.0  # continuity-corrected
    return b_only, c_only, chi
