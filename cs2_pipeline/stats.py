"""Small statistics helpers shared by the Gold layer.

Esports samples are tiny (one match is ~24 rounds), so every rate that reaches a
user travels with its sample size, a confidence interval and a plain-language
confidence label. A rate is only called a *finding* when its interval clears the
baseline it is compared against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Minimum sample sizes (events) for each confidence label.
MIN_N_LOW = 10
MIN_N_MEDIUM = 30
MIN_N_HIGH = 100


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; well-behaved for small n and p near 0/1."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def confidence_label(n: int) -> str:
    """'insufficient' | 'low' | 'medium' | 'high', from the number of events behind a rate."""
    if n < MIN_N_LOW:
        return "insufficient"
    if n < MIN_N_MEDIUM:
        return "low"
    if n < MIN_N_HIGH:
        return "medium"
    return "high"


@dataclass(frozen=True)
class RateSummary:
    successes: int
    n: int
    rate: float
    ci_low: float
    ci_high: float
    baseline: float
    lift: float
    confidence: str
    #: "above" / "below" when the whole interval is on one side of the baseline, else "inconclusive".
    verdict: str

    def asdict(self) -> dict:
        return self.__dict__.copy()


def summarize_rate(successes: int, n: int, baseline: float = 0.5) -> RateSummary:
    """Rate, Wilson interval, lift over `baseline` and whether the data can tell them apart."""
    lo, hi = wilson_ci(successes, n)
    rate = successes / n if n else float("nan")
    label = confidence_label(n)
    if label == "insufficient" or n == 0:
        verdict = "inconclusive"
    elif lo > baseline:
        verdict = "above"
    elif hi < baseline:
        verdict = "below"
    else:
        verdict = "inconclusive"
    return RateSummary(int(successes), int(n), rate, lo, hi, baseline,
                       rate - baseline if n else float("nan"), label, verdict)


def _log_pmf(k: int, n: int, p: float) -> float:
    """log P(X = k) for X ~ Binomial(n, p), in log space so large n cannot overflow."""
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binomial_p(successes: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial test: P(an outcome at least as unlikely as this one).

    Sums the probability of every count whose likelihood is no greater than the observed
    one, which is the usual exact two-sided test and needs no normal approximation - the
    samples here are far too small for one.
    """
    if n <= 0:
        return float("nan")
    p0 = min(max(p0, 0.0), 1.0)
    obs = _log_pmf(successes, n, p0)
    if obs == -math.inf:          # the observation is impossible under p0 (p0 of 0 or 1)
        return 0.0
    total = 0.0
    for k in range(n + 1):
        lp = _log_pmf(k, n, p0)
        if lp <= obs + 1e-9:      # tolerance so ties on either side are both counted
            total += math.exp(lp)
    return min(1.0, total)


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg step-up: ``(rejected, q_values)`` in the order given.

    Controls the expected share of false findings among those reported, which is the right
    guarantee when one run tests dozens of team/map/side combinations at once. ``nan``
    p-values are not part of the family: they are never rejected and keep a ``nan`` q-value.

    The slices tested here are nested (a team's overall rate re-uses the rounds of its
    per-map rates), so the p-values are positively correlated rather than independent. BH
    holds under positive dependence of this kind, but the honest check on the whole chain is
    the permutation test in the suite, which measures the rate on data with no real effect.
    """
    tested = [(i, p) for i, p in enumerate(pvalues) if not (p is None or math.isnan(p))]
    rejected = [False] * len(pvalues)
    q = [float("nan")] * len(pvalues)
    if not tested:
        return rejected, q
    m = len(tested)
    tested.sort(key=lambda ip: ip[1])
    running = 1.0
    for rank in range(m, 0, -1):                 # step up, enforcing monotone q-values
        i, p = tested[rank - 1]
        running = min(running, p * m / rank)
        q[i] = running
        rejected[i] = running <= alpha
    return rejected, q


def _log_hypergeom(k: int, n: int, successes: int, draws: int) -> float:
    """log P(X = k) for the hypergeometric distribution, in log space."""
    def lc(a: int, b: int) -> float:
        if b < 0 or b > a:
            return -math.inf
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)
    num = lc(successes, k) + lc(n - successes, draws - k)
    return num - lc(n, draws) if num != -math.inf else -math.inf


def fisher_exact_p(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on ``[[a, b], [c, d]]``.

    Use this instead of :func:`binomial_p` whenever the rate is compared with another rate
    measured from data (a 'field' baseline) rather than with a known constant: the baseline
    carries its own sampling error, and treating it as exact makes p-values far too small.
    Row 1 is the team (``a`` successes, ``b`` failures), row 2 the field.
    """
    if min(a, b, c, d) < 0:
        raise ValueError("counts must not be negative")
    n = a + b + c + d
    row1, col1 = a + b, a + c
    if n == 0 or row1 == 0 or col1 == 0 or row1 == n or col1 == n:
        return 1.0                     # no contrast to test: one margin is degenerate
    obs = _log_hypergeom(a, n, col1, row1)
    total = 0.0
    for k in range(max(0, col1 - (c + d)), min(row1, col1) + 1):
        lp = _log_hypergeom(k, n, col1, row1)
        if lp <= obs + 1e-9:
            total += math.exp(lp)
    return min(1.0, total)
