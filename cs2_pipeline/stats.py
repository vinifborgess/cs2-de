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
