"""Statistical helpers: known values, edge cases and the honesty rules."""
import math

from cs2_pipeline.stats import (benjamini_hochberg, binomial_p, confidence_label, summarize_rate,
                                wilson_ci)


def test_wilson_matches_reference_values():
    lo, hi = wilson_ci(8, 10)          # textbook example: 80% of 10
    assert math.isclose(lo, 0.4902, abs_tol=1e-3) and math.isclose(hi, 0.9433, abs_tol=1e-3)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.31   # never a zero-width interval at the boundary
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and 0.69 < lo < 0.75


def test_wilson_edge_cases_and_monotonicity():
    lo, hi = wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)
    widths = [wilson_ci(n // 2, n)[1] - wilson_ci(n // 2, n)[0] for n in (10, 40, 160, 640)]
    assert widths == sorted(widths, reverse=True), "more data must narrow the interval"


def test_confidence_labels_follow_sample_size():
    assert [confidence_label(n) for n in (0, 9, 10, 29, 30, 99, 100)] == [
        "insufficient", "insufficient", "low", "low", "medium", "medium", "high"]


def test_small_samples_are_never_called_findings():
    s = summarize_rate(9, 9, baseline=0.5)     # 9/9 looks perfect but n < 10
    assert s.rate == 1.0 and s.verdict == "inconclusive" and s.confidence == "insufficient"


def test_verdict_requires_the_interval_to_clear_the_baseline():
    assert summarize_rate(80, 100).verdict == "above"
    assert summarize_rate(20, 100).verdict == "below"
    assert summarize_rate(26, 50).verdict == "inconclusive"       # 52% of 50 is noise
    assert summarize_rate(14, 24, baseline=0.5).verdict == "inconclusive"  # one match of rounds
    assert summarize_rate(64, 100, baseline=0.65).verdict == "inconclusive"


def test_binomial_p_matches_exact_reference_values():
    assert math.isclose(binomial_p(8, 10, 0.5), 0.109375, abs_tol=1e-12)
    assert math.isclose(binomial_p(10, 10, 0.5), 0.001953125, abs_tol=1e-12)
    assert math.isclose(binomial_p(0, 10, 0.5), 0.001953125, abs_tol=1e-12)
    # The sum of every pmf term is 1 up to floating-point error, never exactly 1.
    assert math.isclose(binomial_p(5, 10, 0.5), 1.0, abs_tol=1e-12)
    assert math.isclose(binomial_p(3, 10, 0.3), 1.0, abs_tol=1e-12)
    # Worked by hand: P(X=6)=0.03675, and the counts no likelier than it are 0 and 6..10.
    assert math.isclose(binomial_p(6, 10, 0.3), 0.0756, abs_tol=1e-4)


def test_binomial_p_edge_cases_and_large_n():
    assert math.isnan(binomial_p(0, 0))
    assert binomial_p(1, 10, 0.0) == 0.0 and binomial_p(9, 10, 1.0) == 0.0
    assert 0.0 < binomial_p(600, 1000, 0.5) < 1e-9        # no overflow at large n
    assert math.isclose(binomial_p(500, 1000, 0.5), 1.0, abs_tol=1e-9)
    # More of the same evidence can only make the result less likely under the baseline.
    assert binomial_p(16, 20, 0.5) < binomial_p(8, 10, 0.5) < binomial_p(4, 5, 0.5)


def test_benjamini_hochberg_step_up():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.5]
    rejected, q = benjamini_hochberg(p, 0.05)
    assert rejected == [True, True, False, False, False, False, False]
    assert math.isclose(q[0], 0.007) and math.isclose(q[1], 0.028)
    assert q == sorted(q), "q-values must be monotone in the p-value order"
    # Every p-value tiny: everything is rejected. Every p-value large: nothing is.
    assert all(benjamini_hochberg([1e-6] * 5, 0.05)[0])
    assert not any(benjamini_hochberg([0.5] * 5, 0.05)[0])


def test_benjamini_hochberg_is_stricter_than_testing_each_rate_alone():
    # One rate at p=0.04 would pass on its own; among 19 uninteresting ones it does not.
    p = [0.04] + [0.9] * 19
    rejected, q = benjamini_hochberg(p, 0.05)
    assert not any(rejected) and math.isclose(q[0], 0.8)
    # Twenty rates all at 0.04 are a different case: BH does reject them, allowing 5% of the
    # reported findings to be false, which is exactly the guarantee it makes.
    assert all(benjamini_hochberg([0.04] * 20, 0.05)[0])


def test_benjamini_hochberg_ignores_untested_rows():
    rejected, q = benjamini_hochberg([float("nan"), 0.001, 0.9])
    assert rejected == [False, True, False]
    assert math.isnan(q[0]) and math.isclose(q[1], 0.002)
    assert benjamini_hochberg([float("nan")] * 3) == ([False] * 3, [float("nan")] * 3) or True
    rej, qq = benjamini_hochberg([float("nan")] * 3)
    assert rej == [False] * 3 and all(math.isnan(x) for x in qq)


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
