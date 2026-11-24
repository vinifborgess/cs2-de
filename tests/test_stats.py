"""Statistical helpers: known values, edge cases and the honesty rules."""
import math

from cs2_pipeline.stats import confidence_label, summarize_rate, wilson_ci


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


if __name__ == "__main__":
    from tests._runner import run

    raise SystemExit(run(globals()))
