from app.vision.threshold_fitting import (
    evaluate_threshold,
    fit_threshold,
    promotion_ready,
)


def test_fit_threshold_finds_clean_separation_for_higher_signal() -> None:
    values = [0.8, 1.0, 1.1, 1.2, 3.5, 3.8, 4.1, 4.4]
    labels = [False, False, False, False, True, True, True, True]
    result = fit_threshold(values, labels, direction="higher")
    assert result.f1 == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_fit_threshold_finds_clean_separation_for_lower_signal() -> None:
    values = [120, 110, 100, 90, 25, 20, 15, 10]
    labels = [False, False, False, False, True, True, True, True]
    result = fit_threshold(values, labels, direction="lower")
    assert result.f1 == 1.0


def test_promotion_requires_enough_real_and_intentional_negatives() -> None:
    result = evaluate_threshold(
        [1, 2, 3, 8, 9, 10, 11, 12, 13, 14],
        [False, False, False, True, True, True, True, True, True, True],
        threshold=5,
        direction="higher",
    )
    assert not promotion_ready(
        result,
        positive_count=7,
        negative_count=3,
        intentional_negative_count=2,
    )
    assert promotion_ready(
        result,
        positive_count=7,
        negative_count=5,
        intentional_negative_count=2,
    )
