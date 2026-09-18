from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    direction: str
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    tp: int
    fp: int
    tn: int
    fn: int


def _predict(value: float, threshold: float, direction: str) -> bool:
    if direction == "lower":
        return value <= threshold
    if direction == "higher":
        return value >= threshold
    raise ValueError(f"Unsupported direction: {direction}")


def evaluate_threshold(
    values: list[float],
    labels: list[bool],
    *,
    threshold: float,
    direction: str,
) -> ThresholdResult:
    if len(values) != len(labels):
        raise ValueError("values and labels must have equal length")
    tp = fp = tn = fn = 0
    for value, label in zip(values, labels):
        pred = _predict(float(value), float(threshold), direction)
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and not label:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return ThresholdResult(
        threshold=float(threshold),
        direction=direction,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def fit_threshold(
    values: Iterable[float],
    labels: Iterable[bool],
    *,
    direction: str,
) -> ThresholdResult:
    vals = [float(v) for v in values]
    labs = [bool(v) for v in labels]
    if len(vals) != len(labs) or not vals:
        raise ValueError("non-empty values and labels of equal length are required")
    if not any(labs) or all(labs):
        raise ValueError("both positive and negative examples are required")

    unique = sorted(set(vals))
    candidates: list[float] = []
    if len(unique) == 1:
        candidates = unique
    else:
        candidates.append(unique[0])
        for left, right in zip(unique, unique[1:]):
            candidates.append((left + right) / 2.0)
        candidates.append(unique[-1])

    results = [
        evaluate_threshold(vals, labs, threshold=t, direction=direction)
        for t in candidates
    ]
    # Primary: F1. Tie-breakers prefer precision, lower FPR, then recall.
    return max(
        results,
        key=lambda item: (
            item.f1,
            item.precision,
            -item.false_positive_rate,
            item.recall,
        ),
    )


def promotion_ready(
    result: ThresholdResult,
    *,
    positive_count: int,
    negative_count: int,
    intentional_negative_count: int = 0,
    min_positive: int = 5,
    min_negative: int = 5,
    min_intentional_negative: int = 2,
    min_precision: float = 0.80,
    min_recall: float = 0.75,
    min_f1: float = 0.80,
    max_fpr: float = 0.20,
) -> bool:
    return (
        positive_count >= min_positive
        and negative_count >= min_negative
        and intentional_negative_count >= min_intentional_negative
        and result.precision >= min_precision
        and result.recall >= min_recall
        and result.f1 >= min_f1
        and result.false_positive_rate <= max_fpr
    )
