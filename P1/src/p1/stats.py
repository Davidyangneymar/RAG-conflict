"""Statistical-significance utilities for NLI evaluation.

This module provides two complementary tests for comparing two models on the
same test set:

* `bootstrap_macro_f1_ci` — non-parametric 95% confidence interval on the
  macro-F1 of a single system, computed by resampling test indices with
  replacement.
* `paired_bootstrap_test` — the paired-resampling test for the macro-F1
  *delta* between two systems on the same items (Berg-Kirkpatrick et al.,
  2012). Reports the bootstrap CI of the delta and a one-sided p-value
  (probability that the better system is *not* actually better).

Both functions are pure-Python (stdlib `random` only) so they work in
the project venv without adding scipy/numpy as runtime dependencies.

Usage:
    from p1.stats import paired_bootstrap_test
    result = paired_bootstrap_test(
        gold_labels, baseline_preds, candidate_preds,
        iterations=10_000, seed=42,
    )
    print(result["delta_macro_f1"], result["ci_95"], result["p_value"])
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Sequence


_LABELS = ("entailment", "contradiction", "neutral")


# ---------------------------------------------------------------------------
# Macro-F1 from indexable predictions
# ---------------------------------------------------------------------------
def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Compute macro-averaged F1 over the canonical 3 NLI labels."""
    confusion: dict[str, Counter[str]] = {label: Counter() for label in _LABELS}
    for g, p in zip(gold, pred):
        if g not in confusion:
            confusion[g] = Counter()
        confusion[g][p] += 1

    f1_scores: list[float] = []
    for label in _LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in _LABELS if other != label)
        fn = sum(confusion[label][other] for other in _LABELS if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall:
            f1_scores.append(2 * precision * recall / (precision + recall))
        else:
            f1_scores.append(0.0)
    return sum(f1_scores) / len(f1_scores)


# ---------------------------------------------------------------------------
# Single-system bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_macro_f1_ci(
    gold: Sequence[str],
    pred: Sequence[str],
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> dict[str, float | int]:
    """Bootstrap CI for the macro-F1 of a single system.

    Returns a dict with point estimate, CI bounds, iterations, and seed.
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    if not gold:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "iterations": 0, "seed": seed}

    rng = random.Random(seed)
    n = len(gold)
    samples: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled_gold = [gold[i] for i in indices]
        sampled_pred = [pred[i] for i in indices]
        samples.append(macro_f1(sampled_gold, sampled_pred))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    low = samples[int(alpha * iterations)]
    high = samples[min(iterations - 1, int((1.0 - alpha) * iterations))]
    return {
        "point": macro_f1(gold, pred),
        "ci_low": low,
        "ci_high": high,
        "iterations": iterations,
        "confidence": confidence,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Paired bootstrap (per-item delta)
# ---------------------------------------------------------------------------
def paired_bootstrap_test(
    gold: Sequence[str],
    baseline_pred: Sequence[str],
    candidate_pred: Sequence[str],
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> dict[str, float | int]:
    """Paired bootstrap test on macro-F1 delta (candidate − baseline).

    Both systems are evaluated on the *same* resampled subset each
    iteration, so the variance of the delta is much smaller than the
    sum-of-variances. Returns the observed delta, bootstrap distribution
    quantiles, and a one-sided p-value defined as the proportion of
    bootstrap deltas ≤ 0 (i.e. the empirical probability that the
    candidate is *not* better than baseline on a resampled test set).
    """
    if not (len(gold) == len(baseline_pred) == len(candidate_pred)):
        raise ValueError("gold, baseline_pred, candidate_pred must align in length")
    if not gold:
        return {
            "delta": 0.0,
            "baseline_macro_f1": 0.0,
            "candidate_macro_f1": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "iterations": 0,
            "seed": seed,
        }

    rng = random.Random(seed)
    n = len(gold)
    deltas: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled_gold = [gold[i] for i in indices]
        sampled_baseline = [baseline_pred[i] for i in indices]
        sampled_candidate = [candidate_pred[i] for i in indices]
        delta = macro_f1(sampled_gold, sampled_candidate) - macro_f1(sampled_gold, sampled_baseline)
        deltas.append(delta)
    deltas.sort()
    alpha = (1.0 - confidence) / 2.0
    low = deltas[int(alpha * iterations)]
    high = deltas[min(iterations - 1, int((1.0 - alpha) * iterations))]
    no_lift = sum(1 for d in deltas if d <= 0.0)
    p_value = no_lift / iterations

    observed_baseline = macro_f1(gold, baseline_pred)
    observed_candidate = macro_f1(gold, candidate_pred)

    return {
        "baseline_macro_f1": observed_baseline,
        "candidate_macro_f1": observed_candidate,
        "delta": observed_candidate - observed_baseline,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "iterations": iterations,
        "confidence": confidence,
        "seed": seed,
    }
