"""Output-length prediction and the SRPT misprediction mini-experiment.

SRPT wants to order by *service time*, which for LLM decode is dominated by OUTPUT length — unknown
at admission. This module provides three size sources for SRPT and quantifies the cost of getting
them wrong:

  * oracle      — the request's ACTUAL output length (perfect knowledge; upper bound). Obtained
                  from a deterministic profiling run keyed by arrival seq (greedy decode => the same
                  prompt yields the same length regardless of batching).
  * prediction  — a light class-aware linear regressor on (prompt_len, class).
  * prompt-proxy — prompt length (the cheap default in scheduler.prompt_proxy_size).

Caveat worth stating in the writeup: in this synthetic workload output length is drawn per-class
independently of prompt length, so the prompt proxy is intentionally weak — the experiment measures
how much SRPT degrades as the size estimate degrades, which is the point.
"""
from __future__ import annotations

import numpy as np

from common.workload import RequestSpec


class LengthPredictor:
    """Class-aware linear regressor: output_tokens ~ bias + a*prompt_tokens + per-class effect."""

    def __init__(self):
        self.classes: list[str] = []
        self.coef: np.ndarray | None = None

    def _features(self, prompt_tokens: float, profile: str) -> list[float]:
        onehot = [1.0 if profile == c else 0.0 for c in self.classes]
        return [1.0, float(prompt_tokens)] + onehot

    def fit(self, records) -> "LengthPredictor":
        """records: objects with .prompt_tokens, .profile, .output_tokens_actual, .success."""
        rows = [r for r in records if r.success and r.output_tokens_actual > 0]
        self.classes = sorted({r.profile for r in rows})
        X = np.array([self._features(r.prompt_tokens, r.profile) for r in rows], dtype=float)
        y = np.array([r.output_tokens_actual for r in rows], dtype=float)
        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, prompt_tokens: float, profile: str) -> float:
        if self.coef is None:
            return 128.0
        val = float(np.array(self._features(prompt_tokens, profile)) @ self.coef)
        return max(1.0, val)

    def predict_spec(self, spec: RequestSpec) -> float:
        return self.predict(spec.prompt_words, spec.profile)

    def mae(self, records) -> float:
        rows = [r for r in records if r.success and r.output_tokens_actual > 0]
        if not rows:
            return float("nan")
        errs = [abs(self.predict(r.prompt_tokens, r.profile) - r.output_tokens_actual) for r in rows]
        return float(np.mean(errs))


# --- size_fn factories for SRPT variants ------------------------------------------------


def oracle_size_fn(actuals: dict[int, float]):
    """Size = actual output length, looked up by arrival seq (from a profiling run)."""
    return lambda spec: float(actuals.get(spec.seq, 128.0))


def prediction_size_fn(predictor: LengthPredictor):
    return lambda spec: predictor.predict_spec(spec)


def actuals_by_seq(results) -> dict[int, float]:
    """Map arrival seq -> actual output tokens from a profiling run's RequestResults.

    Arrivals are deterministic under a fixed seed and greedy decode is length-deterministic, so this
    mapping is stable across runs and usable as oracle knowledge in a later SRPT run.
    """
    return {
        r.seq: float(r.output_tokens_actual)
        for r in results
        if r.success and r.output_tokens_actual > 0
    }
