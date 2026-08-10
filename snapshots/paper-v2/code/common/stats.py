"""Uncertainty quantification for the scheduling experiments.

Every cell in the policy comparison has n=3 repeats. That is enough to *report* uncertainty and
far too few to hide it, so this module is deliberately conservative:

  * Confidence intervals use the **t distribution**, not the normal. At n=3 the two-sided 95%
    multiplier is t(0.975, df=2) = 4.303 versus z = 1.96 — a CI computed with z would be 2.2x
    too narrow. This is the single most common way small-n benchmark tables overstate certainty.
  * Effect sizes are **Hedges' g**, i.e. Cohen's d with the small-sample bias correction
    J = 1 - 3/(4(n1+n2) - 9). At n1=n2=3 that is J = 0.8, so the uncorrected d overstates the
    effect by 25%.
  * Comparisons use **Welch's t-test** (unequal variances). Throughput variance differs sharply
    between policies — an OOM-adjacent policy is far noisier than a capped one — so the
    equal-variance assumption of Student's t is not tenable here.
  * Proportions (SLO attainment) additionally get a **Wilson score interval** on pooled
    request counts, because a t-interval on three proportions can run outside [0, 1] and has
    poor coverage near the boundary, which is exactly where an SLO of 0.99 sits.

Nothing here corrects for multiple comparisons on its own; call ``holm`` on a family of p-values
when reporting more than one contrast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats as _st


@dataclass
class Descriptive:
    n: int
    mean: float
    sd: float
    sem: float
    ci95_lo: float
    ci95_hi: float
    ci95_halfwidth: float
    cv_pct: float          # coefficient of variation — relative noise, comparable across metrics
    t_multiplier: float    # recorded so a reader can see the small-n penalty explicitly
    min: float
    max: float

    def as_dict(self) -> dict:
        return asdict(self)

    def fmt(self, prec: int = 2) -> str:
        return f"{self.mean:.{prec}f} ± {self.ci95_halfwidth:.{prec}f}"


def describe(values, ddof: int = 1) -> Descriptive:
    """Mean, SD and a t-based 95% CI. Non-finite values are dropped before computing."""
    a = np.asarray([v for v in np.asarray(values, dtype=float).ravel() if np.isfinite(v)])
    n = a.size
    if n == 0:
        nan = float("nan")
        return Descriptive(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    mean = float(a.mean())
    if n == 1:
        return Descriptive(1, mean, 0.0, 0.0, mean, mean, 0.0, 0.0, float("nan"), mean, mean)

    sd = float(a.std(ddof=ddof))
    sem = sd / math.sqrt(n)
    tmul = float(_st.t.ppf(0.975, n - ddof))
    half = tmul * sem
    return Descriptive(
        n=n,
        mean=mean,
        sd=sd,
        sem=sem,
        ci95_lo=mean - half,
        ci95_hi=mean + half,
        ci95_halfwidth=half,
        cv_pct=(sd / mean * 100.0) if mean else float("nan"),
        t_multiplier=tmul,
        min=float(a.min()),
        max=float(a.max()),
    )


@dataclass
class Contrast:
    """A treatment-vs-baseline comparison on one metric."""

    metric: str
    baseline: str
    treatment: str
    n_base: int
    n_treat: int
    mean_base: float
    mean_treat: float
    diff: float                 # treatment - baseline
    pct_change: float
    diff_ci95_lo: float
    diff_ci95_hi: float
    t_stat: float
    df: float
    p_value: float
    hedges_g: float
    g_ci95_lo: float
    g_ci95_hi: float
    magnitude: str              # negligible / small / medium / large
    significant_05: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _magnitude(g: float) -> str:
    a = abs(g)
    if not np.isfinite(a):
        return "undefined"
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def hedges_g(treat, base) -> tuple[float, float, float]:
    """Hedges' g with an approximate 95% CI. Returns (g, lo, hi).

    g = J * (m_t - m_b) / s_pooled, with J the small-sample bias correction. The CI uses the
    usual large-sample variance approximation; at n=3 it is indicative rather than exact, which
    is why it is reported alongside the raw difference CI rather than instead of it.
    """
    t = np.asarray([v for v in np.asarray(treat, float).ravel() if np.isfinite(v)])
    b = np.asarray([v for v in np.asarray(base, float).ravel() if np.isfinite(v)])
    n1, n2 = t.size, b.size
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan"), float("nan")

    s_pooled = math.sqrt(
        ((n1 - 1) * t.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    )
    if s_pooled == 0:
        return float("nan"), float("nan"), float("nan")

    d = (t.mean() - b.mean()) / s_pooled
    J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    g = J * d
    se_g = math.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2.0 * (n1 + n2 - 2)))
    return float(g), float(g - 1.96 * se_g), float(g + 1.96 * se_g)


def contrast(metric: str, treatment_name: str, treat, baseline_name: str, base) -> Contrast:
    """Welch's t-test plus Hedges' g for one treatment-vs-baseline pair."""
    t = np.asarray([v for v in np.asarray(treat, float).ravel() if np.isfinite(v)])
    b = np.asarray([v for v in np.asarray(base, float).ravel() if np.isfinite(v)])

    if t.size < 2 or b.size < 2:
        nan = float("nan")
        return Contrast(metric, baseline_name, treatment_name, b.size, t.size,
                        float(b.mean()) if b.size else nan, float(t.mean()) if t.size else nan,
                        nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, "undefined", False)

    res = _st.ttest_ind(t, b, equal_var=False)
    tstat, p = float(res.statistic), float(res.pvalue)
    df = float(getattr(res, "df", np.nan))

    diff = float(t.mean() - b.mean())
    se_diff = math.sqrt(t.var(ddof=1) / t.size + b.var(ddof=1) / b.size)
    tcrit = float(_st.t.ppf(0.975, df)) if np.isfinite(df) and df > 0 else float("nan")
    half = tcrit * se_diff if np.isfinite(tcrit) else float("nan")

    g, glo, ghi = hedges_g(t, b)
    return Contrast(
        metric=metric,
        baseline=baseline_name,
        treatment=treatment_name,
        n_base=int(b.size),
        n_treat=int(t.size),
        mean_base=float(b.mean()),
        mean_treat=float(t.mean()),
        diff=diff,
        pct_change=(diff / b.mean() * 100.0) if b.mean() else float("nan"),
        diff_ci95_lo=diff - half,
        diff_ci95_hi=diff + half,
        t_stat=tstat,
        df=df,
        p_value=p,
        hedges_g=g,
        g_ci95_lo=glo,
        g_ci95_hi=ghi,
        magnitude=_magnitude(g),
        significant_05=bool(np.isfinite(p) and p < 0.05),
    )


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, preserving input order.

    Preferred over plain Bonferroni: uniformly more powerful, same familywise error control,
    and no independence assumption — appropriate for correlated latency/throughput metrics
    measured on the same runs.
    """
    idx = [i for i, p in enumerate(p_values) if np.isfinite(p)]
    if not idx:
        return list(p_values)
    m = len(idx)
    order = sorted(idx, key=lambda i: p_values[i])
    out = list(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * p_values[i]
        running = max(running, adj)          # enforce monotonicity
        out[i] = float(min(1.0, running))
    return out


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion. Returns (p_hat, lo, hi).

    Used for SLO attainment, where the point estimate sits near 1.0 and a normal-approximation
    interval would extend above 1 and under-cover.
    """
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return float(p), float(max(0.0, centre - margin)), float(min(1.0, centre + margin))
