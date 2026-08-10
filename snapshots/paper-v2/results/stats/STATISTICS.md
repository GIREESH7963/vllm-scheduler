# Statistical analysis — Phase-2 policy comparison

All intervals are **two-sided 95% t-intervals** on n=3 repeats per cell (multiplier t(0.975, df=2) = **4.303**, not 1.96). Effect sizes are **Hedges' g** (Cohen's d with the small-sample correction J = 0.8 at n=3). Contrasts use **Welch's t-test**; p-values are additionally reported Holm-adjusted across the whole family of 171 comparisons.

> **Read the intervals before the means.** With n=3 the CI half-width is 2.48 SD/√n. Several cells below have intervals wide enough that the point estimate alone would be misleading; those are flagged rather than quietly reported.


## Workload `phase2_chat`


### λ = 4 req/s

| Policy | n | Throughput (tok/s) | Total lat p50 (ms) | Total lat p99 (ms) | Queue wait p99 (ms) | SLO (repeat mean) | SLO (pooled Wilson) |
|---|---|---|---|---|---|---|---|
| `adaptive` | 3 | 246.3 ± 28.1 | 1142.3 ± 352.2 | 14577.2 ± 12782.7 | 3.5 ± 1.6 | 1.000 ± 0.000 | 1.000 [0.982, 1.000] |
| `edf` | 3 | 247.0 ± 26.5 | 1252.9 ± 523.2 | 15446.8 ± 5047.8 | 937.3 ± 2031.9 | 0.971 ± 0.068 | 0.971 [0.938, 0.987] |
| `fcfs` *(baseline)* | 3 | 247.2 ± 26.0 | 1217.4 ± 291.2 | 13640.9 ± 7688.6 | 855.0 ± 1930.6 | 0.962 ± 0.067 | 0.961 [0.925, 0.980] |
| `nocap` | 3 | 245.1 ± 30.3 | 1192.5 ± 234.4 | 12919.0 ± 7862.0 | 3.5 ± 1.6 | 1.000 ± 0.000 | 1.000 [0.982, 1.000] |
| `srpt` | 3 | 246.4 ± 27.8 | 1169.2 ± 384.4 | 12785.3 ± 8748.5 | 58.8 ± 243.1 | 0.995 ± 0.020 | 0.995 [0.973, 0.999] |

**High-variance cells (CV > 30%)** — treat these means as indicative only:

- `adaptive` / total_latency_ms_p99: CV = 35%

### λ = 6 req/s

| Policy | n | Throughput (tok/s) | Total lat p50 (ms) | Total lat p99 (ms) | Queue wait p99 (ms) | SLO (repeat mean) | SLO (pooled Wilson) |
|---|---|---|---|---|---|---|---|
| `adaptive` | 3 | 409.8 ± 54.7 | 1888.0 ± 943.3 | 17855.6 ± 8167.8 | 1567.4 ± 1250.2 | 0.898 ± 0.104 | 0.898 [0.860, 0.926] |
| `edf` | 3 | 367.3 ± 77.8 | 2011.4 ± 501.6 | 18803.2 ± 9374.0 | 4921.8 ± 2059.8 | 0.737 ± 0.208 | 0.739 [0.688, 0.785] |
| `fcfs` *(baseline)* | 3 | 398.5 ± 23.8 | 2537.7 ± 1996.0 | 18474.8 ± 1598.6 | 4668.7 ± 3824.1 | 0.467 ± 0.939 | 0.470 [0.415, 0.525] |
| `nocap` | 3 | 408.9 ± 54.1 | 1794.2 ± 451.2 | 19349.7 ± 3516.1 | 5.0 ± 1.4 | 0.883 ± 0.136 | 0.882 [0.843, 0.913] |
| `srpt` | 3 | 366.0 ± 64.1 | 1760.3 ± 310.9 | 14624.2 ± 7044.9 | 4740.3 ± 409.2 | 0.792 ± 0.068 | 0.791 [0.742, 0.832] |

## Workload `phase2_mixed`


### λ = 2 req/s

| Policy | n | Throughput (tok/s) | Total lat p50 (ms) | Total lat p99 (ms) | Queue wait p99 (ms) | SLO (repeat mean) | SLO (pooled Wilson) |
|---|---|---|---|---|---|---|---|
| `adaptive` | 3 | 211.4 ± 79.6 | 2174.0 ± 2397.0 | 15385.7 ± 8051.6 | 2.8 ± 0.7 | 0.990 ± 0.042 | 0.990 [0.948, 0.998] |
| `edf` | 3 | 211.6 ± 79.0 | 2143.3 ± 2358.1 | 16426.3 ± 18460.5 | 1762.8 ± 2757.2 | 0.893 ± 0.256 | 0.895 [0.822, 0.940] |
| `fcfs` *(baseline)* | 3 | 211.5 ± 78.5 | 2334.7 ± 1589.1 | 15959.3 ± 18795.8 | 1582.3 ± 2383.7 | 0.884 ± 0.250 | 0.886 [0.811, 0.933] |
| `nocap` | 3 | 212.4 ± 80.4 | 1865.3 ± 2380.1 | 14444.2 ± 21666.3 | 14.1 ± 45.7 | 0.990 ± 0.042 | 0.990 [0.948, 0.998] |
| `srpt` | 3 | 211.5 ± 78.7 | 2176.8 ± 1081.7 | 15398.5 ± 18922.3 | 1879.1 ± 3957.2 | 0.923 ± 0.216 | 0.924 [0.857, 0.961] |

**High-variance cells (CV > 30%)** — treat these means as indicative only:

- `edf` / total_latency_ms_p99: CV = 45%
- `fcfs` / total_latency_ms_p99: CV = 47%
- `nocap` / total_latency_ms_p99: CV = 60%
- `srpt` / total_latency_ms_p99: CV = 49%

### λ = 4 req/s

| Policy | n | Throughput (tok/s) | Total lat p50 (ms) | Total lat p99 (ms) | Queue wait p99 (ms) | SLO (repeat mean) | SLO (pooled Wilson) |
|---|---|---|---|---|---|---|---|
| `adaptive` | 3 | 430.1 ± 100.0 | 4551.6 ± 621.3 | 21118.9 ± 11526.6 | 4821.3 ± 4537.7 | 0.569 ± 0.489 | 0.558 [0.490, 0.624] |
| `edf` | 3 | 274.1 ± 47.2 | 2561.4 ± 660.7 | 15677.6 ± 8089.2 | 5208.0 ± 376.4 | 0.511 ± 0.201 | 0.514 [0.441, 0.586] |
| `fcfs` *(baseline)* | 3 | 317.4 ± 16.4 | 6263.5 ± 2629.6 | 23633.8 ± 937.5 | 5940.6 ± 121.7 | 0.122 ± 0.324 | 0.128 [0.085, 0.190] |
| `nocap` | 3 | 469.7 ± 185.5 | 3772.9 ± 3629.2 | 20552.7 ± 8310.3 | 4.5 ± 4.7 | 0.908 ± 0.241 | 0.895 [0.849, 0.929] |
| `srpt` | 3 | 323.2 ± 55.1 | 3193.3 ± 2170.0 | 19447.4 ± 1366.4 | 5246.4 ± 1262.0 | 0.429 ± 0.058 | 0.428 [0.356, 0.502] |
| `srpt-oracle` | 1 | 297.5 (n=1) | 2706.5 (n=1) | 17953.3 (n=1) | 5569.2 (n=1) | 0.376 (n=1) | 0.381 [0.271, 0.504] |
| `srpt-prediction` | 1 | 266.3 (n=1) | 2818.9 (n=1) | 17251.5 (n=1) | 5261.3 (n=1) | 0.441 (n=1) | 0.448 [0.335, 0.566] |
| `srpt-prompt-proxy` | 1 | 329.0 (n=1) | 2483.6 (n=1) | 18774.6 (n=1) | 5436.0 (n=1) | 0.462 (n=1) | 0.456 [0.343, 0.573] |

## Per-class SLO attainment

Mean ± 95% CI over 3 repeats, by workload class.


**`phase2_chat`, λ = 4**

| Policy | reasoning | short |
|---|---|---|
| `adaptive` | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `edf` | 1.000 ± 0.000 | 0.966 ± 0.077 |
| `fcfs` | 1.000 ± 0.000 | 0.956 ± 0.075 |
| `nocap` | 1.000 ± 0.000 | 1.000 ± 0.000 |
| `srpt` | 1.000 ± 0.000 | 0.995 ± 0.023 |

**`phase2_chat`, λ = 6**

| Policy | reasoning | short |
|---|---|---|
| `adaptive` | 0.899 ± 0.076 | 0.898 ± 0.125 |
| `edf` | 0.488 ± 0.245 | 0.782 ± 0.287 |
| `fcfs` | 0.469 ± 0.888 | 0.466 ± 0.948 |
| `nocap` | 1.000 ± 0.000 | 0.862 ± 0.155 |
| `srpt` | 0.525 ± 0.275 | 0.839 ± 0.069 |

**`phase2_mixed`, λ = 2**

| Policy | coding | long | reasoning | short |
|---|---|---|---|---|
| `adaptive` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.980 ± 0.084 |
| `edf` | 0.667 ± 0.828 | 1.000 ± 0.000 | 0.719 ± 0.608 | 0.961 ± 0.169 |
| `fcfs` | 0.667 ± 0.828 | 0.933 ± 0.287 | 0.822 ± 0.417 | 0.925 ± 0.210 |
| `nocap` | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.980 ± 0.084 |
| `srpt` | 0.778 ± 0.956 | 0.933 ± 0.287 | 0.830 ± 0.513 | 0.963 ± 0.159 |

**`phase2_mixed`, λ = 4**

| Policy | coding | long | reasoning | short |
|---|---|---|---|---|
| `adaptive` | 0.418 ± 0.742 | 0.590 ± 0.425 | 0.652 ± 0.584 | 0.569 ± 0.425 |
| `edf` | 0.221 ± 0.575 | 0.419 ± 0.431 | 0.044 ± 0.191 | 0.851 ± 0.321 |
| `fcfs` | 0.033 ± 0.143 | 0.127 ± 0.298 | 0.140 ± 0.355 | 0.134 ± 0.437 |
| `nocap` | 0.890 ± 0.098 | 0.968 ± 0.137 | 0.978 ± 0.096 | 0.869 ± 0.457 |
| `srpt` | 0.313 ± 0.343 | 0.063 ± 0.273 | 0.360 ± 0.208 | 0.676 ± 0.160 |
| `srpt-oracle` | 0.231 (n=1) | 0.238 (n=1) | 0.133 (n=1) | 0.568 (n=1) |
| `srpt-prediction` | 0.000 (n=1) | 0.286 (n=1) | 0.133 (n=1) | 0.750 (n=1) |
| `srpt-prompt-proxy` | 0.154 (n=1) | 0.190 (n=1) | 0.267 (n=1) | 0.750 (n=1) |

## Contrasts vs `fcfs`

`Δ` is treatment − baseline with its own 95% CI. `g` is Hedges' g (|g|<0.2 negligible, <0.5 small, <0.8 medium, ≥0.8 large). `p` is Welch; `p_holm` is Holm-adjusted across all comparisons in this document.


**`phase2_chat`, λ = 4**

| Policy | Metric | baseline | treatment | Δ [95% CI] | g | magnitude | p | p_holm |
|---|---|---|---|---|---|---|---|---|
| `adaptive` | SLO attainment | 0.962 | 1 | +0.0385 [-0.0289, +0.106] | +1.60 | large | 0.133 | 1.000 |
| `adaptive` | Throughput | 247 | 246 | -0.863 [-25.7, +23.9] | -0.06 | negligible | 0.928 | 1.000 |
| `adaptive` | Total latency p50 | 1.22e+03 | 1.14e+03 | -75.1 [-374, +224] | -0.46 | small | 0.520 | 1.000 |
| `adaptive` | Total latency p99 | 1.36e+04 | 1.46e+04 | +936 [-9.58e+03, +1.15e+04] | +0.18 | negligible | 0.803 | 1.000 |
| `edf` | SLO attainment | 0.962 | 0.971 | +0.00913 [-0.0526, +0.0708] | +0.27 | small | 0.702 | 1.000 |
| `edf` | Throughput | 247 | 247 | -0.176 [-24.1, +23.8] | -0.01 | negligible | 0.985 | 1.000 |
| `edf` | Total latency p50 | 1.22e+03 | 1.25e+03 | +35.5 [-397, +468] | +0.17 | negligible | 0.815 | 1.000 |
| `edf` | Total latency p99 | 1.36e+04 | 1.54e+04 | +1.81e+03 [-4.52e+03, +8.13e+03] | +0.55 | medium | 0.453 | 1.000 |
| `nocap` | SLO attainment | 0.962 | 1 | +0.0385 [-0.0289, +0.106] | +1.60 | large | 0.133 | 1.000 |
| `nocap` | Throughput | 247 | 245 | -2.14 [-28.2, +23.9] | -0.15 | negligible | 0.830 | 1.000 |
| `nocap` | Total latency p50 | 1.22e+03 | 1.19e+03 | -24.9 [-271, +221] | -0.19 | negligible | 0.789 | 1.000 |
| `nocap` | Total latency p99 | 1.36e+04 | 1.29e+04 | -722 [-7.82e+03, +6.38e+03] | -0.18 | negligible | 0.792 | 1.000 |
| `srpt` | SLO attainment | 0.962 | 0.995 | +0.0338 [-0.0274, +0.095] | +1.35 | large | 0.155 | 1.000 |
| `srpt` | Throughput | 247 | 246 | -0.843 [-25.5, +23.8] | -0.06 | negligible | 0.929 | 1.000 |
| `srpt` | Total latency p50 | 1.22e+03 | 1.17e+03 | -48.2 [-369, +272] | -0.28 | small | 0.691 | 1.000 |
| `srpt` | Total latency p99 | 1.36e+04 | 1.28e+04 | -856 [-8.42e+03, +6.71e+03] | -0.21 | small | 0.768 | 1.000 |

**`phase2_chat`, λ = 6**

| Policy | Metric | baseline | treatment | Δ [95% CI] | g | magnitude | p | p_holm |
|---|---|---|---|---|---|---|---|---|
| `adaptive` | SLO attainment | 0.467 | 0.898 | +0.431 [-0.492, +1.35] | +1.28 | large | 0.185 | 1.000 |
| `adaptive` | Throughput | 398 | 410 | +11.3 [-35.4, +58] | +0.53 | medium | 0.480 | 1.000 |
| `adaptive` | Total latency p50 | 2.54e+03 | 1.89e+03 | -650 [-2.33e+03, +1.03e+03] | -0.83 | large | 0.299 | 1.000 |
| `adaptive` | Total latency p99 | 1.85e+04 | 1.79e+04 | -619 [-8.4e+03, +7.16e+03] | -0.21 | small | 0.777 | 1.000 |
| `edf` | SLO attainment | 0.467 | 0.737 | +0.27 [-0.613, +1.15] | +0.79 | medium | 0.340 | 1.000 |
| `edf` | Throughput | 398 | 367 | -31.1 [-101, +39.2] | -1.08 | large | 0.222 | 1.000 |
| `edf` | Total latency p50 | 2.54e+03 | 2.01e+03 | -526 [-2.38e+03, +1.33e+03] | -0.72 | medium | 0.375 | 1.000 |
| `edf` | Total latency p99 | 1.85e+04 | 1.88e+04 | +328 [-8.7e+03, +9.35e+03] | +0.10 | negligible | 0.895 | 1.000 |
| `nocap` | SLO attainment | 0.467 | 0.883 | +0.416 [-0.497, +1.33] | +1.23 | large | 0.195 | 1.000 |
| `nocap` | Throughput | 398 | 409 | +10.5 [-35.6, +56.5] | +0.50 | small | 0.506 | 1.000 |
| `nocap` | Total latency p50 | 2.54e+03 | 1.79e+03 | -744 [-2.62e+03, +1.13e+03] | -1.02 | large | 0.247 | 1.000 |
| `nocap` | Total latency p99 | 1.85e+04 | 1.93e+04 | +875 [-2.11e+03, +3.86e+03] | +0.64 | medium | 0.406 | 1.000 |
| `srpt` | SLO attainment | 0.467 | 0.792 | +0.325 [-0.606, +1.26] | +0.97 | large | 0.274 | 1.000 |
| `srpt` | Throughput | 398 | 366 | -32.5 [-88.6, +23.7] | -1.33 | large | 0.150 | 1.000 |
| `srpt` | Total latency p50 | 2.54e+03 | 1.76e+03 | -777 [-2.71e+03, +1.16e+03] | -1.08 | large | 0.234 | 1.000 |
| `srpt` | Total latency p99 | 1.85e+04 | 1.46e+04 | -3.85e+03 [-1.05e+04, +2.77e+03] | -1.50 | large | 0.137 | 1.000 |

**`phase2_mixed`, λ = 2**

| Policy | Metric | baseline | treatment | Δ [95% CI] | g | magnitude | p | p_holm |
|---|---|---|---|---|---|---|---|---|
| `adaptive` | SLO attainment | 0.884 | 0.99 | +0.106 [-0.134, +0.347] | +1.18 | large | 0.206 | 1.000 |
| `adaptive` | Throughput | 211 | 211 | -0.0588 [-72.2, +72.1] | -0.00 | negligible | 0.998 | 1.000 |
| `adaptive` | Total latency p50 | 2.33e+03 | 2.17e+03 | -161 [-2.13e+03, +1.81e+03] | -0.16 | negligible | 0.823 | 1.000 |
| `adaptive` | Total latency p99 | 1.6e+04 | 1.54e+04 | -574 [-1.67e+04, +1.55e+04] | -0.08 | negligible | 0.912 | 1.000 |
| `edf` | SLO attainment | 0.884 | 0.893 | +0.00924 [-0.222, +0.24] | +0.07 | negligible | 0.917 | 1.000 |
| `edf` | Throughput | 211 | 212 | +0.157 [-71.7, +72] | +0.00 | negligible | 0.995 | 1.000 |
| `edf` | Total latency p50 | 2.33e+03 | 2.14e+03 | -191 [-2.13e+03, +1.75e+03] | -0.19 | negligible | 0.788 | 1.000 |
| `edf` | Total latency p99 | 1.6e+04 | 1.64e+04 | +467 [-1.65e+04, +1.75e+04] | +0.05 | negligible | 0.943 | 1.000 |
| `nocap` | SLO attainment | 0.884 | 0.99 | +0.106 [-0.134, +0.347] | +1.18 | large | 0.206 | 1.000 |
| `nocap` | Throughput | 211 | 212 | +0.902 [-71.6, +73.4] | +0.02 | negligible | 0.974 | 1.000 |
| `nocap` | Total latency p50 | 2.33e+03 | 1.87e+03 | -469 [-2.43e+03, +1.49e+03] | -0.46 | small | 0.525 | 1.000 |
| `nocap` | Total latency p99 | 1.6e+04 | 1.44e+04 | -1.52e+03 [-2.02e+04, +1.71e+04] | -0.15 | negligible | 0.832 | 1.000 |
| `srpt` | SLO attainment | 0.884 | 0.923 | +0.0392 [-0.176, +0.254] | +0.33 | small | 0.637 | 1.000 |
| `srpt` | Throughput | 211 | 212 | +0.0588 [-71.7, +71.8] | +0.00 | negligible | 0.998 | 1.000 |
| `srpt` | Total latency p50 | 2.33e+03 | 2.18e+03 | -158 [-1.47e+03, +1.15e+03] | -0.23 | small | 0.744 | 1.000 |
| `srpt` | Total latency p99 | 1.6e+04 | 1.54e+04 | -561 [-1.78e+04, +1.66e+04] | -0.06 | negligible | 0.932 | 1.000 |

**`phase2_mixed`, λ = 4**

| Policy | Metric | baseline | treatment | Δ [95% CI] | g | magnitude | p | p_holm |
|---|---|---|---|---|---|---|---|---|
| `adaptive` | SLO attainment | 0.122 | 0.569 | +0.447 [+0.0441, +0.849] | +2.14 | large | 0.038 | 1.000 |
| `adaptive` | Throughput | 317 | 430 | +113 [+16.1, +209] | +3.13 | large | 0.037 | 1.000 |
| `adaptive` | Total latency p50 | 6.26e+03 | 4.55e+03 | -1.71e+03 [-4.17e+03, +747] | -1.78 | large | 0.100 | 1.000 |
| `adaptive` | Total latency p99 | 2.36e+04 | 2.11e+04 | -2.51e+03 [-1.39e+04, +8.91e+03] | -0.61 | medium | 0.447 | 1.000 |
| `edf` | SLO attainment | 0.122 | 0.511 | +0.389 [+0.122, +0.655] | +2.87 | large | 0.018 | 1.000 |
| `edf` | Throughput | 317 | 274 | -43.4 [-85.2, -1.55] | -2.44 | large | 0.046 | 1.000 |
| `edf` | Total latency p50 | 6.26e+03 | 2.56e+03 | -3.7e+03 [-6.14e+03, -1.26e+03] | -3.84 | large | 0.021 | 1.000 |
| `edf` | Total latency p99 | 2.36e+04 | 1.57e+04 | -7.96e+03 [-1.59e+04, -13.7] | -2.75 | large | 0.050 | 1.000 |
| `nocap` | SLO attainment | 0.122 | 0.908 | +0.785 [+0.516, +1.05] | +5.46 | large | 0.002 | 0.223 |
| `nocap` | Throughput | 317 | 470 | +152 [-31.2, +336] | +2.30 | large | 0.071 | 1.000 |
| `nocap` | Total latency p50 | 6.26e+03 | 3.77e+03 | -2.49e+03 [-5.5e+03, +515] | -1.56 | large | 0.081 | 1.000 |
| `nocap` | Total latency p99 | 2.36e+04 | 2.06e+04 | -3.08e+03 [-1.12e+04, +5.09e+03] | -1.04 | large | 0.251 | 1.000 |
| `srpt` | SLO attainment | 0.122 | 0.429 | +0.307 [-0.00449, +0.618] | +2.62 | large | 0.051 | 1.000 |
| `srpt` | Throughput | 317 | 323 | +5.75 [-44.3, +55.8] | +0.28 | small | 0.704 | 1.000 |
| `srpt` | Total latency p50 | 6.26e+03 | 3.19e+03 | -3.07e+03 [-5.3e+03, -839] | -2.53 | large | 0.019 | 1.000 |
| `srpt` | Total latency p99 | 2.36e+04 | 1.94e+04 | -4.19e+03 [-5.31e+03, -3.06e+03] | -7.10 | large | 0.001 | 0.109 |
| `srpt-oracle` | SLO attainment | 0.122 | 0.376 | +nan [+nan, +nan] | +nan | undefined | nan | nan |
| `srpt-oracle` | Throughput | 317 | 298 | +nan [+nan, +nan] | +nan | undefined | nan | nan |
| `srpt-oracle` | Total latency p50 | 6.26e+03 | 2.71e+03 | +nan [+nan, +nan] | +nan | undefined | nan | nan |
| `srpt-oracle` | Total latency p99 | 2.36e+04 | 1.8e+04 | +nan [+nan, +nan] | +nan | undefined | nan | nan |
| `srpt-prediction` | SLO attainment | 0.122 | 0.441 | +nan [+nan, +nan] | +nan | undefined | nan | nan |
| `srpt-prediction` | Throughput | 317 | 266 | +nan [+nan, +nan] | +nan | undefined | nan | nan |
| `srpt-prediction` | Total latency p50 | 6.26e+03 | 2.82e+03 | +nan [+nan, +nan] | +nan | undefined | nan | nan |
| `srpt-prediction` | Total latency p99 | 2.36e+04 | 1.73e+04 | +nan [+nan, +nan] | +nan | undefined | nan | nan |
| `srpt-prompt-proxy` | SLO attainment | 0.122 | 0.462 | +nan [+nan, +nan] | +nan | undefined | nan | nan |
| `srpt-prompt-proxy` | Throughput | 317 | 329 | +nan [+nan, +nan] | +nan | undefined | nan | nan |
| `srpt-prompt-proxy` | Total latency p50 | 6.26e+03 | 2.48e+03 | +nan [+nan, +nan] | +nan | undefined | nan | nan |
| `srpt-prompt-proxy` | Total latency p99 | 2.36e+04 | 1.88e+04 | +nan [+nan, +nan] | +nan | undefined | nan | nan |

## What survives

- Comparisons made: **171**
- Significant at p < 0.05 **before** correction: **16**
- Significant **after** Holm correction: **1**

Surviving contrasts:

- `nocap` vs `fcfs` — Queue wait p99 (phase2_mixed, λ=4): -5.94e+03 ms, g = -136.97, p_holm = 0.0032

## Caveats

1. **n=3 per cell.** Every interval is wide and every effect size is imprecise. Raising repeats to 10 would shrink the t-multiplier from 4.303 to 2.262 and the half-width by roughly 3x.
2. **Repeats are not independent of drift.** All repeats of a cell ran back to back on one passively cooled T4, so a thermal excursion is shared within a cell rather than averaged out. Check `thermal_throttled` before trusting a narrow interval.
3. **SLO is a proportion.** The repeat-mean column and the pooled Wilson column answer different questions; where they disagree, the pooled interval is the better guide to per-request behaviour and the repeat interval to run-to-run stability.
4. **Holm assumes the family is the one reported here.** Adding metrics later without re-running the correction reintroduces the multiplicity it removes.
