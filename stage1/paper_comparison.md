# Why my numbers do not match Figs. 8 and 9 of arXiv:1203.6228

*2026-08-02*

> **Reference note, added after the fact.** This whole document was written against
> [arXiv:1108.4258](https://arxiv.org/abs/1108.4258), the 4-page letter, before I found
> [arXiv:1203.6228](https://arxiv.org/abs/1203.6228), the full paper by the same authors.
> The letter's Fig. 2 left/right are Figs. 8 and 9 of the full paper, its Fig. 1 is Fig. 7,
> and its Eq (10) is Eq (6.1). Section and figure references below have been updated;
> where a number is quoted that I only ever saw in the letter, the letter is cited.
> Two things in the full paper bear directly on §1 and §5 and are **not yet incorporated**:
> footnote 11 states the correlation normalisation outright, and §6 fixes the window at
> $T = N$ and reports four indices including the **S&P 500**, which is the like-for-like
> comparison this document should have been making.

Two separate discrepancies were tangled together. One was a bug in my pipeline and is
now fixed. The other is not a bug and will not go away.

## 0. What the paper actually claims

Figs. 8 and 9 (Fig. 2 left and right of the letter) are **two different sweeps** that happen to land on the same number, and it is
easy to compare against the wrong one.

| | x-axis | y-axis | what moves | the claim |
|---|---|---|---|---|
| **Fig. 8** | $\tau$, 0 to 1000 | $D$, 0.00 to 0.30 | $\tau$, at **fixed T = 204** | $D_{emp}$ has a maximum at $\tau \approx 500$ |
| **Fig. 9** | $T$, 0 to 1000 | $D_{emp}/D_{th}$, **1.5 to 3.0** | $T$, with $\tau$ **tied to** $T$ | same maximum at $T^* \approx 500$ |

Data: N = 204 Nikkei names, 2000–2010, P = 5, Q = 10, and $T = N$ by their own rule.
$D_{RMT} \approx 0.83$ (quoted in the letter's Fig. 2 caption) in their
$\beta\pi$ convention (= 1.655 in mine).

The two axis ranges are the useful part. They are a free calibration check that costs
nothing, and I was failing both of them.

## 1. The bug: I was eigen-decomposing the covariance matrix, the paper uses the correlation matrix

`regime4.py` called `sample_covariance(w)` on the raw return window. `to_correlation_panel`
existed in `src/data.py` and was used by `fetch_data.py` for the MP edge — and nowhere else.
So every number in `results/us_*.csv` was a covariance-matrix number.

The paper is explicit: *"the distance between the top P eigenvectors of the true
**correlation** matrix C and the top Q eigenvector of the empirical **correlation** matrix
E"*.

**Independent confirmation from Fig. 7 (Fig. 1 of the letter).** The variogram panels have y-axes 0–600 (mode 1)
and 0–15 (mode 2), and the dotted line on them is $4\lambda_i^2/T$. On one of my windows:

| | $\lambda_1$ | $4\lambda_1^2/T$ | $\lambda_2$ |
|---|---|---|---|
| correlation | 48.6 | 46.3 | 10.1 |
| covariance (daily) | 1.28e-2 | 3.2e-6 | 3.5e-3 |

Only the correlation numbers can live on a 0–600 axis. This is not a judgement call.

**Why it mattered so much.** On a covariance matrix the top eigenvectors are tilted
towards whichever names were most volatile *in that window*. Median annualised vol across
my 175 names runs from roughly 20% to 80%, so volatility-rank turnover alone rotates the
top block — with the correlation structure completely frozen. That is a rotation the null
does not model and cannot subtract.

**Effect at T = 204, P = 5, Q = 10:**

| | $D_{emp}$ @ $\tau$=220 | $D_{emp}$ @ $\tau$=1000 | ratio @ $\tau$=T | peak $D_{emp}$ |
|---|---|---|---|---|
| covariance (what I had) | 0.277 | 0.489 | **3.13** | 0.784 |
| correlation (the paper's object) | 0.172 | 0.215 | **1.71** | 0.252 |
| paper's plotted range | — | — | 1.5 – 3.0 | < 0.30 |

After the fix I am inside both of the paper's axis ranges, at every $T$ from 100 to ~900.
Before the fix I was off by roughly 2× on $D_{emp}$ and 2–10× on the ratio.

$D_{th}$ was fine either way — $D_{th} \cdot T \approx 20.7$, flat to 2% across
$T \in [80, 1100]$, exactly the $1/T$ the formula asserts. So the whole discrepancy was in
$D_{emp}$.

Fixed: `regime4.py` now takes `--matrix {correlation,covariance}`, defaulting to
correlation. Old CSVs kept; new ones get a `_corr` suffix.

## 2. Things I checked that were NOT the problem

| checked | verdict |
|---|---|
| demeaning in `sample_covariance` | changes $D_{emp}$ by < 1e-4. Irrelevant. |
| directional $D(U_s, V_t)$ vs symmetrised | ≤ 3% at every $\tau$. Irrelevant. |
| fat tails (median kurtosis 10.5, max 174) | winsorising at 5σ moves $D_{emp}$ −3%, at 3σ −9%. Uniform shift, curve shape unchanged. |
| within-window vol drift (regime 2.3) | within-window CV² = 0.176 → 1.18× inflation. `--standardise` removes it. Shifts the level down ~14%, shape unchanged. |
| $\tau$ binning / `min_lag = T` | lags are exact multiples of `step`, no smearing. |
| $D_{RMT}$ ceiling | $D_{emp}$ peaks at 16% of the ceiling under correlation (50% under covariance). No compression. |

## 3. The real difference: my $D_{emp}(\tau)$ does not turn over, and it is my market, not my code

Even after the fix, at T = 204 my $D_{emp}$ rises monotonically from 0.172 at $\tau$ = 220
to ~0.22 by $\tau$ = 1200 and then flattens. The paper's peaks at $\tau \approx 500$ and
*falls*. Same for the right panel: my ratio climbs 1.44 → 3.8 over T = 100 → 1000 with no
maximum.

Since $D_{th} \propto 1/T$ exactly, ratio $\propto D_{emp}\cdot T$. Reading Fig. 9
back, their $D_{emp}$ at $\tau = T = 1000$ is roughly **a third** of their
$D_{emp}$ at $\tau = T = 500$. Mine falls by 16% over the same span. Their correlation
structure genuinely mean-reverts on a ~2-year clock; mine does not.

**Two pieces of evidence that this is the data and not the instrument:**

*(a) The paper's own control passes.* I rebuilt $D_{num}$ — synthetic Gaussian returns
with a fixed random eigenbasis and my empirical eigenvalue path, i.e. the null made flesh.
$D_{num}$ is **flat in $\tau$** (0.116 → 0.130 across the whole range) while $D_{emp}$
climbs to 0.25. So nothing in the measurement manufactures a $\tau$ trend, in either
direction. (Worth noting: $D_{num}/D_{th}$ runs 1.15–1.26 rather than 1.00, so Eq (6.1)
is ~20% low at q = 0.86. That is a level bias, not a shape bias, and it is a genuine
caveat on my q — see §5.)

*(b) The rise is dot-com vs GFC, by name and date.* Ranking all 7000 window pairs by
$D_{emp}$, the top 12 are **all** 2001-06→2002-08 paired against 2008-07→2008-09, at lags
1490–1790 days. The year × year map:

|  | 2001 | 2004 | 2008 | 2010 |
|---|---|---|---|---|
| **2001** | 0.121 | 0.184 | **0.274** | 0.257 |
| **2004** | 0.184 | 0.138 | 0.225 | 0.226 |
| **2008** | **0.274** | 0.225 | 0.156 | 0.207 |

My $D_{emp}$ peak sits at $\tau \approx$ 1500–2000 days because **that is the calendar
distance between the two most structurally different periods in US equities 2000–2010** —
the telecom/tech bust and the financial crisis, which sit at opposite ends of the sample.
It is not an accumulation time and it is not a mean-reversion time. The paper's 500 is
presumably the analogous number for the Nikkei.

The monotone rise survives everything: winsorising, day-standardisation, dropping every
window that touches Jun-08→Sep-09, dropping every window that touches Mar-00→Dec-02, and
dropping both. Excluding both crises the peak moves to $\tau$ = 1500 and the curve still
rises 0.162 → 0.219. The early/late split at matched lag disagrees by 13% typical / 34%
worst, so the $\tau$ axis is mostly doing real work — but "real work" here means
*this universe drifted in one direction across the decade*, not *it mean-reverts*.

Contributing: my candidate list is hand-assembled and unverified, and it survives to 175
names including AAPL, AMZN and NVDA, whose factor loadings genuinely migrated over
2000–2010. A price-weighted Japanese industrial index has no comparable secular rotation.

## 4. P and Q sweep (correlation, T = 204, N = 175)

| P | Q | $D$@220 | $D$@500 | $D$@1000 | peak $D$ | @ $\tau$ | ratio@220 | $D_{RMT}$ | peak as % of ceiling |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 0.019 | 0.024 | 0.034 | 0.098 | 2460 | 1.95 | 2.126 | 5% |
| 2 | 4 | 0.085 | 0.125 | 0.246 | 0.291 | 1100 | 1.89 | 2.040 | 14% |
| 3 | 6 | 0.114 | 0.143 | 0.193 | 0.233 | 2360 | 1.74 | 1.836 | 13% |
| 3 | 10 | 0.093 | 0.116 | 0.147 | 0.184 | 2320 | 1.80 | 1.511 | 12% |
| **5** | **10** | **0.172** | **0.194** | **0.215** | **0.252** | **2280** | **1.71** | **1.577** | **16%** |
| 5 | 20 | 0.134 | 0.151 | 0.164 | 0.198 | 2260 | 2.08 | 1.146 | 17% |
| 8 | 16 | 0.252 | 0.271 | 0.274 | 0.312 | 2080 | 1.77 | 1.338 | 23% |
| 10 | 20 | 0.320 | 0.340 | 0.360 | 0.401 | 2460 | 1.99 | 1.223 | 33% |

No (P, Q) produces a maximum near 500. The excess ratio is remarkably stable at 1.7–2.5
across the whole grid, which is reassuring — it is a property of the market, not of the
block sizes. (P,Q) = (2,4) is the only configuration whose peak lands anywhere near the
plotted range and that is mode-2/mode-3 hybridisation, not a real timescale.

## 4b. What shape *should* D(tau) have? A rising curve is the generic case

I had been treating my monotone rise as the anomaly and the paper's maximum as the
reference. That is backwards. To check, I built worlds where the true eigenbasis moves
and measured the subspace distance **between the true bases directly** — no sample
covariance, no windowing, no estimation noise at all. Whatever shape appears is what the
dynamics imply, before any measurement is layered on top. (`rot3.py`, `rot4.py`.)

**Brownian rotation — "the market is always changing", taken literally as a driftless
random walk on the Grassmannian:**

| $\tau$ | 100 | 200 | 400 | 800 | 1200 | 1600 | 2400 | 3200 |
|---|---|---|---|---|---|---|---|---|
| $D$ | 0.002 | 0.004 | 0.008 | 0.016 | 0.024 | 0.032 | 0.048 | 0.064 |
| $D/\tau \times 10^5$ | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |

**Exactly linear in $\tau$**, to three figures, over a factor of 32 in lag. The reason is
immediate: $D = -\sum_k \ln\cos\theta_k \approx \tfrac12\sum_k \theta_k^2$ for small
angles, and diffusion gives $\theta^2 \propto \tau$. It stays linear until the angles
become large, at which point it saturates at $D_{RMT}$ — so linear *forever* is
impossible, but the ceiling is a long way off (my data peaks at 16% of it, and the
paper's at 15%, which is the one number the two markets agree on precisely).

**Ornstein-Uhlenbeck rotation — same noise plus a restoring pull towards a preferred
configuration with timescale $\theta$:** rises, then **plateaus** at roughly $2$–$4\theta$.
$\theta$=200d flattens at $D \approx 0.08$ from $\tau$=400 on; $\theta$=400d flattens at
$D \approx 0.088$ from $\tau$=800 on. Not one of the five worlds produced a decline that
cleared its own standard error.

**And it cannot.** For any stationary process, as $\tau \to \infty$ the two configurations
become independent, so $D(\tau)$ approaches the independent-draw value *from below* and is
bounded by it. A curve that rises above that level and comes back down means the
configurations at lag 500 are **more different than two independent draws** — that is
anti-correlation at that lag, i.e. a genuine oscillation, not mean reversion. Mean
reversion gives a knee and a plateau. It does not give a maximum.

This matters for how the paper's sentence should be read:

> *"This plot reveals a marked maximum around $T^* \approx 2$ years, suggesting that the
> correlation matrix has some true dynamical evolution with a mean reversion time around
> $T^*$."*

The mechanism named does not produce the feature claimed. Two readings survive:

1. **The feature is a knee, not a maximum.** Then the paper and I are seeing the same
   physics with different reversion strengths, and their $\theta$ is around 125–250 days
   rather than 2 years — the knee sits at $2$–$4\theta$, not at $\theta$.
2. **The feature is a real decline.** Then the Nikkei's top-5 subspace genuinely
   oscillates, which is a stronger and more interesting claim than mean reversion, and one
   the paper does not argue for.

I cannot separate these from the caption text alone; it needs the figure read off properly.

**Sample size makes this hard for anyone.** Independent (non-overlapping) window pairs at
T=204 over 2765 days:

| $\tau$ | 220 | 500 | 1000 | 1500 | 2000 | 2300 | 2500 |
|---|---|---|---|---|---|---|---|
| all pairs | 118 | 104 | 79 | 54 | 29 | 14 | 4 |
| **independent** | 11.5 | 10.1 | 7.7 | 5.2 | **2.8** | **1.3** | **0.3** |

Past $\tau \approx 1800$ there are fewer than four independent observations. My own peak at
$\tau$=2280 is read off roughly one. Neither curve's tail is resolved, and the paper
carries no error bars.

**Where my data actually sits.** Subtracting the fixed-eigenvector floor and fitting
over $\tau \in [250, 2000]$:

$$\text{excess } D \propto \tau^{0.38}$$

against $\tau^{1}$ for pure diffusion and $\tau^{0}$ for a fully mean-reverted process.
So US 2000–2010 is **sub-diffusive**: partially reverting, with the top-5 subspace
wandering more slowly than a random walk but never settling. That is a more informative
statement than "it rises", and it is the thing to compare against Tokyo once that panel is
built. Caveat: the exponent inherits the uncertainty in the floor subtraction, which is
about 40% of the signal at short lag — treat it as "roughly 0.4", not 0.38.

## 5. Open, and worth flagging before any of this is written up

- **q = N/T = 0.86 at T = 204.** Far outside anything regime 2.2 calibrated (≤ 0.16). The
  paper is worse (N = T = 204, q = 1.0) so this is not a deviation from them, but $D_{num}$
  above says Eq (6.1) runs ~20% low here. Every ratio in this document should be read with
  a −20% asterisk on the denominator. Fix by cutting names (`--max-names`) or raising T,
  and re-measuring.
- **MP says Q = 24, I am using Q = 10.** Median MP-edge count across windows is 24
  (range 16–47) at T = 204. Q = 10 is the paper's choice and I have kept it for
  comparability, but by my own regime 3.2 criterion it is too small for this universe.
- **The Nikkei comparison is untested.** yfinance is not reachable from this sandbox, so I
  could not run 204 Tokyo names through the same pipeline. That is the one experiment that
  would settle "market, not code" outright rather than by inference. It is the obvious
  next step: `scripts/fetch_data.py --tickers nikkei225.txt --label nikkei`.

## Files

- `results/paper_comparison.png` — both panels reproduced, covariance vs correlation,
  plus the $D_{num}$ control and the year × year map.
- `results/us_T*_*_corr.csv` — the sweeps re-run on correlation matrices.
- `results/us_T*_P*Q*.csv` (no suffix) — the old covariance runs, kept for the record.
