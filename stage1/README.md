# Stage 1 — Instrument and forecast-entry gates

**Current status:** synthetic calibration, the 2000–2010 paper replication and
the full-history direction, coherence, ERSE-attribution and partial-flag gates
are complete through Regime 4.7. Stage 2 model fitting has not started, but its
master target is now frozen as $\mathrm{Flag}(N;1,3,6)$. The sub-universe
coherence scaling test remains necessary for a literal scaling claim; it is no
longer a blocker to fitting the first chronological forecast.

## Scope — what counts as stage 1

Stage 1 contains everything required before fitting a forecast:

1. Regimes 1–3 calibrate the overlap instrument in synthetic worlds where the
   truth is known.
2. Regimes 4.1–4.3 reproduce the published real-data measurements on frozen
   2000–2010 panels.
3. Regimes 4.4–4.7 use separate full-history panels through the latest 2026
   observation to ask whether a learnable and economically interpretable label
   exists.

No fitted forecast belongs to Stage 1. Regimes 4.4–4.5 are empirical findings,
but they are entry gates rather than ML results.

| date | regimes | what |
|---|---|---|
| 2026-08-01 | 1.1–1.5, 2.1–2.3, 3.1–3.2 | synthetic calibration: null, confounds, additivity, power, block choice |
| 2026-08-02 | 4.1–4.6 | real panels: eigenvalue variogram, subspace variogram, `T*`, tangent persistence, coherence, ERSE attribution |
| 2026-08-03 | 4.7 | partial-flag persistence, coherence and ERSE-residual validation at dimensions 1, 3 and 6 |

### Current verdict

- **Instrument:** pass. Under a static population covariance, measured rotation
  matches the finite-window null; injected rotation adds to that null and is
  recovered once it clears the detection floor.
- **Heavy tails and within-window volatility:** characterised. Standardising
  each day by its cross-sectional volatility removes the common-scale
  inflation without fitting a Student parameter. This is the primary real-data
  specification.
- **Block choice:** the adaptive hypothesis failed. A spectral-gap rule chooses
  $P=1$ and is exactly blind to lower-mode rotations. $P=3$ is a documented
  coverage/noise compromise, not an optimum. The MP edge recovers $Q$ only in
  its ideal homogeneous-noise world and is not a credible real-data signal
  count. Existing $D(P,Q)$ experiments retain fixed $P=3,Q=6$; Regimes
  4.4–4.6 use $P=3$, and Regime 4.7 promotes $(1,3,6)$ to a partial flag.
- **Direction:** pass on all four full-history panels. The preceding tangent
  direction contains information, but repeating its full length is 47–67%
  worse than holding the current subspace fixed. The S&P result is exploratory
  because it currently has only 20 shuffle-null repetitions.
- **Coherence:** pass against the independently timed loading-history null on
  all four panels. The common component is broadest on S&P and weakest on CAC.
- **ERSE attribution:** pass for the claim that the directional signal is not
  ERSE rearranged. ERSE explains only 0.05–0.92% of outgoing tangent energy and
  removing that direction leaves the persistence cosine essentially unchanged.
  ERSE itself is a poor future-subspace forecast on these daily panels.
- **Flag target:** pass. The nested market/core/buffer state
  $\mathrm{Flag}(N;1,3,6)$ preserves significant persistence and coherence on
  Nikkei, DAX and CAC; S&P is strongly sign-consistent but exploratory at its
  current 20-null resolution. The outer six-space is weaker than the market
  and top-three spaces on DAX and CAC, so it is a containing buffer rather than
  an equally strong standalone target.
- **Not yet established:** whether the coherence result scales with universe
  size and whether a chronologically held-out damped predictor improves
  subspace or covariance risk. The first is a mechanism-claim check; the second
  is now the first Stage 2 experiment.

See the [Stage 2 forecast-entry report](../stage2/README.md) for the model-facing
interpretation and [BUILDNOTES](../BUILDNOTES.md) for the chronological
experimental notebook.

## Notation

| symbol | meaning | more is |
|--------|---------|---------|
| `N` | number of assets; `C` is `N x N` | **worse** — `C` has `N(N+1)/2` parameters but the data supplies only `N x T` numbers |
| `T` | observations (days) in one window | better |
| `q = N/T` | aspect ratio; parameters per observation. `q -> 0` data-rich, `q = 1` as many assets as days | lower is better |
| `P` | size of the inner eigenvector block being tracked | — |
| `Q` | size of the outer block it is compared against, `Q >= P` | — |
| `Y_t` | leading $P$-space at time $t$, a point on $\mathrm{Gr}(N,P)$ | — |
| `H_t^-`, `H_t^+` | incoming and outgoing Grassmann tangent velocities at $Y_t$ | alignment is more learnable |
| trials | Monte Carlo repetitions of a whole experiment | better — pure precision, costs only compute |

`N`, `T`, `P`, `Q` describe the problem. `trials` describes the measurement of
it, and is the only one of the five that is free to increase. Easy to conflate
`N` with `trials` in the tables below, since both appear as bare integers.

---

## Hypothesis ledger

| # | Bet | Prediction | Result | Verdict |
|---|-----|------------|--------|---------|
| 1.1 | Static `C`, true spectrum known | Measured excess ~ 0 within noise | Eq (7) ratio 0.99-1.01; Eq (10) ratio 0.99-1.03 at T=1000/4000/16000 | **Pass** |
| 1.2 | Eq (9) eigenvalue variogram | Empirical `<(λs-λt)²>` matches `4λ²/T` | Within 20% for the top 3 modes | **Pass** |
| 1.3 | Excess scales as 1/T, no floor | `D(T=1000)/D(T=4000) = 4` | 4.0 within 15% | **Pass** |
| 1.4 | `D_RMT` benchmark | Reproduces the paper's 0.83 at (5,10,204) | 0.8275, paper convention | **Pass** |
| 1.5 | Heavy-tailed null | Student common-scale inflation follows $(\nu-2)/(\nu-4)$ and can be removed without fitting $\nu$ | Mean inflation follows the prediction; daily cross-sectional standardisation removes its $\nu$ dependence | **Pass** |
| 2.1 | Regime 2 — fixed eigenvectors, time-varying eigenvalues | `D_num ≈ D_th` | ratio 1.0015 at `sigma=0.06`; pooled 1.005 ± 0.003 over 4000 trials | **Pass** |
| 2.2 | Estimated spectrum substituted for the true one | Bias small and shrinking in `T` | Sign and size depend on the sampled spectral geometry; no portable law and bootstrap correction fails | **Characterised, not corrected** |
| 2.3 | Common eigenvalue drift *within* a window | Eq (10) fails; `D` inflated | Inflation is $\langle c^2\rangle/\langle c\rangle^2$; daily standardisation removes the common-scale mechanism synthetically and on real panels | **Confound found and controlled** |
| 3.1 | Regime 3 — injected drift | Recovers injected magnitude | `D_emp = D_th + D_inject` to 2% over a 300x range of theta, 0.1% once clear of the floor | **Pass** |
| 3.2 | Automatic block choice | Adaptive $P$ and MP-edge $Q$ remove the fixed block choice | Gap-based $P$ chooses 1 and misses lower-mode rotations; MP $Q$ works only under a homogeneous-noise bulk and explodes on real spectra | **Fail — retain explicit fixed blocks** |
| 4.1 | Eigenvalue variogram reproduction | Reproduce the overlap artifact and excess eigenvalue motion | Windowing shape reproduced; standardised S&P clears the null, small $T=N$ panels do not | **Pass for replication; small panels inconclusive** |
| 4.2 | Subspace variogram | Excess grows with lag if the population space evolves | S&P excess grows after overlap saturates; static-offset reading rejected there | **Pass on S&P** |
| 4.3 | Optimal measurement window | Reproduce the U-shaped $D_{emp}(T)$ curve | U-shape appears on all panels; minima are imprecise because fewer than three independent pairs support them | **Shape reproduced; location uncertain** |
| 4.4 | Tangent persistence | $H_t^-$ aligns with $H_t^+$ beyond a matched calendar-shuffle null | Direction passes on all full-history panels (S&P exploratory at 20 nulls); unit-speed forecast loses to static everywhere | **YAY direction; NAY full step** |
| 4.5 | Cross-asset coherence | Common tangent component exceeds independently timed loading histories | Passes on all panels under 999 synchrony-null repetitions | **YAY; sub-universe scaling remains** |

## Reproduction map

Run from the repository root:

| Regime | Command |
|---|---|
| 1.1–1.4 | `python -m pytest tests/test_regime1_static.py tests/test_overlap_properties.py tests/test_rmt_benchmark.py -q` |
| 1.5 | `python scripts/regime1_5_student.py` |
| 2.1–2.3 | `python -m pytest tests/test_regime2_varying_eigenvalues.py -q` |
| 3.1 | `python -m pytest tests/test_regime3_injected_drift.py -q` |
| 3.2 | `python -m pytest tests/test_regime3_2_block_choice.py -q` |
| 4.1 | `python scripts/regime4_1_variogram.py --label nikkei` |
| 4.2 | `python scripts/regime4_2_subspace.py --label sp500` |
| 4.3 | `python scripts/regime4_3_tstar.py --label nikkei` |
| 4.4 | `python scripts/regime4_4_tangent.py --label nikkei_full --T 250 --mode standardised --shuffles 99 --no-plot` |
| 4.5 | `python scripts/regime4_5_coherence.py --label nikkei_full --T 250 --shuffles 999` |

## Regime 1 status — passing

Run: `python -m pytest tests -q` from the repo root.

Settings: `N=40, P=3, Q=6, T=2000, 400 trials`, factor spectrum
`[25, 10, 6, 4, 3, 2.2]` over a bulk from 1.3 down to 0.4, Haar-random basis.

The true spectrum is passed to the null, never an estimated one. That is the
entire point of the regime: it is the guard against metric-liar #4.

## Regime 2 status — passing

Run: `python -m pytest tests/test_regime2_varying_eigenvalues.py -q`.

Settings: regime 1's spectrum and `N, P, Q, T = 40, 3, 6, 2000`, with the top 3
eigenvalues given independent unit-mean lognormal jitter of `sigma = 0.06` per
window and the basis held fixed forever. 400 trials.

### What this regime adds over regime 1

Regime 1 passed the *same* spectrum to both windows, so Eq (10) collapsed to
`2 x Eq (7)` and its two-spectrum structure was never executed. More
importantly, regime 1 could only establish that the instrument does not invent
rotation out of nothing. It could not establish that the instrument does not
mistake a change in eigenvalue *magnitude* for a change in eigenvector
*direction*, because nothing was moving. Eigenvalues demonstrably do move in
real data, so if that motion leaked into `D`, every downstream result would be
a restatement of volatility rather than a claim about correlation structure.

It does not leak. `D_num / D_th = 1.0015` at these settings, and 1.005 ± 0.003
pooled over 4000 trials — the same half-percent order as regime 1's static
result, consistent with the dropped `O(1/T²)` term and not with a confound.

### The `sigma` ceiling is a property of the test world, not the instrument

`perturb_top` raises rather than resorting when the jitter reorders the
spectrum. A crossing permutes the eigenvector labels, and a permuted label is
indistinguishable from a rotated eigenvector — precisely the signal this regime
must not contain. Resorting would smuggle it in silently.

The usable ceiling is set by the tightest gap in the perturbed block. On this
spectrum the binding pair is `10 : 6`, giving 0 rejections in 100000 draws at
`sigma = 0.06`, 20 at `sigma = 0.10`, and 1307 at `sigma = 0.15`. A spectrum
with wider top-block gaps tolerates more. An earlier attempt used a 3-factor
spectrum `[25, 10, 4]` to buy jitter headroom; it works, but it moves the `Q`
boundary into the dense bulk and is not worth the trade.

### Level versus shape

`D` is invariant under a common rescaling of the whole spectrum. A market-wide
volatility change that scales every eigenvalue equally moves the null not at
all; only the relative arrangement registers. This is unit-tested.

The consequence is not the reassuring one it first looks like. See 2.3.

### 2.2 — the cost of substituting estimated eigenvalues

Notation: `q ≡ N/T`, the aspect ratio — variables estimated per observation.
`q → 0` is data-rich; `q = 1` means as many assets as days. Sweeps below are
gridded on `(N, q)` with `T = N/q` derived, not on `(N, T)`, so that reading
down a column holds `q` fixed while `N` and `T` both move. That is the only
layout in which "is this a `q`-law?" is answerable.

The paper allows replacing the true `λ` by empirical estimates "up to
corrections of order `T^(-3/2)`". Since `D` is itself `O(1/T)`, that predicts a
*relative* error of order `T^(-1/2)`. Regime 2 is the last world in which both
the true and the estimated spectrum are available, so it is the only place the
claim can be measured rather than taken on trust.

| `T` | `N/T` | est/true | bias |
|-----|-------|----------|------|
| 250 | 0.16 | 0.9893 | −1.07% |
| 500 | 0.08 | 0.9969 | −0.31% |
| 1000 | 0.04 | 0.9978 | −0.22% |
| 2000 | 0.02 | 0.9993 | −0.07% |

(1200 trials per row. An earlier 700-trial single-seed run put `T=250` at −1.38%;
that was noise, and −1.07% ± 0.18% is the settled value.)

The relative bias falls roughly as `1/T`, i.e. an absolute correction of order
`T^(-2)` — steeper than advertised. **This law holds only for `N/T` below about
0.3, while the paper-replication panels use `T=N`.** Extending the sweep towards
the ratio used by that replication:

| `T` | `N/T` | measured bias | `1/T` law predicts |
|-----|-------|---------------|--------------------|
| 250 | 0.16 | −1.15% | −1.15% |
| 150 | 0.27 | −2.06% | −1.91% |
| 100 | 0.40 | −3.13% | −2.87% |
| 70 | 0.57 | −5.28% | −4.10% |
| 55 | 0.73 | −8.58% | −5.22% |
| 45 | 0.89 | −11.54% | −6.38% |

The law is fine to `N/T ≈ 0.4` and then fails, reaching 1.8x its own prediction
by `N/T ≈ 0.9`. Anything quoted from the low-`N/T` table understates the error
at high `N/T` by roughly a factor of two. Three caveats, in order of weight:

- **The sign is not even fixed.** Everything above is at `N = 40`. Holding `q`
  fixed and varying `N` — which also varies the bulk density, since
  `factor_spectrum` keeps six factors and fills the rest — the bias *changes
  sign*:

  | `q` | `N=20` | `N=40` | `N=80` |
  |-----|--------|--------|--------|
  | 0.2 | −6.2% | −1.5% | **+1.0%** |
  | 0.4 | −13.6% | −3.0% | **+2.0%** |
  | 0.6 | −22.3% | −5.6% | **+2.0%** |

  A denser bulk spreads under sampling and pushes `λ̂_j` upward towards the top
  block, shrinking the gaps and *inflating* `D_th(est)`. A sparse bulk does the
  opposite. So "substitution inflates apparent excess" is false in general — it
  is a statement about `N = 40` with this bulk, nothing more.
- **It is not universal.** The size is set by how tight the spectral gaps are
  relative to sampling noise, not by anything about the instrument. Holding
  `sigma` fixed at 0.06 and changing only the spectrum:

  | spectrum | `T=250` | `T=1000` |
  |----------|---------|----------|
  | `[25, 10, 6, 4, 3, 2.2]` | −1.38% | −0.28% |
  | `[25, 10, 4]` (compressed) | −5.08% | −1.80% |

  A factor of 3.7 at `T=250` and 6.4 at `T=1000`. The jitter amplitude is by
  contrast almost irrelevant: at `T=250` the bias moves only from −1.16% to
  −1.33% as `sigma` goes 0.02 → 0.10. So every number in this section belongs
  to its spectrum, and none of them transfers. Regime 4 runs at `T = 204` on a
  real spectrum whose gaps are not known in advance — the worst corner of this
  table, and the one where the figure cannot be looked up beforehand.

- **It is T-dependent, and `T*` is found by scanning `T`.** This is the part
  that actually threatens a result rather than just a number. The bias shrinks
  monotonically as `T` grows, so it inflates `D_emp / D_th` at short windows and
  leaves it alone at long ones — a monotone downward drag on the ratio across
  exactly the axis the `T* ≈ 2yr` peak is read off. It cannot manufacture a peak
  out of a flat curve, but it can move one, and it biases the location towards
  shorter `T`. Any `T*` quoted before this is corrected is not a measurement.

### The decomposition: two mechanisms, opposite signs

`D_th` sums `λ_i λ_j / (λ_i − λ_j)²` over `i < P` and `j > Q`, so it is driven
entirely by the gap between the top block and the bulk. Substituting sample
eigenvalues moves *both* ends of that gap, in opposite directions. Swapping in
only the estimated top block, or only the estimated bulk, separates them:

| `N` | `q` | top only | bulk only | sum | both | per-dataset sd |
|-----|-----|----------|-----------|-----|------|----------------|
| 20 | 0.2 | −0.98% | −5.94% | −6.92% | −6.77% | 12.7% |
| 20 | 0.6 | −1.70% | −21.38% | −23.08% | −22.54% | 17.3% |
| 40 | 0.2 | −1.88% | +0.30% | −1.58% | −1.60% | 9.6% |
| 40 | 0.6 | −5.57% | −0.18% | −5.75% | −6.25% | 15.2% |
| 80 | 0.2 | −2.83% | +3.71% | +0.87% | +0.59% | 6.6% |
| 80 | 0.6 | −7.21% | +11.83% | +4.62% | +2.20% | 11.7% |
| 160 | 0.2 | −3.21% | +5.30% | +2.10% | +1.69% | 4.8% |
| 160 | 0.6 | −8.07% | +18.01% | +9.94% | +6.26% | 8.9% |

- **Top-block term — always negative.** Sample eigenvalues of the top block are
  pushed *up* by repulsion from the bulk. That widens `λ_i − λ_j`, shrinking
  `D_th`. Monotone in `q` and in `N`, and never changes sign anywhere tested.
- **Bulk term — positive whenever the bulk is dense.** The sample bulk spreads,
  and its upper edge climbs *towards* the top block. That narrows the gap and
  inflates `D_th`. It is the larger of the two whenever `N` is large.

The driver is the bulk edge, not `N` or `q` as such. Holding `N = 80`, `q = 0.4`
and the top block fixed, and changing only the bulk:

| bulk range | est. bulk top (true) | top | bulk | both |
|------------|----------------------|-----|------|------|
| 2.0 → 0.4 | 3.21 (2.00) | −9.18% | +15.50% | +2.30% |
| 1.3 → 0.4 | 2.27 (1.30) | −4.73% | +7.61% | +1.87% |
| 1.3 → 1.1 | 2.86 (1.30) | −8.06% | +15.47% | +4.02% |
| 0.6 → 0.2 | 1.07 (0.60) | −1.15% | +0.88% | −0.32% |

`λ_P = 6.0` in every row. The bulk term varies by a factor of 17 at fixed `N`
and `q`, tracking exactly how far the estimated bulk edge climbs toward `λ_P`.

**This is why nothing generalises.** The net error is a residual of two large,
opposite, comparably-sized terms — typically three to five times smaller than
either. `N = 40` with this bulk happens to sit near the cancellation point,
which is why the first six configurations tested all looked like a single clean
mechanism with a stable sign. They were an accident of the grid.

### The per-dataset scatter is larger than the bias

The rightmost column above is the standard deviation of `D_th(est)/D_th(true)`
across independent datasets. Compare it to the bias:

| | |
|---|---|
| `|bias| / sd` across the 16-point grid | 0.09 to 1.30 |
| configurations where sd exceeds the bias | 15 of 16 |

At `N=40, q=0.2` the systematic tilt is −1.6% while a single dataset lands
anywhere within roughly ±10%. **The dominant error is variance, not bias.** That
is a different problem from the one this section started out chasing, and it has
a different fix: `D_th` needs an uncertainty band, not a bias correction.

### …but it does not survive into the ratio, which is what gets reported

The scatter above is on `D_th` in isolation, and `D_th` in isolation is never
reported. What gets reported is `D_emp / D_th`, and both come from the *same*
returns — so their errors are correlated and cancel. Measured over 1500 trials:

| `N` | `q` | `D_emp` alone | `D_th(est)` alone | ratio, true spectrum | ratio, est spectrum | corr |
|-----|-----|---------------|-------------------|----------------------|---------------------|------|
| 40 | 0.2 | 19.5% | 6.7% | 19.5% | **18.8%** | +0.27 |
| 80 | 0.4 | 13.4% | 6.8% | 13.4% | **12.5%** | +0.38 |
| 160 | 0.6 | 10.4% | 6.3% | 10.4% | **9.3%** | +0.46 |

**Substituting the estimated spectrum adds no noise to the ratio at all** — it
subtracts a little. In every configuration the ratio computed with estimated
eigenvalues is *quieter* than the one computed with the true spectrum, because
a window that happens to produce a large `D_emp` also produces a large
`D_th(est)`, and the two move together.

The ratio's noise is set almost entirely by `D_emp`'s own sampling scatter, which
is 10–20% per window pair and is unavoidable whatever spectrum the null is fed.
It also averages down across window pairs, which the systematic part does not.

So the earlier framing — "the dominant error is variance" — was measured on the
wrong object. Corrected: **the variance is real but free, and the systematic
part is what survives.** For the ratio that systematic part is +1.5%, −1.4% and
−5.4% across the three rows above, against a target effect (`D_emp/D_th` of
1.5–3.0 in the paper) of +50% to +200%. It is not a threat to the sign of the
result. It remains a threat to `T*`, because it is `T`-dependent and does not
average away.

### Curve fits, and what they are worth

Fitted to the `N = 40` sweep (six points, `|bias|`):

| model | R² | max residual |
|-------|-----|--------------|
| `c1·q + c2·q²`  (c1=0.042, c2=0.098) | 0.997 | 19% |
| `c·q^1.5` | 0.987 | 21% |
| free power law, `17.6·T^(−1.34)` | 0.978 | 15% |
| `c·q/(1−q)` | −4.99 | 191% |
| `c·T^(−1/2)` — the paper's implied rate | 0.519 | 138% |

Two things to take from this and one not to.

The paper's bound is **not violated, it is loose**. A `T^(−3/2)` absolute
correction implies a `T^(−1/2)` relative one; measured is `T^(−1.34)`. The
substitution decays far faster than the paper needs it to. `q/(1−q)`, the
obvious RMT-flavoured guess, is decisively rejected.

What *not* to take: **at fixed `N`, `q = N/T` and `1/T` are the same variable.**
No fit above can distinguish a `q`-law from a `T`-law, and the `N`-sweep in the
previous section shows the truth is neither — the bias is not a function of `q`
alone, nor of `T` alone. The exponent −1.34 describes one spectrum at one `N`.

### The bootstrap correction does not work

The obvious fix is to treat the estimated spectrum as pseudo-truth, regenerate
returns inside that world, and measure the bias there — using only what real
data provides. It fails:

| `N` | `T` | `q` | true bias | bootstrap says | recovered |
|-----|-----|-----|-----------|----------------|-----------|
| 20 | 50 | 0.4 | −14.31% | −20.61% | 144% |
| 40 | 100 | 0.4 | −3.37% | −8.02% | 238% |
| 80 | 200 | 0.4 | **+1.26%** | −1.17% | **−93%** |

It over-corrects by 1.4–2.4x where the sign is right, and gets the sign wrong
where the true bias is positive. Applying it would do more damage than leaving
the bias alone.

The reason is structural, not a tuning problem. The bootstrap requires the
plug-in spectrum to be roughly unbiased, but `λ̂` is *already* spread relative to
`λ` — and that spread is precisely the quantity generating the error. So the
bootstrap world is systematically wider than reality and reproduces the wrong
bias. Fixing it needs a spectrum estimator that undoes the spreading — nonlinear
shrinkage in the Ledoit–Péché / QuEST family — not a resample of the raw one.

**Status: characterised, not corrected.** The mechanism is now understood — two
opposing terms set by the top-block/bulk gap, netting to a small unstable
residual — but there is no portable number, no portable law, and no working
correction. Three things transfer:

1. **Do not attempt a bias correction.** The net bias is smaller than the
   per-dataset scatter in 15 of 16 configurations, and its sign depends on bulk
   density. Correcting it would add error.
2. **Put an uncertainty band on `D_th`.** Rebuild this regime at regime 4's
   exact `(N, T, spectrum)` and report the spread, not a point value. Cheap, and
   it is the honest object.
3. **Watch the bulk edge.** The single best predictor of trouble is how close
   the estimated bulk's upper edge sits to `λ_P`. It is directly observable on
   real data — no true spectrum needed — so it can be checked before committing
   to a `P`/`Q` choice. Choosing `P` and `Q` so the gap is wide is the one
   design lever available.

### 2.3 — where Eq (10) actually breaks

Eq (10) assumes each window has one well-defined spectrum. Regime 2.1 satisfies
that: the spectrum is constant *within* each window and only differs *between*
them. Under that arrangement Eq (10) is exact for arbitrarily large
between-window changes — the "sufficiently slowly varying" hedge in the paper is
not about between-window movement at all.

It is about movement *inside* a window. Let the overall level ramp linearly from
`(1-h)` to `(1+h)` across the window. A whole-window estimate recovers the
time-average, but the sampling noise is governed by the time-average of the
*square*, so `D` is inflated by `<c²>/<c>² = 1 + h²/3`:

| `h` | measured ratio | `1 + h²/3` |
|-----|----------------|------------|
| 0.0 | 1.008 | 1.000 |
| 0.6 | 1.144 | 1.120 |
| 0.8 | 1.233 | 1.213 |

Zero rotation is present at every row. This is the shape a false positive takes,
and it is the one confound regime 2 finds rather than clears.

Note the inversion against the invariance result above. A common rescaling held
constant across a window is invisible to `D`. The same rescaling spread out
*over* the window is not — it inflates `D` quadratically in the drift amplitude.
Since realised volatility plainly varies inside any multi-year window, this is a
live mechanism on real data, not a synthetic curiosity, and it is not something
`D_emp > D_th` alone can distinguish from genuine rotation.

**Resolved operationally in Regimes 1.5 and 4:** daily cross-sectional
standardisation divides out this common scale before estimating a window. In
the synthetic drift experiment it reduces a 183% inflation to 1.3%, and it also
removes the Student common-scale inflation. The real-panel analyses therefore
report the standardised series as the primary specification. This controls the
specific common-scale mechanism derived here; it does not claim that all forms
of covariance nonstationarity have disappeared.

### Drift confined to the top block does nothing

A first attempt ramped only the top 3 eigenvalues and found no effect at any
`h` up to 0.9. That test was degenerate, not reassuring: `D`'s numerator is
`λ_i λ_j` with `i` in the top block and `j` in the bulk, so if the bulk is held
fixed the time-average factorises as `<λ_i> λ_j` and the drift cancels exactly.
Only drift that is common to both sides of the `P/Q` split can break the
factorisation. Recorded because the null result is misleading on its own.

## Regime 3 status — 3.1 passing, 3.2 answered in the negative

Run: `python -m pytest tests/test_regime3_injected_drift.py tests/test_regime3_2_block_choice.py -q`.

Regimes 1 and 2 asked whether the instrument reports rotation when there is
none. Regime 3 asks the mirror question, and in doing so tests the one
assumption the other two structurally could not.

### Two different worlds, deliberately

The numbers in 3.1 and 3.2 are not comparable, and the difference is not an
oversight:

| | 3.1 | 3.2 |
|---|---|---|
| bulk | `linspace(1.3, 0.4)`, as regimes 1 and 2 | flat at 1.0 |
| injection | single Givens plane | whole block |
| `D_inject` | `-ln(cos theta) / P` | `-ln(cos theta)` |

3.1 continues the established world so its results sit alongside regimes 1 and
2. 3.2 needs a bulk that is genuinely sampling noise, because the
Marchenko-Pastur edge is meaningless otherwise — see below. And 3.2 needs an
injection whose cost does not depend on `P`, or the comparison it exists to
make is rigged before it starts.

Consequently, the raw detection angles from 3.1 and 3.2 should not be compared
as though one superseded the other: they describe different spectra and
different injected events.

### 3.1 — what the subtraction rests on

Every result downstream is `excess = D_emp - D_th`. That subtraction assumes
**additivity**: that noise-rotation and real-rotation combine by addition.
There is a reason to expect it, since for small angles
`D = -(1/P) sum_k ln cos(theta_k) ~ (1/2P) sum_k theta_k^2` and squared angles
from independent sources add. But "approximately, for small angles" is doing
the work in that sentence, and regimes 1 and 2 both had zero signal, so the
assumption never had to hold.

Injection is a single Givens rotation, chosen so the truth is known in closed
form rather than simulated: exactly one principal angle equals `theta` and the
rest are zero, giving `D_inject = -ln(cos theta) / P`.

Settings: regime 1's spectrum, `N, P, Q, T = 40, 3, 6, 2000`, mode 0 rotated
into mode 10, 300–400 trials.

| `theta` | `D_inject` | predicted | `D_emp` | ratio | recovery |
|---|---|---|---|---|---|
| 0.00 | — | 0.001930 | 0.001967 | 1.019 | — |
| 0.05 | 0.000417 | 0.002347 | 0.002395 | 1.021 | 1.115 |
| 0.10 | 0.001669 | 0.003599 | 0.003656 | 1.016 | 1.034 |
| 0.20 | 0.006712 | 0.008642 | 0.008710 | 1.008 | 1.010 |
| 0.30 | 0.015231 | 0.017160 | 0.017232 | 1.004 | 1.005 |
| 0.80 | 0.120464 | 0.122394 | 0.122257 | 0.999 | 0.999 |

Additivity holds to 2% across a 300-fold range of injected magnitude, tightening
to 0.1% once the injection clears the noise floor. **The subtraction is
legitimate.**

The residual is a roughly constant *absolute* offset of about 3% of `D_th` —
the same `O(1/T^2)` positive bias regime 1 showed at +0.25% and regime 2 at
+0.5%. A fixed offset matters proportionally more the smaller the quantity it
is divided by, which is the whole of the `recovery` column: 1.115 at
`theta=0.05`, 1.010 at `theta=0.20`. Near the detection floor, recovered
magnitude runs about 10% high. This does **not** affect the detection threshold
below, because that threshold is derived from a null distribution carrying the
same offset, so it cancels.

Verified out to `theta = 0.8` rad (46 degrees). Beyond that, untested.

### Guarding the injection

Three noiseless checks run before any sweep, because an injection that injects
nothing produces a clean-looking null result:

| rotate mode 0 into | where it lands | `D` |
|---|---|---|
| mode 1 | inside the top-`P` block | 0 |
| mode 4 | the `P..Q` buffer | 0 |
| mode 10 | past `Q` | `-ln(cos theta)/P`, exactly |

The middle row is the only direct evidence in the project that `Q > P` does
what it is supposed to. A fourth test pins the boundary itself: rotation into
mode `Q-1` is invisible, into mode `Q` it is not.

### Power curve — settled calibration

Not a hypothesis and so not a ledger row — a recorded characterisation, and the
exit gate's third condition. For a **single** pair of windows, set the threshold
at the 95th percentile of the zero-rotation null (5% false-positive rate), then
ask what injected angle reaches 80% power. The settled percentile calibration is:

| `T` | Gaussian | Student `nu=12` | `nu=8` | `nu=6` | `nu=6`, standardised |
|---|---|---|---|---|---|
| 250 | 15.8° | 18.1° | 21.4° | 28.2° | **17.1°** |
| 500 | 11.2° | 12.7° | 14.6° | 17.8° | **11.6°** |
| 1000 | 7.8° | 8.9° | 9.7° | 12.2° | **8.2°** |

The Gaussian column gives `theta_min * sqrt(T) = 250, 249, 247` degree-days,
the expected `1/sqrt(T)` scaling from `D_inject ~ theta^2/(2P)` against a
`1/T` floor. Fat tails increase the *upper-tail* detection cost more than the
mean correction `(nu-2)/(nu-4)` predicts; daily standardisation almost restores
the Gaussian threshold.

**The honest two-year headline is therefore 11°, not the retired 8° scan.** On
`nu=6` returns it is 18°, and after standardisation it is 12°. These are
single-pair thresholds at `N=40, P=3, Q=6`, not transferable panel-wide floors.
A lagged real-data estimate averages dependent window pairs, so it cannot claim
a naive `sqrt(n)` improvement without a dependence-aware calibration.

### 3.2 — automatic block choice fails

**The MP edge supplies a deterministic rule only under its noise model.** Its
upper edge, `lam_+ = sigma^2 (1 + sqrt(N/T))^2`, bounds the sample spectrum of a
homogeneous noise bulk. `sigma^2` is re-estimated from the sub-edge eigenvalues
and iterated to a fixed point because factors inflate a naive average. This is
implemented as `q_from_mp_edge`; it is a useful diagnostic, not an accepted
real-panel selector.

On a spectrum of six factors over a flat homogeneous bulk at 1.0, it recovers
`sigma^2 = 1.0000` and `Q = 6`, and picks `Q = 6` from *sample* spectra in
99.6% of trials. That validates the implementation in the world where the
criterion applies; it does not eliminate a free parameter on real equities.

This is Marchenko & Pastur (1967) by way of Laloux, Cizeau, Bouchaud & Potters,
PRL 83 (1999) 1467 — cited by Allez & Bouchaud, arXiv:1203.6228, so the rule comes
from the instrument's own lineage rather than from outside it.

**Determined is not the same as justified, and the difference shows up on real
data.** The rule is closed-form and tuning-free whatever spectrum it is handed —
that property is unconditional. What is conditional is the reading of its
output as "the number of modes above the noise", because that requires MP's
noise model, and MP assumes the residual variances are homogeneous.

Two ways real data breaks that, both unit-tested:

- **Volatility spread across names.** MP describes variables of a common
  variance; equities differ by factors of several. On a raw covariance the
  criterion collapses `sigma^2` and calls most of the spectrum signal. The edge
  must be taken on the **correlation** matrix — `to_correlation_panel` — which
  is what Laloux et al. do. On a synthetic panel with realistic volatility
  spread this alone moves `Q` from 93 to 37.
- **Heterogeneous factor loadings.** Even after correlation scaling, unequal
  loadings leave unequal *residual* variances, which widen the bulk past the
  edge. Three true factors, `N=111`, `T=2765`:

  | residual variance spread | `Q` reported |
  |---|---|
  | 0.03 (homogeneous) | 3 |
  | 0.13 (realistic betas) | 27 |
  | 0.29 (adversarial) | 78 |

  The flat-bulk result above is the first row. It validated the criterion under
  precisely the condition that makes it exact, which real equities do not meet.

Applied to regime 1's `linspace` bulk the procedure returns `Q = 27` for the
same reason — that bulk is genuine spread structure, not noise.

The edge is therefore retained as a model-check: a very large return says that
the residual bulk does not resemble homogeneous MP noise. It must **not** then
be fed back as a huge real-data `Q`, because everything inside the `P..Q` buffer
is an exact blind spot. The paper's `Q=2P` is not theoretically selected either.
For the calibrated overlap experiments this project keeps the declared
`P=3,Q=6`; the paper replication keeps `P=5,Q=10`; the Grassmann tangent tests
track `P=3` and do not use `Q` at all.

**`P` has no dominating choice.** Detection threshold by block size, `Q = 6`
fixed by the edge, whole-block injection so `D_inject` is `P`-independent:

| `P` | whole block tilts | only mode 0 | only mode 2 | only mode 4 |
|---|---|---|---|---|
| 1 | 1.76° | 1.76° | **blind** | **blind** |
| 2 | 2.10° | 2.97° | **blind** | **blind** |
| 3 | 2.58° | 4.46° | 4.46° | **blind** |
| 4 | 3.04° | 6.07° | 6.07° | **blind** |
| 5 | 3.49° | 7.80° | 7.80° | 7.80° |

Read the first column alone and `P=1` wins. Read across and that conclusion is
worthless: `P=1` cannot see rotation in modes 2 or 4 at all — not poorly,
**exactly zero**, with no noise involved and no quantity of data able to
recover it. The blindness is exact and is unit-tested without any sampling.

So the trade is sharpness against coverage, and the cost of coverage is
superlinear. `D_inject` is `P`-independent by construction, so only the noise
moves, and each mode added to the block sits closer to the bulk with a smaller
gap:

| mode | `lambda` | noise contribution | vs mode 0 |
|---|---|---|---|
| 0 | 25.0 | 1.476 | 1.0x |
| 1 | 10.0 | 4.198 | 2.8x |
| 2 | 6.0 | 8.160 | 5.5x |
| 3 | 4.0 | 15.111 | 10.2x |
| 4 | 3.0 | 25.500 | 17.3x |
| 5 | 2.2 | 51.944 | 35.2x |

35-fold across six factors. The `1/P` in `D` cannot damp a term growing that
fast.

**The ledger bet fails.** Adaptive-gap selection picks `P=1` here, the largest
log-gap being `ln(25/10) = 0.916` at the very top. And `P=1` is blind in two of
the three injection scenarios. The gap rule is not arbitrary — Davis-Kahan
bounds subspace rotation by `||E|| / delta`, so wide gaps genuinely do buy
stability, and Eq (7) has precisely that shape with the gap in the denominator
and the noise magnitude on top. But stability is not the objective. A perfectly
stable block that excludes the rotation you were looking for is worse than a
noisier one that contains it, and the gap rule cannot see the difference.

**Direction does not matter.** Comparing old-`P`-in-new-`Q` against
new-`P`-in-old-`Q` gives 1.65 vs 1.63, 2.42 vs 2.45, 2.31 vs 2.26, 3.10 vs 3.06
degrees. Under 2%, no consistent sign. The asymmetry in `subspace_distance` is
real but immaterial for detection, so the paper's convention is kept for
interpretability at no cost.

### Reproduction notes and limits

Every figure above comes from a fixed seed and is reproduced by the test suite;
the basis is always drawn from `default_rng(20260801)` so the same Haar frame is
shared across all three regimes.

Two limits worth stating plainly:

- **The block-comparison `theta_min` figures above are approximate and somewhat
  optimistic.** They are computed
  from the null distribution via an additivity shortcut — the injection needed
  is the 95th minus the 20th percentile of the null — rather than by injecting
  at each angle. Direct spot checks give 74% power where the table claims 80%
  at `P=1, 1.76 deg`, and 72% at `P=5, 3.49 deg`. Injection widens the
  distribution as well as shifting it, and the shortcut ignores the widening.
  Ordering, scaling and the blindness structure are unaffected. Any threshold
  that leaves this project should be measured directly.
- **`P=3` is a defensible compromise on this spectrum and nothing more.** It
  covers the three strongest factors at 2.58 degrees against 1.76 for a `P=1`
  that watches only the market mode. The number will not transfer to the Nikkei.
  What transfers is the procedure: locate the factors, choose `P` to cover the
  ones you care about, and pay the threshold.

## Note on the `D_RMT` normalisation

`d_random_subspaces` takes a `convention` argument.

- `"paper"` reproduces the unnumbered `D_RMT` display in §2 of arXiv:1203.6228
  and the 0.83 quoted in its Fig. 2 caption. That paper only quotes the result;
  it originates in its ref [6] (Bouchaud, Laloux, Miceli & Potters, EPJB 55
  (2007) 201). This is *not* the eigenvalue variogram of §4 — that one is the
  eigenvalue variogram used in 1.2. The underlying density integrates to `P/Q`,
  not 1.
- `"normalised"` divides by `alpha*pi` instead of `beta*pi`, integrates to
  exactly 1.0, and is therefore the genuine mean of `-ln sigma` over the `P`
  singular values — which is what `subspace_distance` computes.

They differ by exactly `P/Q`. Use `"normalised"` when comparing against your own
measured `D`. Both are unit-tested.

## Regime 4 status — real-panel gates

Regimes 4.1–4.3 use the frozen 2000–2010 panels to reproduce the published
measurement. Regimes 4.4–4.7 use separate fixed-universe panels through the
latest 2026 observation to decide whether Stage 2 has a label worth learning.

### 4.1–4.3 — paper replication

**What is tested.** Whether the eigenvalue variogram, subspace variogram and
U-shaped measurement-error curve can be recovered before making any forecasting
claim.

**Setup.** Four paper-matched universes, frozen at 2000–2010; correlation
matrices; raw and daily-standardised returns; the paper's `T=N` convention for
4.1–4.2 and a sweep over `T` for 4.3. The overlap experiments use `P=5,Q=10`
when matching Fig. 9.

**Verdict.** **YAY for the replication shape, INCONCLUSIVE for a precise
universal `T*`.** The rolling-window overlap artifact is reproduced, S&P keeps
accumulating excess subspace displacement after that artifact saturates, and all
four panels show the expected U-shape. Fewer than three independent window pairs
support several minima, so their exact locations are not precise estimates.

### 4.4 — tangent persistence

**What is tested.** At the current leading space `Y_t`, compare the incoming
Grassmann tangent `H_t^-` with the outgoing tangent `H_t^+`. A cosine of `+1`
means continuation, `0` means no directional relation and `-1` means reversal.
The matched shuffle null is essential because overlapping rolling windows create
some apparent persistence even when long calendar order is destroyed.

**Setup.** Standardised full-history panels; `P=3`; 42-day horizon; 14-day step;
`T=357` for S&P and `T=250` elsewhere. The null permutes intact 21-day return
blocks and rebuilds the complete rolling-eigenspace history.

**Verdict.** **YAY for direction; NAY for repeating the full previous step.**

| panel | observed cosine | shuffled cosine | windows with positive cosine | directional `p` | full-step loss versus holding still |
|---|---:|---:|---:|---:|---:|
| S&P full | **0.2019** | 0.0506 | 79.8% | 0.0476, exploratory | **47.5% worse** |
| Nikkei full | **0.1158** | 0.0198 | 72.2% | 0.010 | **67.1% worse** |
| DAX full | **0.0973** | 0.0237 | 63.6% | 0.010 | **56.2% worse** |
| CAC full, cleaned | **0.0830** | 0.0217 | 61.6% | 0.020 | **59.9% worse** |

The positive-cosine percentage is the fraction of eligible rolling-window
triples whose outgoing arrow lies in the same tangent-space half as the incoming
arrow. It is descriptive, not the significance test. The `p`-value compares the
*mean cosine* with complete shuffled histories. The final percentage is the
relative increase in mean containment loss from applying the whole previous
tangent instead of forecasting no rotation. The arrow points usefully but is
systematically too long; Stage 2 would need to learn its damping.

### 4.5 — cross-asset coherence

**What is tested.** Whether the tangent motion contains a synchronised
cross-asset component, rather than merely aggregating company loading changes
that happen at unrelated times.

**Setup.** Procrustes-align the rolling bases, form each asset's tangent-increment
history and measure the leading share of the cross-asset covariance. The null
independently shifts each asset's history, then projects it back into the valid
tangent space and restores its observed speed. Each panel uses 999 nulls.

**Verdict.** **YAY on all four panels.**

| panel | observed common share | desynchronised null | `p` | leading-vector participation |
|---|---:|---:|---:|---:|
| S&P full | **24.36%** | 6.07% | 0.001 | about 125 / 357 names |
| Nikkei full | **13.71%** | 5.30% | 0.001 | about 41 / 131 |
| DAX full | **17.56%** | 8.63% | 0.001 | about 9 / 29 |
| CAC full, cleaned | **14.03%** | 10.10% | 0.021 | about 6 / 23 |

Here the first percentage is the fraction of all tangent-increment variation
captured by its strongest common cross-asset pattern. The null percentage asks
how large that share would be if each asset kept the same individual history but
lost synchrony with every other asset. Participation is an effective breadth,
not a literal list of selected companies.

### 4.6 — is the directional signal just ERSE rearranged?

**What is tested.** Apply Liu & Liu's actual pairwise eigenvector-rotation
algorithm to every current-window correlation matrix. At the current leading
space $Y_t$, define the ERSE direction
$E_t=\operatorname{Log}_{Y_t}(Y_t^{\mathrm{ERSE}})$. Then ask four separate
questions: does $E_t$ point toward the realised outgoing tangent; how much
outgoing tangent energy lies along $E_t$; does incoming/outgoing persistence
survive after both tangents are projected off $E_t$; and how much of the next
covariance transition crosses the current top-$P$/complement boundary, which an
eigenvalue-only update in the current basis cannot create.

**Setup.** The Regime 4.4 full-history specification is unchanged: standardised
returns, $P=3$, 42-day horizon, 14-day step, $T=357$ for S&P and $T=250$
elsewhere. ERSE uses Liu & Liu's primary deviation floor $\delta=0.25$; observed
sensitivity runs use $0.15$ and $0.35$. The matched null permutes intact 21-day
return blocks and rebuilds the rolling covariance, ERSE correction and tangent
series. DAX, CAC and Nikkei use 99 null histories; S&P uses 20 and remains
exploratory. The paper assumes an all-positive correlation matrix, so each run
also records how often that strict assumption holds rather than silently
exporting the theorem to these individual-stock panels.

**Verdict.** **YAY: the Regime 4.4 signal is distinct from ERSE. NAY: ERSE is
not a useful future-subspace forecast here. NAY for evidence that the observed
top/complement covariance share exceeds its matched null.** The last NAY limits
the mechanism claim; it does not undo the direct attribution result.

| panel | original cosine (null) | ERSE/outgoing cosine | outgoing energy attributed to ERSE | residual cosine (null) | top/complement covariance share (null) | ERSE forecast skill vs holding still | strict all-positive windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| S&P full | **0.2019** (0.0506) | −0.0150 | **0.05%** | **0.2021** (0.0509) | 40.6% (39.9%), $p=0.143$ | **−317.5%** | 0.0% |
| Nikkei full | **0.1158** (0.0198) | −0.0141 | **0.06%** | **0.1161** (0.0201) | 39.7% (39.4%), $p=0.190$ | **−158.1%** | 3.6% |
| DAX full | **0.0973** (0.0237) | −0.0119 | **0.92%** | **0.1009** (0.0237) | 38.3% (39.3%), $p=0.960$ | **−128.7%** | 16.0% |
| CAC full, cleaned | **0.0830** (0.0217) | −0.0072 | **0.22%** | **0.0830** (0.0220) | 37.8% (39.1%), $p=0.980$ | **−143.2%** | 36.6% |

The first and fourth columns are directly comparable: after removing the ERSE
direction, the directional persistence is not weakened. Its matched upper-tail
$p$-values are 0.0476 exploratory for S&P, 0.010 for Nikkei and DAX, and 0.020
for CAC. The ERSE/outgoing cosines are negative on every panel, so ERSE points
slightly away from the realised future move; a small $p$ attached to “less
negative than shuffled” on Nikkei or S&P does not change that required-sign
failure. “Attributed energy” is the squared-length share removed by the
one-direction projection, not a percentage of windows. The result is robust
across $\delta=0.15,0.25,0.35$: attributed energy stays between 0.0% and 1.2%
and the residual cosine remains effectively the original cosine.

Reproduce one panel with, for example:

```bash
python scripts/regime4_6_erse.py --label nikkei_full --T 250 --P 3 --step 14 --horizon 42 --mode standardised --delta 0.25 --shuffles 99
```

### 4.7 — does the signal survive in the complete partial flag?

**What is tested.** Whether the market direction, established top-three core
and six-dimensional collision buffer can be carried together without losing
the persistence, cross-asset coherence or ERSE-distinctness established in
Regimes 4.4–4.6. One time snapshot is

$$\mathcal F_t=(Y_t^{(1)}\subset Y_t^{(3)}\subset Y_t^{(6)})
\in\mathrm{Flag}(N;1,3,6).$$

This is one flag at time $t$, not a container holding every time point. The
history is the sequence $\{\mathcal F_t\}_t$. Each frame is obtained from one
top-six eigendecomposition, so nesting is exact; the three Grassmann logarithms
are then computed separately and retained as a tuple. The flag inner product
weights level $d$ by $1/d$, preventing the six-space from dominating merely
because it has more columns. This is a basis-invariant nested-projector
embedding, not a claim to have implemented the intrinsic quotient-manifold
flag logarithm.

**Setup.** The same standardised full-history panels, 42-day horizon, 14-day
step, $T=357$ for S&P and $T=250$ elsewhere. Persistence uses the same 21-day
calendar-block permutation null; coherence uses 999 independently shifted
asset histories; ERSE uses $\delta=0.25$ with $0.15$ and $0.35$ sensitivity
runs. The predeclared family comprises the market level, top-six level and
complete nested flag, with Holm adjustment. Top three is the already-confirmed
anchor; the disjoint $2{:}3$ and $4{:}6$ blocks diagnose where a result comes
from rather than creating extra gates.

**Verdict.** **YAY: the complete flag is a valid Stage 2 representation.**
Nikkei, DAX and CAC clear both the persistence and coherence gates after Holm
adjustment. S&P has the largest practical separation from its null but is
**INCONCLUSIVE confirmatorily** because 20 calendar nulls make its smallest raw
$p=1/21$ and its smallest three-test Holm value $0.143$. Its coherence result
is nevertheless decisive. ERSE explains little flag-tangent energy, and the
residual persistence remains.

| panel | flag cosine (calendar null) | raw / Holm $p$ | coherent share (shift null) | raw / Holm $p$ | ERSE-attributed energy | residual cosine |
|---|---:|---:|---:|---:|---:|---:|
| S&P full | **0.1545** (0.0313) | 0.0476 / 0.143, exploratory | **13.05%** (2.54%) | 0.001 / 0.003 | 0.11% | **0.1552** |
| Nikkei full | **0.0706** (0.0196) | 0.010 / 0.030 | **6.95%** (2.03%) | 0.001 / 0.003 | 0.25% | **0.0721** |
| DAX full | **0.0561** (0.0217) | 0.020 / 0.040 | **10.98%** (6.51%) | 0.002 / 0.003 | 1.11% | **0.0608** |
| CAC full, cleaned | **0.0622** (0.0273) | 0.020 / 0.040 | **10.36%** (7.42%) | 0.002 / 0.003 | 1.32% | **0.0630** |

The flag cosine asks whether consecutive complete nested motions point in a
similar direction. The coherent-share percentage is the fraction of variation
captured by the strongest synchronised asset pattern, not the fraction of
windows. ERSE-attributed energy is the squared tangent-length fraction removed
by projecting off the ERSE direction. Sensitivity over
$\delta\in\{0.15,0.25,0.35\}$ leaves the residual flag cosine positive and
essentially unchanged.

The component diagnostics qualify the simple YAY. Nikkei validates every
level. DAX's raw top-six persistence is a NAY ($p=0.08$), although its residual
passes after removing ERSE; CAC's top-six result is borderline ($p=0.05$) and
its residual is just outside ($p=0.06$). Thus $Y^{(6)}$ is justified as a
learnability/collision buffer inside the flag, but it should not be described
as equally forecastable on every panel.

Reproduce one confirmatory panel with, for example:

```bash
python scripts/regime4_7_flag.py --label nikkei_full --T 250 --step 14 --horizon 42 --mode standardised --delta 0.25 --calendar-shuffles 99 --coherence-shuffles 999
```

### Stage 2 entry decision

**The representation gate is cleared.** Begin with chronological held-out
baselines on the complete flag, scoring its market, top-three and top-six levels
separately as well as jointly. The sub-universe coherence scaling test remains
required before making a literal claim about how coherence scales with $N$; it
does not need to delay model fitting. Regimes 4.6–4.7 establish a nonredundant
target, not out-of-sample ML value—the latter is precisely Stage 2's job.
