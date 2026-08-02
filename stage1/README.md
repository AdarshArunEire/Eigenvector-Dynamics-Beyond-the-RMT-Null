# Stage 1 — Instrument

Correctness, not science. Nothing in this stage is a finding.

**Entry condition:** Stage 0 exit gate met (spectral theorem usable without
reference; literature checks 2 and 3 written up).

**Exit gate:** regimes 1-3 pass; power curve recorded; `D_emp > D_th`
reproduced on real equity data with the qualitative `T* ~ 2yr` peak present.

**Status:** regimes 1, 2 and 3 run; power curve recorded. Only the real-data
condition remains.

Regime 2 leaves two items open rather than closed: the estimated-spectrum
substitution error has no portable law and no working correction (2.2), and
within-window level drift inflates `D` with no rotation present (2.3). Both are
`T`-dependent, and they pull in opposite directions along the axis `T*` is read
off.

Regime 3 closes the assumption the other two rested on — `excess = D_emp - D_th`
is a legitimate subtraction — and settles `Q` deterministically via the
Marchenko-Pastur edge. Its second bet fails honestly: adaptive-gap block
selection loses to a fixed `P`, because it optimises stability while the
instrument needs coverage.

**Failure route:** none. If this fails, the code is wrong. Fix it.

## Notation

| symbol | meaning | more is |
|--------|---------|---------|
| `N` | number of assets; `C` is `N x N` | **worse** — `C` has `N(N+1)/2` parameters but the data supplies only `N x T` numbers |
| `T` | observations (days) in one window | better |
| `q = N/T` | aspect ratio; parameters per observation. `q -> 0` data-rich, `q = 1` as many assets as days | lower is better |
| `P` | size of the inner eigenvector block being tracked | — |
| `Q` | size of the outer block it is compared against, `Q >= P` | — |
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
| 2.1 | Regime 2 — fixed eigenvectors, time-varying eigenvalues | `D_num ≈ D_th` | ratio 1.0015 at `sigma=0.06`; pooled 1.005 ± 0.003 over 4000 trials | **Pass** |
| 2.2 | Estimated spectrum substituted for the true one | Bias small and shrinking in `T` | Under 1.5% at `N/T < 0.3` for `N=40`, but −11.5% by `N/T=0.89`, and sign-flips with `N`. No portable law; bootstrap correction fails. | **Open — measure in situ at regime 4's settings** |
| 2.3 | Common eigenvalue drift *within* a window | Eq (10) fails; `D` inflated | Inflated by `1 + h²/3`, matched to 2% at `h=0.6, 0.8` | **Pass (breaks as predicted)** |
| 3.1 | Regime 3 — injected drift | Recovers injected magnitude | `D_emp = D_th + D_inject` to 2% over a 300x range of theta, 0.1% once clear of the floor | **Pass** |
| 3.2 | Label granularity | Adaptive-gap blocks beat fixed and per-mode on recovery | `Q` solved by the MP edge (99.6% correct on samples). `P`: no scheme dominates, and adaptive-gap picks `P=1`, which is blind to two of three injection sites | **Fail — adaptive answers the wrong question** |
| 4.1 | Nikkei reproduction | `D_emp > D_th`, ratio peaks near τ ≈ 500d | not run | — |

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
0.3, and regime 4 does not run there.** Extending the sweep towards the ratio
regime 4 actually uses:

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

**Open, carried into stage 2:** an `h` of only 0.35 buys a 4% inflation, which
is the same order as the ratios regime 4 will be trying to interpret. Either the
within-window level drift has to be measured and divided out, or the excess has
to be shown to survive a variance-standardised return series. Not resolved here.

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

So `theta_min = 4.1` degrees in 3.1 and `2.58` degrees at `P=3` in 3.2 are
both correct and describe different experiments.

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

### Power curve

Not a hypothesis and so not a ledger row — a recorded characterisation, and the
exit gate's third condition. Detection threshold on a **single** pair of
windows, threshold set at the 95th percentile of the `theta=0` distribution
(5% false positive), read at 80% power:

| `T` | `theta_min` (rad) | `theta_min` (deg) |
|---|---|---|
| 250 | 0.240 | 13.8° |
| 500 | 0.149 | 8.5° |
| 1000 | 0.103 | 5.9° |
| 2000 | 0.072 | 4.1° |

`theta_min ~ 3.2/sqrt(T)`, confirmed rather than assumed: `theta_min * sqrt(T)`
comes out 3.80, 3.34, 3.27, 3.21 across those rows, converging, with `T=250`
off in the same direction everything else at low `T` has been. The scaling is
what the algebra predicts — `D_inject ~ theta^2/2P` against a floor `~ 1/T`.

At two years of daily data the instrument resolves a rotation of roughly 8
degrees. In portfolio terms: a position built orthogonal to the top directions
two years ago now carries a `sin(8 deg) ~ 14%` component along them, and that
is the smallest such exposure that could be demonstrated rather than asserted.

Note this is single-window-pair detection. A real analysis averages over many
pairs at fixed lag, which lowers the floor by roughly `sqrt(n)` — so this table
is conservative for the paper's setting, not optimistic.

### 3.2 — `Q` is solved, `P` is not, and cannot be

**`Q` is deterministic.** The Marchenko-Pastur upper edge
`lam_+ = sigma^2 (1 + sqrt(N/T))^2` is the largest eigenvalue pure noise
produces; anything above it is inconsistent with noise. `sigma^2` is
re-estimated from the sub-edge bulk and iterated to a fixed point, since the
factors inflate any naive average. Implemented as `q_from_mp_edge`.

On a spectrum of six factors over a flat bulk at 1.0, it recovers
`sigma^2 = 1.0000` and `Q = 6`, and picks `Q = 6` from *sample* spectra in
99.6% of trials. One free parameter eliminated, using only the observed
spectrum and `T`.

This is Marchenko & Pastur (1967) by way of Laloux, Cizeau, Bouchaud & Potters,
PRL 83 (1999) 1467 — which is ref [8] of arXiv:1108.4258, so the rule comes
from the instrument's own lineage rather than from outside it.

**It carries an assumption that must be checked, not assumed: the bulk has to
actually be noise.** Applied to regime 1's spectrum, whose bulk is a `linspace`
from 1.3 to 0.4, the same procedure returns `Q = 27`. That is not a failure of
the rule. That bulk is genuine spread structure, and MP is correct to refuse to
call it noise. The large `Q` is the criterion reporting that it does not apply,
and it is unit-tested as such. Before trusting the edge on real data, confirm
the bulk looks Marchenko-Pastur.

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

- **The `theta_min` figures are optimistic by roughly 5–10%.** They are computed
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

- `"paper"` reproduces the unnumbered `D_RMT` display on p.2 of arXiv:1108.4258
  and the 0.83 quoted in its Fig. 2 caption. That paper only quotes the result;
  it originates in its ref [6] (Bouchaud, Laloux, Miceli & Potters, EPJB 55
  (2007) 201). This is *not* Eq (9) of arXiv:1108.4258 — Eq (9) there is the
  eigenvalue variogram used in 1.2. The underlying density integrates to `P/Q`,
  not 1.
- `"normalised"` divides by `alpha*pi` instead of `beta*pi`, integrates to
  exactly 1.0, and is therefore the genuine mean of `-ln sigma` over the `P`
  singular values — which is what `subspace_distance` computes.

They differ by exactly `P/Q`. Use `"normalised"` when comparing against your own
measured `D`. Both are unit-tested.
