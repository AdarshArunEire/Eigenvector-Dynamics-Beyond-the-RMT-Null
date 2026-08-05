# Stage 2 — Forecast

> **Closed, 2026-08-05.** The geometric-loss ladder documented below was
> superseded by the respecified realised-variance capture score, and the
> forecasting question now has a documented negative answer. Model 4.1 — a
> five-parameter correction confined to the only block a subspace metric can
> see, fitted four different ways — does not beat the EWMA it contains on any
> of four markets, and is significantly worse on DAX. All four losses score
> within $4\times10^{-4}$ capture of one another: the objective is flat.
> See the closing section of [`../BUILDNOTES.md`](../BUILDNOTES.md) and the
> result tables in [`../README.md`](../README.md). Everything below this banner
> is the historical record of how the ladder was built and why it was
> respecified; the frozen benchmark registry remains reproducible.

**Model fitting started, but Regime 4.9 forced a structural correction before
Model 3.2.** The entry-gate experiments exist as
`scripts/regime4_4_tangent.py`, `scripts/regime4_5_coherence.py` and
`scripts/regime4_6_erse.py`, with the representation gate in
`scripts/regime4_7_flag.py` and return-level robustness controls in
`scripts/regime4_8_robustness.py` and the deterministic-deletion gate in
`scripts/regime4_9_deletion_attribution.py`. Regime 4.9 finds a confirmatory
incoming-block signal on S&P and an equal-market aggregate signal, but not an
independent four-market result. The state remains
$\mathrm{Flag}(N;1,3,6)$; its causal base is now the Flag of the observations
known to remain in the target window.

Stage 1 builds an instrument that *measures* how far the leading subspace has
rotated between two windows, against a null that says how far it would appear to
rotate if nothing had moved at all. Stage 2 is the attempt to *predict* that
rotation, and to turn the prediction into a covariance estimator that beats a
rotation-invariant one out of sample.

The real-data regimes remain Stage 1 because they establish that the measurement
is trustworthy. The existing chronological benchmarks, oracle line and Global
Damping fit remain reproducible Stage 2 diagnostics, but their Frozen-relative
contest is superseded for model fitting by the Retained-relative residual target.

## Reporting contract

Every gate is reported in the same order: **what is tested; setup; verdict.**
The verdict words have fixed meanings:

- **YAY:** the primary standardised statistic beats its matched null at the 5%
  level and the effect has the sign required by the claim.
- **NAY:** the primary statistic does not clear that gate, or has the wrong
  sign.
- **INCONCLUSIVE:** the data or null resolution cannot decide. This is not
  silently converted into a negative result.

## Stage 2 numbering

Stage 1 used regimes to identify and challenge the phenomenon. Stage 2
restarts its internal numbering and distinguishes fixed rules from fitted
rules:

- **Benchmark family 1:** geometric forecasts, from Frozen Flag and Constant
  Velocity through ERSE, HCAL and BAHC filtered eigenspaces.
- **Benchmark family 2:** complete covariance estimators: sample covariance,
  fixed and tuned EWMA, Ledoit–Wolf, OAS, RIE/nonlinear shrinkage, QuEST,
  cross-validated eigenvalue shrinkage, HCAL and BAHC.
- **Model family 3:** learned global and layerwise tangent damping.
- **Model family 4:** transported tangent autoregression.
- **Model family 5:** richer sequence and full-SPD challengers, attempted only
  after the smaller families earn their place out of sample.

The frozen pre-4.9 rows share chronological examples, target Flags and losses.
Corrected geometric models additionally share one retained-observation base;
only their forecast of the incoming-block tangent may differ.

The complete predeclared list, primary-versus-secondary status, covariance
losses and literature provenance are frozen in
[`stage2/BENCHMARKS.md`](BENCHMARKS.md). Beating a geometric benchmark alone is
not sufficient: after covariance reconstruction, a learned rotation must face
the primary Family 2 estimators on realised future returns.

## Entry condition

**A label.** Concretely: an excess-rotation series that survives all six of
the following, on at least one panel.

1. **Persistence.** The direction of the subspace's motion, not just its
   magnitude, is autocorrelated from one window to the next. A diffusion has a
   growing displacement and no forecastable direction; only the second one can
   be learned. Regime 4.2 shows the growth. Regime 4.4 now measures the incoming
   and outgoing velocities in the same Grassmann tangent space and compares the
   observed alignment with a matched block-permuted-return null.
2. **Not ERSE rearranged.** ERSE applies pairwise Givens rotations and then
   recomputes Rayleigh eigenvalues. For each rotated pair those two values move
   toward one another while preserving their sum; that exact pairwise identity
   does not make the full corrected estimator “only shrinkage.” Regime 4.6
   therefore constructs the actual ERSE direction and attributes the observed
   temporal tangent to it rather than relying on analogy. See `PRIOR_ART.md` on
   arXiv:2507.01545.
3. **Evolution, not mis-specification.** A static but wrongly-specified `C`
   produces an excess that is flat in the lag `tau`; genuine evolution produces
   one that grows with it. Regime 4.2 reports this as `flat frac`.
4. **Coherence.** The surviving tangent motion must contain a cross-sectionally
   common component, rather than being the aggregate effect of many independent
   time-varying betas. That is Regime 4.5. It is an interpretation gate: without
   it a forecast may still work, but it belongs to the established
   time-varying-loading literature rather than supporting the project's claim
   of coherent eigenbasis motion.
5. **Representation.** The signal must survive when the market direction,
   top-three core and six-dimensional collision buffer are carried together as
   a basis-invariant partial flag. Regime 4.7 now passes this gate on Nikkei,
   DAX and CAC, with S&P strongly sign-consistent but inferentially exploratory.
6. **Beyond deterministic deletion.** Every forecast must start from the Flag
   of the $T-42$ observations known to remain. Regime 4.9 tests persistence of
   the realised incoming-block tangent against both calendar and
   volatility-matched return nulls. S&P passes after Holm correction; Nikkei is
   raw-borderline, and DAX/CAC fail. The equal-market aggregate passes.

Until a series clears the forecast gates and its interpretation is settled by
the coherence gate, fitting a large model would risk learning the null. Regime
4.9 supersedes the earlier two-gate outcome map: full-window persistence without
incoming-block persistence is rolling-composition structure, not a Stage 2
label. The corrected result licenses a small S&P model and an explicitly pooled
equal-market experiment; it does not license independent DAX or CAC claims.
Regime 4.7 still establishes that the chosen nested representation is valid.

## Regime 4.4 — does the previous direction help?

**What is tested.** Does the leading eigenspace tend to continue rotating in
the same direction? At the current space $Y_t$, the incoming and outgoing
motions are

$$H_t^-=-\mathrm{Log}_{Y_t}(Y_{t-h}),\qquad
H_t^+=\mathrm{Log}_{Y_t}(Y_{t+h}).$$

Their cosine has an ordinary interpretation: $+1$ means exactly the same
direction, $0$ means no directional information, and $-1$ means reversal.
Positive cosine by itself is not a pass because overlapping rolling windows
manufacture some positive persistence.

**Setup.** Fixed-universe daily panels from 2000 through the latest available
2026 observation; standardised returns; $P=3$; 42-day forecast horizon; window
$T=357$ for the 357-name S&P panel and $T=250$ elsewhere; step 14 days. The null
permutes intact 21-day return blocks and then rebuilds every rolling eigenspace,
preserving fat tails, daily cross-sections and the window-overlap artifact while
destroying longer calendar order. DAX, CAC and Nikkei use 99 nulls. The S&P run
currently uses 20 and is explicitly exploratory; its minimum possible corrected
$p$ is $1/21=0.0476$.

**Verdict: YAY for directional information on all four panels. NAY for copying
the full previous step.**

| panel | names / days | observed cosine | shuffled cosine | continuing fraction | $p$ | full-step forecast versus holding still |
|---|---:|---:|---:|---:|---:|---:|
| S&P full | 357 / 6,678 | **0.2019** | 0.0506 | 79.8% | 0.0476 exploratory | **47.5% worse** |
| Nikkei full | 131 / 6,629 | **0.1158** | 0.0198 | 72.2% | 0.010 | **67.1% worse** |
| DAX full | 29 / 6,788 | **0.0973** | 0.0237 | 63.6% | 0.010 | **56.2% worse** |
| CAC full, cleaned | 23 / 6,828 | **0.0830** | 0.0217 | 61.6% | 0.020 | **59.9% worse** |

In plain language: the previous arrow points somewhat usefully, but it is far
too long. The result licenses a model that learns **how much to damp the arrow**.
It rejects the naive rule “repeat the whole last rotation.” The 2000–2010 runs
are retained as replication-era pilots and are not used for this verdict.

These are fixed-universe panels conditioned on near-continuous coverage, so
extending the time span does not remove survivorship bias. It supplies more
history for the same-dimensional state; it does not reconstruct historical
index membership. CAC additionally excludes Atos and Vivendi because Yahoo's
adjusted prices do not incorporate Atos's 2024 restructuring and Vivendi's 2024
three-company distribution as total returns, plus Accor because its unexplained
2004 jump could not be validated. All three exclusions are explicit in
`cac40_full_drops.csv`, not silently deleted observations.

## Regime 4.5 — is that movement genuinely common?

**What is tested.** Is the tangent motion shared across a meaningful fraction
of assets, or is it merely the sum of independently timed beta wiggles?

**Setup.** Sequentially Procrustes-align the rolling bases and form the
asset-by-(time × factor) tangent-increment matrix. Measure the leading share of
its cross-asset covariance. The null independently shifts every asset's
increment history, projects the surrogate back into the correct tangent space,
and restores its observed speed. There are 999 null repetitions. The leading
eigenvector's participation count checks that one or two damaged names are not
creating the spike.

**Verdict: YAY on all four panels.**

| panel | observed leading share | desynchronised null | $p$ | names materially participating | verdict |
|---|---:|---:|---:|---:|---|
| S&P full | **24.36%** | 6.07% | 0.001 | about 125 / 357 | **YAY** |
| Nikkei full | **13.71%** | 5.30% | 0.001 | about 41 / 131 | **YAY** |
| DAX full | **17.56%** | 8.63% | 0.001 | about 9 / 29 | **YAY** |
| CAC full, cleaned | **14.03%** | 10.10% | 0.021 | about 6 / 23 | **YAY, weakest panel** |

The coherence threat survives its first direct test: the detected motion is
broader and more synchronous than independently timed loading changes. A
sub-universe scaling sweep is still required before translating this into a
literal empirical $O(N^2)$ claim.

## Regime 4.6 — does ERSE explain the tangent signal?

**What is tested.** At every current space $Y_t$, compute the leading space of
Liu & Liu's ERSE correction and its Grassmann direction
$E_t=\mathrm{Log}_{Y_t}(Y_t^{\mathrm{ERSE}})$. Measure alignment with the
realised outgoing tangent, the fraction of outgoing tangent energy attributable
to $E_t$, and the incoming/outgoing persistence left after projecting both
tangents off $E_t$. Separately measure the share of the covariance transition
that crosses the current top-$P$/complement boundary.

**Setup.** The same standardised full-history panels, $P=3$, 42-day horizon,
14-day step and 21-day block-permutation null as Regime 4.4. ERSE uses its
published primary threshold $\delta=0.25$, with $0.15$ and $0.35$ sensitivity
runs. DAX, CAC and Nikkei use 99 null histories; S&P uses 20 and is exploratory.

**Verdict: YAY that the learnable direction is not ERSE rearranged. NAY for
ERSE itself as a forecast. NAY for excess top/complement covariance-transition
share under this test.**

| panel | ERSE-attributed outgoing energy | original cosine | residual cosine | residual $p$ | ERSE skill vs holding still |
|---|---:|---:|---:|---:|---:|
| S&P full | 0.05% | 0.2019 | **0.2021** | 0.0476 exploratory | −317.5% |
| Nikkei full | 0.06% | 0.1158 | **0.1161** | 0.010 | −158.1% |
| DAX full | 0.92% | 0.0973 | **0.1009** | 0.010 | −128.7% |
| CAC full, cleaned | 0.22% | 0.0830 | **0.0830** | 0.020 | −143.2% |

The direct answer is “not the same paper in a new form”: removing ERSE removes
at most 0.92% of outgoing tangent energy at the primary threshold and does not
reduce persistence. However, the top/complement covariance-transition shares
of 37.8–40.6% do not exceed their matched nulls. That secondary result prevents
claiming that this experiment has already established incremental covariance
risk value. Such value must be demonstrated chronologically in Stage 2.

## Regime 4.7 — is the partial flag a valid learning target?

**What is tested.** Does the signal established at $P=3$ remain present when
the state preserves the market mode, core and collision buffer simultaneously?
At each time,

$$\mathcal F_t=(Y_t^{(1)}\subset Y_t^{(3)}\subset Y_t^{(6)})
\in\mathrm{Flag}(N;1,3,6).$$

The three spaces come from one top-six eigendecomposition. Their Grassmann
logs are computed separately and combined as an inverse-dimension-weighted
tuple, so the representation is invariant to signs and within-block basis
rotations. This nested-projector construction is the empirical geometry used
for the gate; it is not the intrinsic quotient-manifold flag logarithm.

**Setup.** The Regime 4.4–4.6 full-history specification is unchanged. The
complete flag and its market, disjoint-block and cumulative components face the
same calendar-block persistence null, 999-shift coherence null and ERSE
projection. The predeclared Holm family is market, top six and complete flag;
top three is the established anchor and the disjoint blocks are diagnostic.

**Verdict: YAY for the master representation.** Complete-flag persistence
clears Holm correction on Nikkei ($0.0706$ versus null $0.0196$), DAX
($0.0561$ versus $0.0217$) and CAC ($0.0622$ versus $0.0273$). Complete-flag
coherence clears on all four panels at Holm $p=0.003$. S&P persistence is
practically strongest ($0.1545$ versus $0.0313$) but remains exploratory because
20 calendar nulls imply Holm $p=0.143$. ERSE accounts for only 0.11–1.32% of
complete-flag outgoing energy at $\delta=0.25$, and residual persistence is not
reduced.

The outer $Y^{(6)}$ component deserves a qualification: it is clean on Nikkei,
raw-NAY on DAX and borderline on CAC. Therefore Stage 2 should retain it as a
containing/collision buffer and report component losses. It should not assume
that all six directions are equally forecastable.

## Regime 4.8 — what survives return-level controls?

**What is tested.** First, whether independent univariate return histories plus
the overlapping-window pipeline can reproduce complete-flag persistence;
second, whether persistence and coherence survive rolling removal of the
internal equal-weight market factor.

**Setup.** The phase control independently applies IAAFT to every asset,
preserving its exact marginal and approximate linear spectrum while destroying
cross-company timing. This is a harsh negative control, not a replacement for
the matched intact-block calendar null. The market ablation instead estimates
an equal-weight factor and rolling OLS betas inside every window, rebuilds the
residual Flag, and repeats that complete operation inside 99 calendar nulls and
999 coherence nulls.

**Verdict.** S&P and Nikkei clear the complete-flag phase control at $p=0.010$;
DAX and CAC finish just outside at $p=0.070$, although their market and
top-three components pass. More importantly for mechanism, the residual complete Flag
passes both persistence and coherence on every market: S&P
($p=0.010/0.001$), Nikkei ($0.010/0.001$), DAX ($0.010/0.006$), and CAC
($0.020/0.017$). Thus Stage 2 is not being licensed merely by a moving market mode, but
the weak DAX/CAC outer-six diagnostics remain a reason to score all Flag levels
separately.

The effect sizes matter alongside those $p$-values. Relative to the original
Flag, the residual complete-flag cosine changes by +0.4% on S&P, +11.3% on
Nikkei, +17.1% on DAX and +2.4% on CAC; coherent share changes by +19.8%,
+2.8%, +4.0% and −13.9%. Individual layers move much more: S&P loses 39% of
its top-six cosine while its residual market layer rises by 50%, CAC loses 66%
of its top-six cosine, DAX loses 30% of its market cosine, and Nikkei loses
approximately 21% from both market and top-six. The complete-flag survival is
therefore not invariance to the ablation; it is persistence being redistributed
across the nested levels.

## Regime 4.9 — the corrected forecast target

Regime 4.9 makes the 42 observations known to leave each rolling window part of
the forecast primitive rather than something a model can earn credit for. Let
$\mathcal B_t$ be the retained-observation Flag. The target is
$A_t^+=\operatorname{Log}_{\mathcal B_t}(\mathcal F_{t+42})$, and its zero
forecast is Retained Window. The preceding realised incoming-block tangent is
carried to $\mathcal B_t$ by ordered orthogonal Flag transport before the two
directions are compared.

Deletion accounts for a mean 39–45% of per-origin outgoing complete-Flag
tangent energy. The old
cosine after deletion projection is 0.0338 on S&P, 0.0013 on DAX and negative
on Nikkei and CAC. The correctly based incoming-block cosines are 0.0977,
0.0358, 0.0200 and 0.0169 respectively.

| panel | incoming-block cosine | calendar null / $p$ | volatility-matched null / $p$ | four-panel Holm verdict |
|---|---:|---:|---:|---|
| S&P 500 | **0.0977** | 0.0107 / 0.01 | 0.0564 / 0.01 | **YAY, 0.04 / 0.04** |
| Nikkei | **0.0358** | −0.0002 / 0.01 | 0.0190 / 0.05 | **NAY, 0.04 / 0.15** |
| DAX | 0.0200 | −0.0016 / 0.07 | 0.0147 / 0.35 | **NAY** |
| CAC 40 | 0.0169 | 0.0006 / 0.09 | 0.0176 / 0.52 | **NAY** |

The equal-market cosine is 0.0426 and passes the two aggregate nulls at
$p=0.01$ and $p=0.03$. The verdict is therefore **PARTIAL YAY**: a residual
signal exists, led by S&P, but the earlier universal four-market interpretation
is withdrawn. All subsequent Family 1 models must predict from the retained
base and report skill versus Retained Window.

## Benchmark 1.1 — Frozen Flag

**What is tested.** Establish the loss incurred by pretending that covariance
geometry does not rotate over the next 42 trading days. The forecast is

$$\widehat{\mathcal F}_{t+42}=\mathcal F_t,$$

or equivalently $\widehat H_t^+=0$. This benchmark fits no parameter. It is the
reference denominator that every later benchmark and model must beat.

**Setup.** Build the same standardised rolling
$\mathrm{Flag}(N;1,3,6)$ histories as Stage 1, retaining identical
past/current/future triples for every later model. Splits are assigned by
target date: training through 2013-12-31; a purged gap; validation from
2015-07-01 through 2018-06-30; another purged gap; and untouched testing from
2020-01-01 through 2026-07-31. The gaps are long enough that target covariance
windows in adjacent splits share no return observation, including the S&P
$T=357$ specification. Within each split the 14-day examples still overlap,
so the table reports both their count and a greedy count of non-overlapping
target windows.

Loss at level $d$ is the normalised projector loss
$d^{-1}\sum_j\sin^2\theta_j\in[0,1]$. Complete-flag loss is the mean of the
normalised $d=1,3,6$ losses, matching the inverse-dimension weighting frozen in
Stage 1. Lower is better. This first benchmark scores geometry only; covariance
reconstruction, RMT excess and realised portfolio losses enter later without
changing these geometric test examples.

**Verdict.** **BASELINE RECORDED.** This is not a YAY/NAY gate: the following
untouched test losses are the numbers later forecasts must reduce.

| panel | market loss | top-three loss | top-six loss | complete Flag mean | complete Flag median (IQR) | test examples / non-overlapping targets |
|---|---:|---:|---:|---:|---:|---:|
| S&P full | 0.00367 | 0.02594 | 0.04357 | **0.02439** | 0.01789 (0.01425–0.02561) | 118 / 5 |
| Nikkei full | 0.00467 | 0.03073 | 0.10801 | **0.04781** | 0.04402 (0.03334–0.06189) | 115 / 7 |
| DAX full | 0.00513 | 0.07255 | 0.09195 | **0.05654** | 0.05270 (0.03562–0.06941) | 120 / 7 |
| CAC full, cleaned | 0.00487 | 0.02674 | 0.09966 | **0.04376** | 0.04036 (0.02851–0.05650) | 120 / 7 |

The market direction is easiest to hold fixed on every panel; the outer
six-space is the largest source of frozen-forecast loss on Nikkei and CAC,
while DAX's top-three and top-six losses are both large. These are descriptive
held-out baselines, not independent-sample significance claims: the effective
non-overlapping counts are only five to seven. Benchmark 1.2 will replace only
the forecast slot with the full incoming tangent and score the same rows.

Reproduce all four panels with:

```bash
python scripts/stage2_benchmark1_1_frozen_flag.py --label all
```

All later Family 1 results use skill relative to this benchmark,

$$
100\left(1-\frac{\overline L_{\rm model}}
                  {\overline L_{\rm Frozen}}\right).
$$

Zero is Frozen Flag, positive skill removes error and negative skill adds it.
The ratio is formed from mean losses rather than averaging per-origin ratios,
which would be unstable when a Frozen loss is nearly zero. Combined skill is
the equal-weight mean of the four panel skills; asset count and the number of
overlapping origins do not determine a panel's weight. Evidence tables report
the paired origin win fraction and a circular calendar-block 95% interval in
skill points. The conservative block spans $(T+2h)/14$ origins, covering the
past, current and future windows used by Constant Velocity.

## Benchmark 1.2 — Constant Velocity

**What is tested.** Does repeating the complete previous Flag motion improve
on predicting no motion? One block-aligned minimum-plane transport maps the
past Flag to the current Flag; applying that same orthogonal transport again
produces a valid nested forecast rather than three independently extrapolated,
potentially non-nested Grassmann spaces.

**Verdict.** **NAY on every panel and component.** Equal-market complete-Flag
skill is -53.71%, with no panel improving. This reproduces the Stage 1 result
in the richer geometry: the previous direction is informative, but its full
length overshoots severely.

| component | combined skill | worst panel | panels improved |
|---|---:|---:|---:|
| market | -59.88% | -68.41% | 0 / 4 |
| block 2:3 | -53.39% | -68.45% | 0 / 4 |
| block 4:6 | -50.58% | -52.91% | 0 / 4 |
| top three | -53.71% | -67.74% | 0 / 4 |
| top six | -54.59% | -58.71% | 0 / 4 |
| complete Flag | **-53.71%** | **-57.44%** | **0 / 4** |

| panel | mean loss | median (IQR) | skill | origins won | paired 95% skill interval |
|---|---:|---:|---:|---:|---:|
| S&P full | 0.03645 | 0.02898 (0.02575–0.03986) | -49.44% | 11.9% | [-54.71%, -43.86%] |
| Nikkei full | 0.07527 | 0.07125 (0.05607–0.08983) | -57.44% | 4.3% | [-64.76%, -51.13%] |
| DAX full | 0.08546 | 0.07930 (0.06122–0.09544) | -51.15% | 10.8% | [-57.71%, -44.63%] |
| CAC full | 0.06861 | 0.06719 (0.05205–0.08137) | -56.80% | 10.0% | [-65.06%, -49.80%] |

## Benchmark 1.3 — ERSE Direction

**What is tested.** Can the Flag of Liu and Liu's within-window ERSE
correction, at its published primary threshold $\delta=0.25$, stand in for the
future empirical Flag?

**Verdict.** **NAY at every origin on every panel.** ERSE barely changes the
two disjoint core blocks, but moves the already-stable market direction so far
that complete-Flag skill falls to -189.90%. This directly confirms that ERSE
is neither the temporal signal nor a usable forecast of it.

| component | combined skill | worst panel | panels improved |
|---|---:|---:|---:|
| market | -3252% | -4481% | 0 / 4 |
| block 2:3 | -1.52% | -5.10% | 1 / 4 |
| block 4:6 | -2.12% | -4.54% | 0 / 4 |
| top three | -147.2% | -211.4% | 0 / 4 |
| top six | -33.16% | -62.84% | 0 / 4 |
| complete Flag | **-189.90%** | **-337.09%** | **0 / 4** |

| panel | mean loss | median (IQR) | skill | origins won | paired 95% skill interval |
|---|---:|---:|---:|---:|---:|
| S&P full | 0.10662 | 0.10008 (0.09699–0.10762) | -337.09% | 0% | [-348.98%, -325.15%] |
| Nikkei full | 0.12509 | 0.12023 (0.10859–0.13849) | -161.67% | 0% | [-167.04%, -156.18%] |
| DAX full | 0.12671 | 0.12243 (0.10738–0.14020) | -124.10% | 0% | [-127.40%, -119.73%] |
| CAC full | 0.10359 | 0.10238 (0.08669–0.11710) | -136.74% | 0% | [-150.11%, -120.33%] |

## Benchmark 1.4 — HCAL Flag

**What is tested.** Does the leading Flag of a single average-linkage
hierarchical correlation filter forecast the next empirical rolling Flag?
HCAL is fitted to the same standardised current return window used by Stage 1.

**Verdict.** **NAY at every origin on every panel.** HCAL remains a legitimate
complete-covariance filter in Family 2, but its structural basis is not a point
forecast of the future empirical eigenspace.

| component | combined skill | worst panel | panels improved |
|---|---:|---:|---:|
| market | -492.1% | -863.3% | 0 / 4 |
| block 2:3 | -892.1% | -1213% | 0 / 4 |
| block 4:6 | -326.9% | -674.7% | 0 / 4 |
| top three | -895.3% | -1233% | 0 / 4 |
| top six | -414.2% | -875.8% | 0 / 4 |
| complete Flag | **-535.2%** | **-1001.6%** | **0 / 4** |

| panel | mean loss | median (IQR) | skill | origins won | paired 95% skill interval |
|---|---:|---:|---:|---:|---:|
| S&P full | 0.26873 | 0.26570 (0.25013–0.29096) | -1001.63% | 0% | [-1067.42%, -950.59%] |
| Nikkei full | 0.27035 | 0.26860 (0.25577–0.28580) | -465.52% | 0% | [-479.41%, -451.76%] |
| DAX full | 0.20769 | 0.20982 (0.18022–0.23828) | -267.31% | 0% | [-302.38%, -228.65%] |
| CAC full | 0.22147 | 0.22433 (0.20346–0.24005) | -406.15% | 0% | [-436.23%, -370.95%] |

## Benchmarks 1.6--1.10 -- causal rolling and financial forecasters

**What is tested.** The first five comparators either held a Flag still,
repeated its complete previous motion, or substituted a contemporaneous
covariance filter. These five forecasters instead use only information
available at the current origin to predict the future rolling Flag.

- **Retained Window** removes the oldest 42 observations, which are already
  known to be absent from the target window, and extracts the Flag of the
  remaining $T-42$ observations.
- **Stationary Roll-Forward** adds a causal stationary fill for the 42 unseen
  observations,
  $$
  \widehat C_{t+42\mid t}
  =\frac{1}{T}\left(\sum_{s\in\mathrm{retained}}z_sz_s^\top
  +42\widehat R_t\right).
  $$
- **RiskMetrics EWMA** extracts the Flag of the canonical fixed
  $\lambda=0.94$ EWMA correlation forecast.
- **Validation-Geometric EWMA** chooses one half-life from the frozen grid by
  mean validation complete-Flag loss, then applies it unchanged to testing.
- **Factor CM-IEWMA** follows Johansson et al.'s large-universe route: a
  causal 20-factor PCA, three IEWMA factor experts with half-life pairs
  $(10,20)$, $(20,60)$ and $(60,120)$, their published ten-day convex
  precision-factor combination, and a 21-day EWMA residual diagonal.

Every method is scored only through the resulting nested
$\mathrm{Flag}(N;1,3,6)$. No covariance, likelihood or portfolio score enters
this Family 1 comparison. The geometric EWMA tuning selected a 252-day
half-life independently on all four validation panels: 54 validation origins
for S&P, Nikkei and CAC and 55 for DAX.

**Verdict.** **YAY for Retained Window, Stationary Roll-Forward and tuned
EWMA; NAY for fixed RiskMetrics and factor CM-IEWMA.** Knowing which returns
will leave the rolling target removes about 40% of Frozen's geometric error.
The stationary fill adds no aggregate value in this implementation. Retained
Window recomputes per-name scaling on the retained subset, whereas the
stationary construction carries current-window scaling into its retained
scatter, so their near equality is not by itself an exact attribution result;
Regime 4.9 supplies that structurally matched test.
Objective-matched EWMA removes another substantial 20%, proving that an
established causal adaptive estimator can beat Frozen, although it remains far
behind Retained Window. The two short-memory financial forecasts aim at
conditional covariance rather than a long-window empirical Flag and miss this
target badly.

| benchmark | market skill | $2{:}3$ skill | $4{:}6$ skill | top-three skill | top-six skill | complete-Flag skill | worst panel |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.6 Retained Window | +40.89% | +38.45% | +39.53% | +38.50% | +40.65% | **+39.56%** | +37.36% |
| 1.7 Stationary Roll-Forward | +41.43% | +37.82% | +39.16% | +37.88% | +40.50% | **+39.24%** | +36.82% |
| 1.8 RiskMetrics EWMA | -1869% | -891.6% | -371.3% | -903.9% | -476.8% | **-629.7%** | -1121.5% |
| 1.9 Validation-Geometric EWMA | +22.18% | +25.05% | +19.06% | +24.90% | +17.61% | **+20.10%** | +16.19% |
| 1.10 Factor CM-IEWMA | -90.17% | -90.71% | -68.59% | -91.18% | -62.42% | **-71.61%** | -187.84% |

Each cell below is `mean complete-Flag loss / Frozen-relative skill`. Every
positive method-panel interval excludes zero; every negative method-panel
interval excludes zero in the adverse direction.

| benchmark | S&P | Nikkei | DAX | CAC 40 |
|---|---:|---:|---:|---:|
| Retained Window | 0.01528 / +37.36% | 0.02852 / +40.33% | 0.03410 / +39.69% | 0.02589 / +40.84% |
| Stationary Roll-Forward | 0.01541 / +36.82% | 0.02875 / +39.86% | 0.03433 / +39.29% | 0.02582 / +41.00% |
| RiskMetrics EWMA | 0.29797 / -1121.49% | 0.30262 / -533.03% | 0.27382 / -384.28% | 0.25381 / -480.06% |
| Validation-Geometric EWMA | 0.01801 / +26.19% | 0.03814 / +20.21% | 0.04647 / +17.82% | 0.03667 / +16.19% |
| Factor CM-IEWMA | 0.07021 / -187.84% | 0.06459 / -35.11% | 0.07262 / -28.44% | 0.05909 / -35.03% |

These ten rows remain useful historical diagnostics, but Regime 4.9 changes the
contest rather than merely its denominator. Every new geometric forecaster must
start from the identical Retained Window Flag and predict an incoming-block
tangent. Beating Retained then establishes information about the unseen part of
the future Flag; Frozen-relative skill alone no longer supports that claim.

Reproduce any completed Family 1 comparator by replacing the estimator name:

```bash
python scripts/stage2_benchmark_family1.py --estimator constant_velocity --label all
python scripts/stage2_benchmark_family1.py --estimator erse --label all
python scripts/stage2_benchmark_family1.py --estimator hcal --label all
python scripts/stage2_benchmark_family1.py --estimator bahc --label all
python scripts/stage2_benchmark_family1.py --estimator retained_window --label all
python scripts/stage2_benchmark_family1.py --estimator roll_forward --label all
python scripts/stage2_benchmark_family1.py --estimator riskmetrics_ewma --label all
python scripts/stage2_benchmark_family1.py --estimator tuned_ewma --label all
python scripts/stage2_benchmark_family1.py --estimator cm_iewma --label all
```

## Benchmark Family 2 — full covariance estimators

**What is tested.** Can any learned geometric forecast eventually improve on
the established ways of estimating the complete covariance matrix? This is a
different contest from Benchmark 1.1. Ledoit-Wolf, OAS, QIS/RIE and
cross-validated eigenvalue shrinkage retain the current sample eigenvectors,
so geometrically they still predict a frozen Flag; they become different
forecasts only because they clean the eigenvalues. HCAL and BAHC genuinely
alter the eigenvectors and therefore compete in both geometry and covariance
quality.

**Setup.** Every estimator receives the same raw $T$ returns available at the
forecast origin. It is scored against the covariance of the next 42 returns,
which begin strictly after that origin. EWMA chooses one half-life using mean
validation Gaussian log loss and never reads test performance. All final
scores use the identical untouched 2020–2026 test origins: 118 for S&P, 115
for Nikkei and 120 for DAX/CAC. BAHC uses 100 bootstrap hierarchies per origin
with fixed seed 20260803. The integrity collector verified all 36
panel-estimator cells have exactly the expected dates and no missing primary
score.

The primary economic score below is realised annualised volatility of the
unconstrained global-minimum-variance portfolio; lower is better.

| estimator | S&P | Nikkei | DAX | CAC 40 |
|---|---:|---:|---:|---:|
| sample covariance | 149.55% | 17.07% | 14.23% | 12.61% |
| validation-tuned EWMA | 149.55% | 17.07% | 14.23% | 12.61% |
| Ledoit-Wolf linear shrinkage | 14.63% | 14.02% | **13.78%** | **12.36%** |
| Ledoit-Wolf QIS / RIE | 24.48% | 13.08% | 13.88% | 12.43% |
| BAHC, 100 bootstraps | 14.41% | 13.80% | 14.36% | 13.32% |
| RiskMetrics EWMA, $\lambda=0.94$ | 18.18% | 31.52% | 18.54% | 16.27% |
| OAS | 16.11% | 14.75% | 13.96% | 12.47% |
| isotonic 10-fold CVC | **12.74%** | **12.95%** | 13.79% | 12.42% |
| HCAL | 15.26% | 14.08% | 14.58% | 13.43% |

The winners depend on the loss, which is why all four were frozen in advance:

| score | S&P winner | Nikkei winner | DAX winner | CAC winner |
|---|---|---|---|---|
| Gaussian predictive log loss | BAHC, -7.467 | BAHC, -7.373 | Ledoit-Wolf, -7.232 | CVC, -7.397 |
| long/short GMV volatility | CVC, 12.74% | CVC, 12.95% | Ledoit-Wolf, 13.78% | Ledoit-Wolf, 12.36% |
| long-only GMV volatility | BAHC, 12.69% | CVC, 13.46% | Ledoit-Wolf, 13.62% | Ledoit-Wolf, 12.47% |
| relative Frobenius error | RiskMetrics, 1.017 | Ledoit-Wolf, 0.855 | RiskMetrics, 0.851 | RiskMetrics, 0.918 |

**Verdict.** **BASELINES RECORDED — YAY for the evaluation system, no model
verdict yet.** No estimator dominates every panel and loss. CVC is the
strongest long/short competitor on the two large panels; Ledoit-Wolf is the
strongest on DAX and CAC; BAHC improves materially on a single HCAL tree and
wins several predictive-log and long-only comparisons. Fast fixed-decay EWMA
can reduce entrywise matrix error while worsening realised portfolio risk,
showing that Frobenius loss is not a sufficient economic verdict.

The S&P sample and validation-tuned EWMA results are intentionally severe. At
$T=N=357$, demeaning leaves at most 356 independent return directions, so the
sample covariance is singular. The common scoring floor is reported rather
than hidden; unconstrained inversion amplifies that null direction, while
linear shrinkage, QIS, CVC and the long-only constraint regularise it in
different ways. Validation selects the exact uniform-weight limit for S&P,
$4032$ days for Nikkei and $252$ days for DAX/CAC.

QuEST remains a historical reference rather than an executed row. Its authors
publish the legacy estimator as a 755 KB MATLAB package and no MATLAB or Octave
runtime is installed here. QIS is their newer, invertible Python implementation
of nonlinear rotationally invariant shrinkage and is the operational RIE
baseline. The code does not relabel QIS as QuEST or substitute an unverified
third-party implementation.

**Pending Benchmark 2.11 — Factor CM-IEWMA covariance.** CM-IEWMA will be
evaluated in Family 2 using its native complete positive-definite covariance
forecast and the same future-return losses as every other covariance estimator.
It is a neighbouring competitor, not the proposed method: CM-IEWMA combines
covariance forecasts over several memory lengths, whereas the Family 1 model
forecasts the signed direction of a rolling partial Flag. Benchmark 1.10
extracted a Flag from CM-IEWMA and found poor geometric skill, but that asks the
method to solve a target it was not designed for and is not treated as a
negative verdict on the published covariance forecaster.

Reproduce and validate the suite with:

```bash
python scripts/stage2_benchmark2_covariance.py --label all
python scripts/stage2_benchmark2_structural.py --label all --estimator cvc
python scripts/stage2_benchmark2_structural.py --label all --estimator hcal
python scripts/stage2_benchmark2_structural.py --label all --estimator bahc --bahc-bootstraps 100
python scripts/stage2_collect_benchmark2.py
```

## The Oracle Line — can Flag geometry become a covariance forecast?

**What is tested.** Determine whether the partial-Flag route has enough useful
information to justify fitting causal Family 1 models, and identify what is
still missing when it does not match the strongest complete-covariance
estimator. The common control holds the current Flag, installs the current
QIS-cleaned correlation spectrum and uses the same validation-selected EWMA
marginal-volatility forecast throughout. The four infeasible oracles then add,
in order, the exact future rolling Flag; its future QIS-cleaned spectrum; the
complete future rolling QIS correlation; and the realised next-42-day marginal
volatilities.

**Setup.** The exact future $\mathrm{Flag}(N;1,3,6)$ is installed through an
ordered minimum-plane orthogonal transport. Each disjoint $1$, $2$ and $3$
dimensional block is first Procrustes-aligned to the current block, preventing
the oracle from reading arbitrary within-block eigenvector bases that the Flag
does not contain. Reconstructed matrices are renormalised to correlation
before scale is installed. Every row uses the same untouched test origins and
the same four losses as Benchmark Family 2. Circular calendar-block intervals
use $\lceil(T+42)/14\rceil$ origins, covering the full return-dependence span.

Oracle 1 improves every score relative to its matched frozen reconstruction.
Positive percentages below mean lower loss, and all sixteen block-bootstrap
95% intervals exclude zero in the favourable direction.

| panel | Frobenius improvement | Gaussian-log improvement | long/short GMV improvement | long-only GMV improvement |
|---|---:|---:|---:|---:|
| S&P full | +2.01% | +1.43% | +11.69% | +5.37% |
| Nikkei full | +3.16% | +0.74% | +9.99% | +4.02% |
| DAX full | +2.43% | +0.54% | +3.19% | +1.97% |
| CAC full | +2.13% | +0.54% | +2.96% | +2.14% |

This is not merely an easy win against Frozen Flag. Against the separately
best Family 2 estimator for each panel and loss, Oracle 1 has the better mean
in 14 of 16 cells. Seven are clear block-interval wins, seven remain
indistinguishable at this effective sample size, and two are clear losses:
S&P Gaussian log loss and S&P unconstrained GMV volatility. The S&P long-only
GMV result nevertheless improves from BAHC's 12.69% to 11.81%.

| panel | Oracle 1 Frobenius | Oracle 1 Gaussian log | Oracle 1 long/short GMV | Oracle 1 long-only GMV |
|---|---:|---:|---:|---:|
| S&P full | **0.875** | -6.861 | 18.80% | **11.81%** |
| Nikkei full | **0.654** | **-7.397** | **11.38%** | **13.00%** |
| DAX full | **0.688** | **-7.240** | **13.14%** | **13.42%** |
| CAC full | **0.757** | **-7.426** | **12.00%** | **12.21%** |

Bold values have a better mean than that panel's corresponding Family 2
winner; this table does not use bold to imply that every difference clears the
block interval. Oracle 2 adds only modest value on most panels and worsens S&P
unconstrained GMV from 18.80% to 19.60%, so eigenvalue forecasting is not the
principal missing ingredient. Oracle 3 produces the large S&P jump: supplying
the complete future rolling correlation reduces unconstrained GMV to 8.57%
and Gaussian log loss to -7.824. Structure beyond the top-six Flag therefore
matters materially on the largest panel. Oracle 4 wins every score decisively,
showing that marginal-volatility forecasting remains a second major problem.

**Verdict.** **YAY — Family 1 is a legitimate second route to full covariance
forecasting, but the oracle has not made a deployable model win.** Perfect
top-six Flag knowledge produces consistent full-matrix value with past-only
spectrum and scale, which justifies Global Damping and Layerwise Damping as
covariance models rather than geometry-only exercises. A learned model must
still reproduce enough of Oracle 1's gain on untouched data. On S&P, a richer
correlation representation or complement model is required before this route
can challenge Family 2 on Gaussian likelihood and unconstrained portfolios;
forecasting eigenvalues alone does not close that gap.

Reproduce the 2,365 scored rows, block comparisons and integrity diagnostics
with:

```bash
python scripts/stage2_oracle_line.py --label all --bootstrap-repetitions 2000
```

## The constraint governing model fitting

The full S&P panel holds **18.7 non-overlapping windows** at $T=357$, up from
7.7 in the 2000–2010 replication period. Nikkei and the small panels hold about
26–27 at $T=250$. Sliding at step 14 creates hundreds of triples, but they share
returns and are not hundreds of independent observations. No architecture
repairs that, and an in-sample curve fitted against them means little.

Three things change the arithmetic, in rough order of value:

- **Synthetic supervision.** Stage 1 leaves behind a generator that produces
  worlds with a known injected rotation and a calibrated null — ground truth, in
  unlimited quantity, at any `nu`, `T`, `N` and bulk geometry. Pretraining there
  and evaluating on real data is a far better-founded use of the panels than
  training on them.
- **More history and more universes.** The replication panels remain frozen at
  2010; separate `*_full` panels now extend through the latest 2026 observation.
  They improve the arithmetic materially but do not create independent data.
- **Pooling across blocks.** Many `(P, Q)` choices and many sub-universes drawn
  from the same panel are not independent, but they are not identical either.

## Shape of the estimator

Not "RIE plus a rotation factor" — RIE's optimality is optimality *within* the
rotationally-invariant class, and rotating leaves the class and forfeits the
guarantee. The construction that survives is a forecast: predict the nested
flag from history, report market/core/buffer losses separately, then apply
eigenvalue shrinkage **in the predicted basis** when constructing a covariance
forecast. Justify the whole estimator on out-of-sample realised risk against an
RIE baseline rather than by inheriting anyone's theory.

The first chronological model ladder is now frozen:

1. finish geometric Benchmark Family 1;
2. make every full-covariance estimator in Benchmark Family 2 operational and
   test its no-lookahead and positive-definiteness properties;
3. **Model 3.1 — Global Damping:** completed below with one validation-selected coefficient;
4. **Model 3.2 — Layerwise Damping:** fit separate market/core/buffer coefficients;
5. **Model 4.1 — Transported Tangent AR:** introduce state-dependent temporal
   structure;
6. only then, attempt Model Family 5's richer sequence and full-SPD challengers.

Every rung must beat the previous rung on untouched future blocks. A richer
model earns its complexity only through held-out improvement; Regime 4.7 gives
it a legitimate target, not a free pass.

## Model 3.1 — Global Damping

**What is tested.** Can one learned scalar turn the directional persistence
from Stage 1 into a genuine out-of-sample Flag forecast? The model changes only
the length of the previous complete Flag rotation. It does not use Retained
Window information, separate the Flag layers or introduce additional state.

**Setup.** At every origin, the ordered minimum-plane rotation carrying the
previous Flag to the current Flag is reconstructed. Every plane angle in that
single nested orthogonal motion is multiplied by the same coefficient
$\alpha$, and the damped motion is continued from the current Flag. Therefore
$\alpha=0$ is exactly Frozen Flag and $\alpha=1$ is exactly Constant Velocity.
For each panel separately, $\alpha$ is selected from
$\{0,0.025,\ldots,1\}$ using only 2015–2018 validation complete-Flag loss. The
selected value is then frozen before evaluation on the existing test origins
from 2020 onward. The same six geometric losses and calendar-block intervals
used by Benchmark Family 1 are retained.

| panel | selected $\alpha$ | validation origins | validation skill versus $\alpha=0$ |
|---|---:|---:|---:|
| S&P 500 | 0.225 | 54 | +4.54% |
| Nikkei | 0.100 | 54 | +0.80% |
| DAX | 0.100 | 55 | +0.85% |
| CAC 40 | 0.050 | 54 | +0.16% |

All four validation fits select a positive but heavily damped continuation.
The untouched test results preserve that ordering.

| panel | test complete-Flag loss | skill versus Frozen | origins won | paired block 95% skill interval |
|---|---:|---:|---:|---:|
| S&P 500 | 0.02341 | **+4.02%** | 69.5% | **[+1.20%, +7.19%]** |
| Nikkei | 0.04729 | **+1.08%** | 62.6% | **[+0.56%, +1.70%]** |
| DAX | 0.05592 | **+1.09%** | 57.5% | **[+0.47%, +1.70%]** |
| CAC 40 | 0.04364 | **+0.26%** | 55.0% | [−0.21%, +0.71%] |

The equal-market result is positive at every diagnostic level.

| component | combined skill versus Frozen | worst panel |
|---|---:|---:|
| Market | **+3.21%** | **+1.21%** |
| $2{:}3$ | **+3.17%** | **+0.87%** |
| $4{:}6$ | **+1.09%** | **+0.02%** |
| Top three | **+3.11%** | **+0.91%** |
| Top six | **+0.79%** | **+0.04%** |
| Complete Flag | **+1.61%** | **+0.26%** |

Retained Window remains much stronger. Global Damping loses to it by 53.2% on
S&P, 65.8% on Nikkei, 64.0% on DAX and 68.6% on CAC; every paired interval is
strictly adverse, for an equal-market complete-Flag skill of −62.9%.

**Verdict.** **LEGACY DIAGNOSTIC — positive versus Frozen, NAY for the corrected
forecast question.** A validation-fitted coefficient improves the old test mean
on all four panels, but the model begins from the full current Flag and therefore
mixes known deletion with the unseen contribution. Its −62.9% skill versus
Retained is not merely a difficult benchmark result: Regime 4.9 shows that this
is the wrong primitive. The numerical fit remains reproducible, but it is not
evidence that unseen-return geometry has been learned. Its replacement must
damp the preceding realised incoming-block tangent from the retained base.

Reproduce the fit, test series and both paired comparisons with:

```bash
python scripts/stage2_model3_1_global_damping.py --label all --bootstrap-repetitions 2000
```
