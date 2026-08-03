# Stage 2 — Forecast

**Model fitting has not started.** The entry-gate experiments now exist as
`scripts/regime4_4_tangent.py`, `scripts/regime4_5_coherence.py` and
`scripts/regime4_6_erse.py`, with the final representation gate in
`scripts/regime4_7_flag.py`; they remain Stage 1 because they test whether a
forecastable and interpretable label exists rather than fitting one. Regime 4.7
clears that gate: the master Stage 2 state is now
$\mathrm{Flag}(N;1,3,6)$.

Stage 1 builds an instrument that *measures* how far the leading subspace has
rotated between two windows, against a null that says how far it would appear to
rotate if nothing had moved at all. Stage 2 is the attempt to *predict* that
rotation, and to turn the prediction into a covariance estimator that beats a
rotation-invariant one out of sample.

Everything currently in the repository is stage 1, including the real-data
regimes. Being on real returns does not make it stage 2 — regime 4 is still
establishing that the measurement is trustworthy, which is the same job as
regimes 1–3 with a harder subject.

## Reporting contract

Every gate is reported in the same order: **what is tested; setup; verdict.**
The verdict words have fixed meanings:

- **YAY:** the primary standardised statistic beats its matched null at the 5%
  level and the effect has the sign required by the claim.
- **NAY:** the primary statistic does not clear that gate, or has the wrong
  sign.
- **INCONCLUSIVE:** the data or null resolution cannot decide. This is not
  silently converted into a negative result.

## Entry condition

**A label.** Concretely: an excess-rotation series that survives all five of
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

Until a series clears the forecast gates and its interpretation is settled by
the coherence gate, fitting a large model would risk learning the null. These
tests are not substitutes. The outcome map is:

| Regime 4.4 persistence | Regime 4.5 coherence | Consequence |
|---|---|---|
| no | either | no Stage 2; write up displacement without forecastability |
| yes | no | forecastable time-varying loadings, but not evidence for coherent rotation |
| no | yes | coherent movement exists but is not forecastable; scientific result, no ML |
| yes | yes | original Stage 2 claim survives; fit the small tangent model |

Regime 4.7 has now checked that the last row is not an artefact of selecting
only $P=3$: the complete nested flag also survives.

## Regime 4.4 — does the previous direction help?

**What is tested.** Does the leading eigenspace tend to continue rotating in
the same direction? At the current space $Y_t$, the incoming and outgoing
motions are

$$H_t^-=-\operatorname{Log}_{Y_t}(Y_{t-h}),\qquad
H_t^+=\operatorname{Log}_{Y_t}(Y_{t+h}).$$

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
$E_t=\operatorname{Log}_{Y_t}(Y_t^{\mathrm{ERSE}})$. Measure alignment with the
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

## Shape of the estimator, when it comes

Not "RIE plus a rotation factor" — RIE's optimality is optimality *within* the
rotationally-invariant class, and rotating leaves the class and forfeits the
guarantee. The construction that survives is a forecast: predict the nested
flag from history, report market/core/buffer losses separately, then apply
eigenvalue shrinkage **in the predicted basis** when constructing a covariance
forecast. Justify the whole estimator on out-of-sample realised risk against an
RIE baseline rather than by inheriting anyone's theory.

The first chronological model ladder is now frozen:

1. hold the current flag fixed;
2. constant-velocity flag extrapolation, including the full-step rule and a
   single validation-fitted damping coefficient;
3. separately damped market/core/buffer velocities;
4. a transported tangent autoregression with state-dependent damping;
5. only then, a richer sequence model and a full-SPD challenger.

Every rung must beat the previous rung on untouched future blocks. A richer
model earns its complexity only through held-out improvement; Regime 4.7 gives
it a legitimate target, not a free pass.
