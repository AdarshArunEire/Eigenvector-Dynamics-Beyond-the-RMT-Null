# Stage 2 benchmark registry

> **Superseded, 2026-08-05.** Rows 1.1–1.10 score a *geometric* loss against
> another estimated Flag. The final Stage 2 design scores realised returns
> instead, which removed the deletion artifact rather than correcting it — under
> the capture score Retained Window went from +39.56% to losing significantly on
> 22 of 24 cells. The live ladder is `scripts/stage2_capture_ladder.py` and the
> final model is `scripts/stage2_model4_1_visible_coupling.py`. This registry is
> retained as the frozen, reproducible historical contest.

This registry was frozen before fitting the first learned rotation model. Regime
4.9 subsequently identified deterministic rolling deletion as a structural
confound in that contest. Rows 1.1–1.10 remain frozen and reproducible as
historical full-Flag diagnostics, but no later model may earn credit for the
known deletion schedule.

The corrected Family 1 primitive is: construct the Flag of the $T-42$
observations known to remain, then predict the tangent contributed by the next
42 unseen returns. Retained Window is the zero-residual rule. Every new model is
scored against the full future Flag and reports skill relative to Retained;
Frozen-relative skill is secondary only.

The registry separates
two questions that cannot be answered by one score:

1. **Can the future eigenspace be forecast?** Compare predicted Flags using
   basis-invariant geometric loss.
2. **Does that forecast improve a useful covariance estimator?** Reconstruct a
   full covariance matrix and compare it with established covariance estimators
   using realised future returns and portfolio risk.

Ledoit–Wolf, OAS, cross-validated shrinkage and RIE retain the current sample
eigenvectors. Their Flag forecast is therefore exactly the Frozen Flag
benchmark; they become distinct only in the full-covariance branch. BAHC and
HCAL alter eigenvectors and can be scored in both branches.

## Family 1 — geometric forecast benchmarks

| number | name | forecast available at time $t$ | role |
|---|---|---|---|
| 1.1 | Frozen Flag | $\widehat H_t^+=0$ | mandatory zero-motion denominator |
| 1.2 | Constant Velocity | $\widehat H_t^+=H_t^-$ | mandatory full-step rule already known to overshoot in-sample |
| 1.3 | ERSE direction | use the published within-window ERSE correction as the predicted direction | paper comparator and expected negative control |
| 1.4 | HCAL filtered Flag | hold the current average-linkage hierarchical eigenspace | structural eigenvector-filter baseline |
| 1.5 | BAHC filtered Flag | hold the current bootstrapped-average hierarchical eigenspace | strongest published eigenvector-filter comparator in the recovered literature |
| 1.6 | Retained-Window Flag | discard the 42 observations known to expire and use the Flag of the retained $T-42$ observations | parameter-free rolling-composition forecast |
| 1.7 | Stationary Roll-Forward | retained scatter plus 42 copies of the current causal correlation estimate | conditional rolling-window forecast with a stationary unseen batch |
| 1.8 | RiskMetrics EWMA Flag | Flag of the fixed $lambda=0.94$ EWMA correlation forecast | canonical short-memory financial comparator |
| 1.9 | Validation-Geometric EWMA Flag | Flag of an EWMA whose half-life minimises validation complete-Flag loss | objective-matched adaptive financial comparator |
| 1.10 | Factor CM-IEWMA Flag | Flag of the published large-universe factor CM-IEWMA forecast | multi-timescale external financial comparator |

All ten historical rows use the same purged dates and are scored at market, top-three,
top-six and complete-flag levels. Benchmark 1.1 is already recorded. Benchmark
1.2 changes only the forecast slot. HCAL and BAHC additionally enter Family 2
as complete covariance estimators. Benchmarks 1.6--1.10 produce covariance or
scatter estimates internally only to locate their predicted Flags; Family 1
scores no matrix, likelihood or portfolio quantity.

Model 3.1 also belongs to this historical full-Flag contest. Its successor must
apply learned damping to the preceding realised incoming-block tangent after
transport to the current retained Flag; $\alpha=0$ must equal Retained Window,
not Frozen Flag.

## Family 2 — full-covariance benchmarks

Every estimator receives returns available through the same forecast origin.
Except for explicitly weighted EWMA, the rolling information window is the same
$T$ used to estimate the Flag. The realised target is the **next 42 returns**,
not the overlapping $T$-day future Flag window.

### Primary comparators — decide whether the model succeeds

| number | estimator | why it is mandatory |
|---|---|---|
| 2.1 | Rolling sample covariance | unfiltered full-matrix hold-still rule |
| 2.2 | Validation-tuned EWMA | strongest simple time-adaptive rule suggested by the external reader; half-life selected on validation only |
| 2.3 | Ledoit–Wolf linear shrinkage | canonical well-conditioned linear covariance estimator |
| 2.4 | Ledoit-Wolf QIS / RIE | modern invertible nonlinear eigenvalue cleaner derived for Frobenius and minimum-variance loss; sample eigenvectors retained |
| 2.5 | BAHC | published eigenvector-filtering competitor with the same 42-day realised-risk protocol |

The learned geometric estimator cannot claim success merely by beating Frozen
Flag. After covariance reconstruction it must be compared with every primary
estimator on the untouched test period.

### Secondary comparators — diagnose where performance comes from

| number | estimator | purpose |
|---|---|---|
| 2.6 | RiskMetrics EWMA, $\lambda=0.94$ | canonical fixed-decay reference, separated from tuned EWMA |
| 2.7 | OAS | closed-form Gaussian shrinkage comparator; cheap sensitivity to the linear-shrinkage formula |
| 2.8 | QuEST nonlinear shrinkage | historical reference used by the BAHC paper; not executed because the authors distribute a MATLAB package and this workspace has no MATLAB/Octave runtime |
| 2.9 | Cross-validated eigenvalue shrinkage | BAHC paper's data-driven rotationally invariant comparator |
| 2.10 | HCAL | single hierarchical-tree filter; isolates what bootstrap averaging adds to BAHC |
| 2.11 | Factor CM-IEWMA covariance | pending native full-covariance competitor; online multi-timescale conditional-covariance forecast scored on the same future-return losses as Family 2 |

Benchmark 2.11 is not a reformulation of the proposed geometric forecaster.
CM-IEWMA predicts a full conditional covariance matrix by combining estimators
with different memory lengths; this project predicts the signed future motion
of a rolling partial Flag. Benchmark 1.10's Flag extraction is therefore a
target-mismatch diagnostic. Its geometric score is not a verdict on CM-IEWMA
at the task for which it was designed.

The EWMA half-life grid is frozen at
$5,10,21,42,63,126,252,504,1008,2016,4032,8064,16128$ trading days plus the
exact uniform-weight limit. It was extended before model fitting because
validation-only smoke tests selected the former upper boundaries; no
test-period score is used for selection. Mean validation Gaussian predictive
log loss is the sole tuning criterion, with the shorter half-life breaking an
exact tie. EWMA removes its weighted mean and uses maximum-likelihood scaling,
without a weighted degrees-of-freedom correction, so its uniform limit exactly
matches Benchmark 2.1's $1/T$ covariance convention.

QIS is not silently renamed QuEST. It is the Ledoit-Wolf authors' newer,
invertible nonlinear RIE implementation and the operational primary baseline.
QuEST remains listed to preserve the historical comparison set, but it cannot
enter a numerical table until its official MATLAB code can be run and checked.

### The Oracle Line — infeasible information ceilings

The common control combines the frozen current Flag, the current QIS-cleaned
correlation spectrum and the validation-selected EWMA marginal-volatility
forecast. Four oracles then add future information cumulatively:

1. **Oracle 1 — Future Flag:** exact future rolling
   $\mathrm{Flag}(N;1,3,6)$; spectrum, complement and scale remain past-only.
2. **Oracle 2 — Future Flag and spectrum:** additionally use the future
   rolling window's QIS-cleaned spectrum.
3. **Oracle 3 — Future rolling correlation:** additionally replace the
   transported complement with the complete future rolling QIS correlation.
4. **Oracle 4 — Future correlation and scale:** additionally use realised
   next-42-day marginal volatilities.

The Flag lift is an ordered minimum-plane orthogonal transport. Each future
Flag block is first Procrustes-aligned to the current block, so reconstruction
cannot read arbitrary within-block eigenvector bases. All forecasts are
renormalised to correlation before a common diagonal volatility forecast is
installed. The actual next-42-day covariance remains a separate
zero-Frobenius score-plumbing check.

Oracle results explain which information a deployable estimator would need;
they can never win the benchmark table. Their paired improvements use circular
calendar blocks of $\lceil(T+42)/14\rceil$ origins.

## Frozen evaluation contract

### Information and dates

- Training targets: through 2013-12-31.
- Purged gap.
- Validation targets: 2015-07-01 through 2018-06-30.
- Purged gap.
- Test targets: 2020-01-01 through the latest 2026 observation.
- Forecast horizon: 42 trading days.
- No estimator, decay factor, shrinkage choice or model parameter may use test
  returns before its final evaluation.
- Estimator benchmark tables are computed on the untouched test rows only.
  Validation rows are evaluated only while selecting EWMA's half-life; scoring
  fitted estimators on training and purged-gap rows would add cost but no
  evidence about out-of-sample performance.

### Geometry scores

- Normalised projector loss at $d=1,3,6$.
- Complete-flag loss: mean of the three normalised cumulative losses.
- Mean, median and IQR.
- Count of sliding examples and fully non-overlapping target windows.

### Full-covariance scores

- Scale-normalised Frobenius error against the next-42-day realised covariance.
- Gaussian predictive log loss, with conditioning reported.
- Realised global-minimum-variance portfolio variance for both unconstrained
  and long-only weights.
- Forecast condition number and any numerical regularisation required.
- Top-$1/3/6$ Flag losses of each estimator as a mechanism diagnostic.

All pairwise improvements are reported as both absolute loss and percentage
skill relative to the named baseline. Because sliding forecast origins overlap,
uncertainty uses calendar blocks and the much smaller non-overlapping count is
reported openly; hundreds of sliding rows are not treated as hundreds of
independent observations.

## Model families — only after the benchmark suite is operational

| family | model | forecast slot changed |
|---|---|---|
| 3.1 | Global damping | $\widehat H_t^+=\alpha H_t^-$ |
| 3.2 | Layerwise damping | separate market/core/buffer coefficients |
| 4.1 | Transported tangent AR | a short transported velocity history |
| 5.x | Rich sequence / full-SPD models | free-form temporal representation, required to beat smaller models and Family 2 covariance estimators |

Model 3.1 is now complete. It scales every plane angle in the one ordered
orthogonal motion of the complete Flag by a panel-specific coefficient selected
only on validation complete-Flag loss. The four selected coefficients are
0.225, 0.100, 0.100 and 0.050 for S&P, Nikkei, DAX and CAC respectively. Test
complete-Flag skill versus Frozen is positive on all four panels and +1.61% on
an equal-market basis, but the model remains −62.91% behind Retained Window.

## Primary precedents

- Ledoit & Wolf, *A Well-Conditioned Estimator for Large-Dimensional
  Covariance Matrices* ([paper](https://ledoit.net/ole1a.pdf)).
- Ledoit & Wolf, *Nonlinear shrinkage estimation of large-dimensional
  covariance matrices* ([arXiv:1207.5322](https://arxiv.org/abs/1207.5322)).
- Ledoit & Wolf, *Quadratic Shrinkage for Large Covariance Matrices*, with the
  authors' [Python QIS implementation](https://github.com/pald22/covShrinkage/blob/main/QIS.py).
- Bongiorno & Challet, *Covariance matrix filtering with bootstrapped
  hierarchies* ([arXiv:2003.05807](https://arxiv.org/abs/2003.05807)). Its
  published comparison set is LW, QuEST, cross-validated eigenvalue shrinkage,
  HCAL and BAHC, scored on 42-day realised minimum-variance risk.
- RiskMetrics Group, *Risk Management: A Practical Guide*
  ([technical guide](https://www.msci.com/resources/research/technical_documentation/RMGuide.pdf)).
- Johansson, Ogut, Pelger, Schmelzer & Boyd, *A Simple Method for Predicting
  Covariance Matrices of Financial Returns*, with the authors'
  [paper and code](https://web.stanford.edu/~boyd/papers/cov_pred_finance.html).
