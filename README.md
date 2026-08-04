# Eigenvector Dynamics Beyond the RMT Null

Do the dominant eigenspaces of covariance matrices genuinely evolve, or does finite-window estimation make a static system merely appear to rotate?

This project builds and calibrates an eigenvector-overlap instrument, reproduces published real-market measurements, and tests whether the remaining rotation contains a signal worth forecasting and learning.

![Stage 1 partial-flag persistence across four equity markets](assets/stage1-flag-signal.png)

## Current position

**Stage 1**

Synthetic experiments established that excess overlap distance is a reliable measure of genuine rotation once it clears the sampling-noise floor. The first full-window test found coherent directional persistence on four equity markets, but a deletion-attribution test subsequently showed that the mean per-origin share of outgoing tangent energy aligned with observations already known to leave each rolling window is 39–45%.

The final learning state is

$$
\mathcal F_t=
\left(Y_t^{(1)}\subset Y_t^{(3)}\subset Y_t^{(6)}\right)
\in\mathrm{Flag}(N;1,3,6),
$$

preserving the market direction, top-three core and six-dimensional collision buffer. The corrected forecast starts from the Flag of the retained observations and predicts only what the next 42 unseen returns add. That residual direction passes both calendar and volatility-matched nulls on S&P 500 and in the equal-market aggregate; Nikkei is borderline before multiplicity correction, while DAX and CAC do not pass.

This leaves a narrower but honest forecastable label. Stage 2 asks whether its S&P-led residual signal can improve out-of-sample covariance or portfolio risk.

## Stage 2

The forecasting ladder is being rebuilt around one common primitive:

1. delete the 42 observations known to expire and form the retained Flag;
2. predict zero contribution from the unseen block;
3. learn a damped continuation of the preceding realised incoming-block effect;
4. learn separate market/core/buffer residual dynamics;
5. fit a transported residual-tangent autoregression;
6. test richer sequence and full-SPD models only after they beat the retained baseline.

Every geometric model will receive the same retained state and be evaluated against the actual future Flag. Positive skill versus Retained Window is therefore evidence about unseen returns rather than credit for deterministic window turnover. Reconstructed forecasts must still face established full-covariance estimators.

The original global-damping fit improved on holding still by 1.61% but lost to Retained Window by 62.9%. Regime 4.9 revealed that this was the wrong primitive, so that fit is retained as a diagnostic rather than claimed as a model of unseen-return dynamics.

## Repository structure

| Path | Purpose |
|---|---|
| [`BUILDNOTES.md`](BUILDNOTES.md) | chronological notebook: what was tested, the setup and what happened |
| [`stage1/README.md`](stage1/README.md) | complete Stage 1 argument and results |
| [`stage2/README.md`](stage2/README.md) | forecasting targets, baselines and evaluation plan |
| [`PRIOR_ART.md`](PRIOR_ART.md) | literature review and novelty boundary |
| [`stage1/BIBLIOGRAPHY.md`](stage1/BIBLIOGRAPHY.md) | Stage 1 references |
| [`src/`](src) | RMT, data, geometry, ERSE and synthetic utilities |
| [`scripts/`](scripts) | reproducible experiment runners |
| [`tests/`](tests) | numerical, statistical and invariance tests |
| `data/` | cached inputs and panel metadata |
| `results/` | generated tables, null distributions and figures |

## Run

```bash
python -m pytest -q
```

For the complete experimental record, start with [`BUILDNOTES.md`](BUILDNOTES.md). For the polished scientific argument, read [`stage1/README.md`](stage1/README.md).
