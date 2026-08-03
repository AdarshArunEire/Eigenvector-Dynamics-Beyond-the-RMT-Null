# Eigenvector Dynamics Beyond the RMT Null

Do the dominant eigenspaces of covariance matrices genuinely evolve, or does finite-window estimation make a static system merely appear to rotate?

This project builds and calibrates an eigenvector-overlap instrument, reproduces published real-market measurements, and tests whether the remaining rotation contains a signal worth forecasting and learning.

## Current position

**Stage 1 is complete. Stage 2 forecasting is next.**

Synthetic experiments established that excess overlap distance is a reliable measure of genuine rotation once it clears the sampling-noise floor. On S&P 500, Nikkei, DAX and CAC 40 data, Stage 1 then found a directional signal that was coherent across assets and could not be explained away as a by-product of within-window eigenvector shrinkage.

The final learning state is

$$
\mathcal F_t=
\left(Y_t^{(1)}\subset Y_t^{(3)}\subset Y_t^{(6)}\right)
\in\mathrm{Flag}(N;1,3,6),
$$

preserving the market direction, top-three core and six-dimensional collision buffer. The representation gate is cleared: yesterday's rotation direction contains information about the next rotation.

This is evidence for a forecastable label, where ML could improve out-of-sample covariance or portfolio risk. That is the Stage 2 question.

## Stage 2

The forecasting ladder is:

1. hold the current flag fixed;
2. repeat its previous velocity;
3. learn a damping coefficient;
4. learn separate market/core/buffer dynamics;
5. fit a transported tangent autoregression;
6. test richer sequence and full-SPD models only after they beat the simpler baselines.

Every model will be evaluated chronologically on untouched future blocks and compared with holding still, constant velocity and a rotationally invariant covariance estimator.

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
