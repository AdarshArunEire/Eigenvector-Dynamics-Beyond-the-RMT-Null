# Build notes — eigenvector dynamics beyond the RMT null

## Stage 1
*2026-08-01*

In this stage I build the eigenvector overlap instrument, and ensure the excess rotation reported matches 4 regimes. 

### Regime 1.1 — false positives

**What's being tested:** When fed two estimates of the same unchanging matrix, does the instrument correctly return $D_{emp} - D_{th} ≈ 0$ ?

**Setup:** First, a chosen spectrum with random orthonormal basis was invented. I held the basis fixed and drew two separate batches of returns from it, with nothing changed between. I took the eigenvectors of each batch's sample covariance, and measured overlap distance between the top blocks. My value is then compared with the noise formula's.

**Verdict:** Under 2000 days of data and 400 trials, the eigenvectors appeared to move. The distance came out at 0.0019, *not zero*, despite the underlying truth never changing. The formula predicted 0.00193 and I measured 0.001935 — a gap of 0.25%, which sits 0.28 standard errors from zero once averaged over the 400 trials.

### Regime 1.2  — magnitudes

**What's being tested:** If I measure the same top eigenvalue on two independent windows and square the difference, [Eq (9)](https://arxiv.org/abs/1108.4258) says it should average to $4\lambda^2/T$. Does it?

**Setup:** Same static world as before. Two independent batches, take the top 3 eigenvalues of each sample covariance, square the differences, average over 400 trials, compare against $4\lambda^2/T$ mode by mode.

**Verdict:** Passes.
- Mode 1: 1.234 measured against 1.25 predicted (−1.3%). 
- Mode 2: 0.187 against 0.200 (−6.5%). 
- Mode 3: 0.0791 against 0.0720 (+9.9%). 

All inside the 20% tolerance, and the agreement degrades as you go down the spectrum as lower modes sit closer to the bulk and the leading-order formula has less room.

### Regime 1.3 — what is the residual?

**What's being tested:** 1.1 says the leftover distance is the right size, it doesn't say if it's noise. If the pipeline manufactures a fixed amount of rotation, that would also look small. Real sampling noise must scale as 1/T; a bug would sit at a constant.

**Setup:** Same static world, same measurement, run at 1000 and 4000 days.

**Verdict:** I quadrupled the sample size and the measured distance fell by a factor of four, confirming the residual is sampling noise, not a bug.

### Regime 1.4 — the ceiling

**What's being tested:**: How far apart are two subspaces that have nothing to do with each other? 

**Setup:** Pick a random P-dimensional and a random Q-dimensional subspace in N dimensions. They still overlap somewhat by pure dimension-counting accident. $D_{RMT}$ is that accidental value, so reproduce the number the [paper quotes](https://arxiv.org/abs/1108.4258) in fig.2 — 0.83 at (P=5, Q=10, N=204).

**Verdict:** 0.8275. Reproduced.

**Note to keep in mind**: The density printed in the paper is not a probability density — it integrates to P/Q, not 1. So the paper's 0.83 is not the mean of $-\ln(\sigma)$, it's P/Q times it, while my `subspace_distance` computes the genuine mean. At (5, 10, 204) the two therefore differ by a factor of $Q/P = 2$. Had I compared my measured $D$ against 0.83 directly I'd have been out by exactly 2 — and Eq (10) *legitimately* carries a factor of 2 for an unrelated reason (two windows instead of one), so I'd have had a ready-made wrong explanation waiting.

### Regime 2.1 — is magnitude and direction conflated?

**What's being tested:** [Eq (10)](https://arxiv.org/abs/1108.4258) predicts that when eigenvectors are held perfectly fixed and only the eigenvalues move between windows, the measured distance equals

$$
\begin{aligned}
D(P,Q;s,t)
&=
-\frac{1}{2P}
\ln\left|\det\!\left((G^{s,t})^\dagger G^{s,t}\right)\right| \\[4pt]
&\approx
\frac{1}{2TP}
\left[
(\boldsymbol{\lambda}_{A}^{(s)})^{\mathsf T}
\mathbf{C}
\boldsymbol{\lambda}_{B}^{(s)}
+
(\boldsymbol{\lambda}_{A}^{(t)})^{\mathsf T}
\mathbf{C}
\boldsymbol{\lambda}_{B}^{(t)}
\right].
\end{aligned}
$$

i.e. the null absorbs eigenvalue movement instead of reporting it as genuine eigenvector rotation.

**Setup:** One Haar basis drawn once and reused for every window, forever. Each window gets its own spectrum: the top 3 eigenvalues multiplied by independent unit-mean lognormal jitter, $\sigma$ = 0.06, bulk untouched. N=40, P=3, Q=6, T=2000, 400 trials. $D_{th}$ is computed per trial from the two actual spectra of that trial.

**Verdict:** $D_{num}$ = 0.001955, $D_{th}$ = 0.001952, ratio = 1.0015, ratio pooled over 4000 trials = 1.005 ± 0.003. The residual is a half-percent, matching the order of regime 1's static 0.25% gap, consistent with the $O(1/T^2)$, and consistent with the same small positive bias regime 1 already showed. 

### Regime 2.2 — loss when using estimated eigenvalues 

**What's being tested:** How much you lose by feeding the null estimated eigenvalues instead of true ones. The [paper](https://arxiv.org/abs/1108.4258) waves the substitution *"Up to corrections of order T^(−3/2), one can replace in the above formulas the λˢ·ᵗ by their empirical estimates."* Since D is itself $O(1/T)$, that predicts a relative error of order $T^{-\frac{1}{2}}$.

**Setup:** Same fixed basis and jittered spectra as 2.1. Per trial, compute $D_{th}$ twice — once from the true $(\lambda^s, \lambda^t)$, once from the sample eigenvalues $(\hat\lambda^{s}, \hat\lambda^{t})$ — and take the ratio. First held N=40 and swept T. Then, because $D_{th}$ only ever touches $\lambda_i$ for $i<P$ and $\lambda_j$ for $j>Q$, re-ran it substituting **one end of that gap at a time** — estimated top block against true bulk, and true top block against estimated bulk — over a grid of N and $q = N/T$.

**Verdict:** My first sweep held N=40 and moved only T (1200 trials per row):

| T | q = N/T | est/true | bias | SE |
|------|------|----------|--------|-------|
| 250 | 0.16 | 0.9893 | −1.07% | 0.18% |
| 500 | 0.08 | 0.9969 | −0.31% | 0.12% |
| 1000 | 0.04 | 0.9978 | −0.22% | 0.09% |
| 2000 | 0.02 | 0.9993 | −0.07% | 0.06% |
| 4000 | 0.01 | 0.9996 | −0.04% | 0.04% |

Small, negative, dying fast — a clean law. *It isn't one, for two reasons*.

First, **with N fixed, $q$ and $1/T$ are the same variable**, so nothing in that table can tell me which of them governs. Second, when I varied N the bias **changed sign**. Splitting by which end of the gap is substituted shows why (600 trials each):

| N | q | T | top only | bulk only | both | sd per dataset |
|---|---|---|---|---|---|---|
| 20 | 0.2 | 100 | +0.4% | −5.9% | −5.5% | 13.0% |
| 40 | 0.2 | 200 | −1.1% | **+0.3%** | −0.8% | 9.7% |
| 80 | 0.2 | 400 | −2.0% | **+3.8%** | +1.6% | 6.7% |
| 160 | 0.2 | 800 | −2.9% | **+5.4%** | +2.1% | 4.7% |
| 20 | 0.6 | 33 | −1.7% | −20.1% | −21.3% | 17.3% |
| 40 | 0.6 | 67 | −5.4% | **+0.4%** | −5.5% | 15.3% |
| 80 | 0.6 | 133 | −8.2% | **+11.8%** | +0.9% | 12.6% |
| 160 | 0.6 | 267 | −8.3% | **+18.0%** | +5.8% | 8.9% |

There are **two mechanisms pulling opposite ways**:

- The **top block** gets pushed *up* by repulsion from the bulk. That widens $\lambda_i - \lambda_j$ and shrinks $D_{th}$. Negative essentially everywhere.
- The **bulk** spreads and its upper edge climbs *toward* the top block. That narrows the same gap and inflates $D_{th}$. Positive as soon as the bulk is dense, and it becomes the larger of the two.

And N=40 with this bulk sits almost exactly on the cancellation point — bulk term +0.3% at both $q$. **Every configuration in my first table was N=40 — null point of the grid.**

What actually drives it is the bulk edge, not N or $q$. Holding N=80, $q$=0.4, $\lambda_P = 6.0$ and changing only the bulk range:

| bulk range | est. bulk top (true) | bulk term |
|---|---|---|
| 2.0 → 0.4 | 3.21 (2.00) | +15.5% |
| 1.3 → 0.4 | 2.27 (1.30) | +7.6% |
| 1.3 → 1.1 | 2.86 (1.30) | +15.5% |
| 0.6 → 0.2 | 1.07 (0.60) | +0.9% |

A factor of 17 at fixed N and $q$, tracking how far the sampled bulk edge climbs toward $\lambda_P$. $\hat\lambda_Q$ against $\hat\lambda_P$ is visible on real data without knowing any true spectrum, so I can check my exposure *before* choosing P and Q. Choosing them for a wide gap is the control I have.

The `sd per dataset` column above is scatter on $D_{th}$ *alone*, and I never report $D_{th}$ alone. I report $D_{emp}/D_{th}$, and both come from the same returns, so their errors cancel. Over 1500 trials:

| N | q | $D_{emp}$ alone | ratio, true $\lambda$ | ratio, est $\hat\lambda$ | corr |
|---|---|---|---|---|---|
| 40 | 0.2 | 19.5% | 19.5% | **18.8%** | +0.27 |
| 80 | 0.4 | 13.4% | 13.4% | **12.5%** | +0.38 |
| 160 | 0.6 | 10.4% | 10.4% | **9.3%** | +0.46 |

Substituting the estimated spectrum adds **no** noise to the ratio — *it takes a little away*. A window that throws a large $D_{emp}$ throws a large $D_{th}(\text{est})$ too, and they move together. The ratio's noise is set by $D_{emp}$'s own sampling scatter, which I'd have whatever spectrum I feed the null, and which averages down over window pairs.

**The bias:** In the ratio the systematic part is +1.5%, −1.4%, −5.4% across those three rows, against a target effect of +50% to +200%. Not a threat to the sign of the result. Still a threat to $T^*$, because it is T-dependent and does *not* average away — and $T^*$ is found by scanning T.

One thing that does hold up: the jitter amplitude is nearly irrelevant. The bias moves only −1.16% → −1.33% as $\sigma$ goes 0.02 → 0.10, while changing the spectrum alone moves it 3.7–6.4×. It is a statement about spectral geometry, not about how much the eigenvalues move.

On the paper's claim: at fixed N the decay fits $T^{-1.34}$ against the $T^{-1/2}$ its bound implies, so the substitution is far better behaved than the paper needs. Given the sign flip I'd treat that exponent as describing one spectrum at one N, as opposed to a universal law.

### Regime 2.3 — where Eq (10) actually breaks

**What's being tested:** The paper hedges Eq (10) with "assuming the eigenvalues are varying sufficiently slowly with time". I wanted to know what that condition actually bites on.

**Setup:** It turns out it is *not* about movement between windows — 2.1 already showed Eq (10) is exact there for arbitrarily large between-window changes. It is about the spectrum moving *inside* one window. So: let C evolve under preset paths, and compare the measured $D_{emp}$ against $D_{th}$ built from the midpoint spectrum (which is what a whole-window estimate recovers). 400 trials per path.

**Verdict:**
| path | $1+\mathrm{CV}^2$ | measured $D_{emp}/D_{th}$ | err |
|---|---|---|---|
| flat (control) | 1.000 | 1.000 | 0.0% |
| linear ramp, h=0.6 | 1.120 | 1.116 | −0.4% |
| sinusoid, matched CV | 1.120 | 1.130 | +0.9% |
| step 25% high, matched CV | 1.170 | 1.180 | +0.9% |
| crisis: 20% of window at 4× variance | 1.562 | 1.584 | +1.4% |
| crisis: 10% of window at 9× variance | 2.778 | 2.832 | +2.0% |

The square shows up because a whole-window estimate recovers the time-average of the level, but its sampling noise is set by the time-average of the *square*, and those two disagree by exactly $1+\mathrm{CV}^2$. Shape-independent to within 2%. So the law is $1 + \mathrm{CV}^2$. This matters because **CV is measurable on real returns without knowing anything about eigenvectors** — it is just the variability of the market's variance level inside each window.

**Parked, tbd with real data.** I currently am not sure if estimating the vol path from returns is accurate, and both standardisation and simply factoring out $1+\mathrm{CV}^2$ rely on it. 