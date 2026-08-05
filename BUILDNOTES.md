# Build notes — eigenvector dynamics beyond the RMT null

## Note on the predecessor eigenvector dynamics paper

Allez & Bouchaud wrote this twice and arXiv does not link the two, because they are separate submissions rather than versions of one. I worked from the short one for most of both dates without knowing the long one existed.
<details>
<summary>Click here to expand</summary>

- **[arXiv:1203.6228](https://arxiv.org/abs/1203.6228)** — *"Eigenvector dynamics: general
  theory and some applications"*, March 2012. The full paper. **This is the reference to
  use.** Section 6 covers four indices (SP500 N=500, Nikkei 204, DAX 30, CAC 40 39, all
  2000–2010), states the correlation normalisation outright in footnote 11, and fixes the
  window at $T = N$.
- **[arXiv:1108.4258](https://arxiv.org/abs/1108.4258)** — *"Eigenvector dynamics: theory
  and some applications"*, August 2011. The 4-page letter. Nikkei only. Cited below only
  where a number appears in it and not in the long version.

Equation and figure numbers differ between them. The map, for reading my earlier notes:

| letter | full paper |
|---|---|
| Eq (3), GOE distance | Eq (3.4) |
| Eq (6), $E = C + \mathcal{E}$ | Eq (4.1) |
| Eq (7), sample-vs-true $D(P,Q)$ | §4, display following Eq (4.5) |
| Eq (9), eigenvalue variogram $4\lambda_i^2/T$ | §4, display near the end |
| Eq (10), two-sample $D(P,Q;s,t)$ | **Eq (6.1)** |
| $D_{RMT}$ display, p.2, unnumbered | §2, also unnumbered |
| Fig. 1, variograms | Fig. 7 |
| Fig. 2 left, $D$ vs $\tau$ | Fig. 8 |
| Fig. 2 right, $D$ vs $T$ | Fig. 9 |

The two §4 entries are cited by section rather than number because the numbering could not
be confirmed from the text I extracted — check them against the PDF before quoting.

Both empirical figures plot $D$ itself, on a $0$–$0.45$ axis, never a ratio:

- **Fig. 8** — $D_{th}$, $D_{num}$, $D_{emp}$ against $\tau$ at $T = N$, $P=5$, $Q=10$, four
  indices. $D_{emp}$ rises from zero and keeps rising for the Nikkei and the SPX; for the
  CAC 40 and the DAX it rises and then holds flat. $D_{num}$ is their fixed-eigenvector
  control and traces the overlap artifact — from zero up to the $D_{th}$ line at $\tau = T$.
- **Fig. 9** — $D_{emp}(\tau{=}T)$ and $D_{th}(\tau{=}T)$ against $T$. The feature is a
  **minimum**: *"the initial decline as T increases follows from reducing the measurement
  noise. However, when T becomes very large, the 'true' evolution of the eigenvectors is
  being felt, and leads to an increase of $D_{emp}$."* §6 gives one $T^*$ per index —
  Nikkei 600, SP500 700, DAX 450, CAC 40 400 days. The "around two years ($T^* = 500$
  days)" in the caption is a round-number summary of those four.

</details>

## Stage 1 — instrument
*2026-08-01*

In this stage I build the eigenvector overlap instrument, and ensure the excess rotation reported matches 4 regimes.

### Regime 1.1 — false positives

**What's being tested:** When fed two estimates of the same unchanging matrix, does the instrument correctly return $D_{emp} - D_{th} ≈ 0$ ?

**Setup:** First, a chosen spectrum with random orthonormal basis was invented. I held the basis fixed and drew two separate batches of returns from it, with nothing changed between. I took the eigenvectors of each batch's sample covariance, and measured overlap distance between the top blocks. My value is then compared with the noise formula's.

**Verdict:** Under 2000 days of data and 400 trials, the eigenvectors appeared to move. The distance came out at 0.0019, *not zero*, despite the underlying truth never changing. The formula predicted 0.00193 and I measured 0.001935 — a gap of 0.25%, which sits 0.28 standard errors from zero once averaged over the 400 trials.

### Regime 1.2 — magnitudes

**What's being tested:** If I measure the same top eigenvalue on two independent windows and square the difference, [the eigenvalue variogram in §4](https://arxiv.org/abs/1203.6228) says it should average to $4\lambda^2/T$. Does it?

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

**What's being tested:** How far apart are two subspaces that have nothing to do with each other?

**Setup:** Pick a random P-dimensional and a random Q-dimensional subspace in N dimensions. They still overlap somewhat by pure dimension-counting accident. $D_{RMT}$ is that accidental value, so reproduce the number the [paper quotes](https://arxiv.org/abs/1108.4258) in fig.2 — 0.83 at (P=5, Q=10, N=204).

**Verdict:** 0.8275. Reproduced.

**Note to keep in mind**: The density printed in the paper is not a probability density — it integrates to P/Q, not 1. So the paper's 0.83 is not the mean of $-\ln(\sigma)$, it's P/Q times it, while my `subspace_distance` computes the genuine mean. At (5, 10, 204) the two therefore differ by a factor of $Q/P = 2$. Had I compared my measured $D$ against 0.83 directly I'd have been out by exactly 2 — and Eq (10) *legitimately* carries a factor of 2 for an unrelated reason (two windows instead of one), so I'd have had a ready-made wrong explanation waiting.

### Regime 1.5 — the null under fat tails

**What's being tested:** Regimes 1.1–1.3 validate the null on Gaussian returns, which is the $\nu \to \infty$ corner of it. Eq (4.7) says that for multivariate Student returns with $\nu$ degrees of freedom the whole formula is multiplied by $\frac{\nu-2}{\nu-4}$ and nothing else changes. Real equity kurtosis is 7–11, so this is not a corner case — it is the case.

Where it comes from is worth stating, because it is not an extra assumption bolted on. The null descends from Eq (4.2), the covariance of the *errors* in the sample covariance matrix, and that is a **fourth moment of returns**. For Gaussians the four-way expectation factors into pairs and gives the clean form. For anything fatter it does not, and the error covariance is larger. The entire null is a claim about how much fourth moments bounce around.

**Setup:** Static world, one Haar basis, spectrum $[12, 7, 4]$ over a bulk. N=40, P=3, Q=6, T=500, 500 trials per row. Returns drawn as $r_t = z_t\sqrt{(\nu-2)/w_t}$ with $w_t \sim \chi^2_\nu$ — a Gaussian whose scale is redrawn every day and **shared by the whole cross-section**. $D_{th}$ uses the true spectrum and the plain Gaussian formula throughout, so the only thing that can move the ratio is the distribution.

**Verdict:**

| $\nu$ | kurtosis | predicted $\frac{\nu-2}{\nu-4}$ | measured $D_{emp}/D_{th}$ | meas/pred |
|---|---|---|---|---|
| ∞ | 3.0 | 1.000 | 1.025 ± 0.008 | 1.025 |
| 20 | 3.4 | 1.125 | 1.150 ± 0.010 | 1.023 |
| 12 | 3.8 | 1.250 | 1.282 ± 0.011 | 1.026 |
| 8 | 4.5 | 1.500 | 1.507 ± 0.014 | 1.005 |
| 6 | 6.0 | 2.000 | 1.881 ± 0.023 | 0.940 |

Eq (4.7) carries the whole effect. The 2.5% overshoot is the same small positive bias regimes 1.1 and 2.1 already showed; the 6% undershoot at $\nu$=6 is where perturbation theory starts to feel a kurtosis of 6.

**The part that matters more.** A multivariate Student's scale factor is *common to every name on a given day*. That is exactly what `standardise` divides out. So it should be removable without ever estimating $\nu$ — and it is:

| N | $2/N$ | Gaussian, standardised | $\nu$=6, standardised | difference |
|---|---|---|---|---|
| 40 | 0.050 | 1.137 | 1.138 | **+0.001** |
| 80 | 0.025 | 1.063 | 1.067 | **+0.004** |
| 160 | 0.013 | 1.019 | 1.017 | **−0.003** |
| 320 | 0.006 | 0.985 | 0.987 | **+0.001** |

The $\nu$-dependence is gone — **completely, at every N**, against 1.88 unstandardised. What is left is an inflation that a *Gaussian* world shows too, so it is an artefact of standardising rather than anything to do with tails, and it decays with N: 14% at N=40, 2% at N=160, and slightly negative by N=320. At my real panels (N=175 US, N=132 Nikkei) it is 1–3%.

So there are two routes and the second is better. Fit $\nu$ and multiply the null by $\frac{\nu-2}{\nu-4}$ — which requires choosing $\nu$, and the paper chooses it by matching curves at small T rather than from moments. Or standardise at window=1 and keep the Gaussian null, which needs no free parameter at all and leaves a bias I can measure at my own N. **I will take the second.**

### Regime 2.1 — is magnitude and direction conflated?

**What's being tested:** [Eq (6.1)](https://arxiv.org/abs/1203.6228) predicts that when eigenvectors are held perfectly fixed and only the eigenvalues move between windows, the measured distance is $D$:

Let $E_s, E_t$ be the sample covariances from the two windows, $U_s \in \mathbb{R}^{N \times P}$
the leading $P$ eigenvectors of $E_s$, and $V_t \in \mathbb{R}^{N \times Q}$ the leading $Q$
of $E_t$. The overlap matrix is $G^{s,t} = V_t^{\mathsf T} U_s \in \mathbb{R}^{Q \times P}$.

$$
\begin{aligned}
D(P,Q;s,t) &= -\frac{1}{2P}\,\mathbb{E}\left[\ln\det\left((G^{s,t})^{\mathsf T} G^{s,t}\right)\right] \\
&\approx \frac{1}{2TP}\sum_{i=1}^{P}\sum_{j=Q+1}^{N}\left[\frac{\lambda_i^s\lambda_j^s}{(\lambda_i^s-\lambda_j^s)^2}+\frac{\lambda_i^t\lambda_j^t}{(\lambda_i^t-\lambda_j^t)^2}\right]
\end{aligned}
$$

i.e. the null absorbs eigenvalue movement instead of reporting it as genuine eigenvector rotation.

**Setup:** One Haar basis drawn once and reused for every window, forever. Each window gets its own spectrum: the top 3 eigenvalues multiplied by independent unit-mean lognormal jitter, $\sigma$ = 0.06, bulk untouched. N=40, P=3, Q=6, T=2000, 400 trials. $D_{th}$ is computed per trial from the two actual spectra of that trial.

**Verdict:** $D_{num}$ = 0.001955, $D_{th}$ = 0.001952, ratio = 1.0015, ratio pooled over 4000 trials = 1.005 ± 0.003. The residual is a half-percent, matching the order of regime 1's static 0.25% gap, consistent with the $O(1/T^2)$, and consistent with the same small positive bias regime 1 already showed.

### Regime 2.2 — loss when using estimated eigenvalues

**What's being tested:** How much you lose by feeding the null estimated eigenvalues instead of true ones. The [paper](https://arxiv.org/abs/1203.6228) waves the substitution *"Up to corrections of order T^(−3/2), one can replace in the above formulas the λˢ·ᵗ by their empirical estimates."* Since D is itself $O(1/T)$, that predicts a relative error of order $T^{-\frac{1}{2}}$.

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

**Reframing, after regime 1.5.** $1 + \mathrm{CV}^2$ and Eq (4.7)'s $\frac{\nu-2}{\nu-4}$ are not two corrections. They are one law:

$$\text{inflation} \;=\; \frac{\mathbb{E}[c^2]}{\mathbb{E}[c]^2}$$

for a variance level $c$ that is not constant. Write a Student's scale as $s = \nu/w$ with $w \sim \chi^2_\nu$. Then $\mathbb{E}[s] = \frac{\nu}{\nu-2}$ and $\mathbb{E}[s^2] = \frac{\nu^2}{(\nu-2)(\nu-4)}$, so

$$\frac{\mathbb{E}[s^2]}{\mathbb{E}[s]^2} \;=\; \frac{\nu-2}{\nu-4}$$

exactly. Same quantity, different $c$.

What differs is *which* variance level, and that is a real physical distinction:

| | what moves | timescale |
|---|---|---|
| $\frac{\nu-2}{\nu-4}$ | scale redrawn independently **every day** | none, memoryless |
| $1 + \mathrm{CV}^2$ | scale **drifts** across the window | slow, persistent |

Real returns have both — fat tails *and* volatility clustering — and if the two are independent the inflations **multiply**. So this regime is not superseded, it is half of a pair. The trap is double-counting: `cv_squared` runs on a 21-day rolling path, which smooths the daily component away and leaves only the slow part, which is what makes multiplying legitimate. Measure CV² at window=1 and it would capture both, and multiplying would then be wrong.

The practical consequence is that one remedy covers both. `standardise` divides each day by that day's cross-sectional volatility, which is the common scale in either mechanism — regime 1.5 shows it removes the fat-tail half exactly, and this regime showed it takes a 183% drift inflation down to 1.3%. One knob, both problems, no $\nu$ to fit.

### Regime 3.1 — is subtracting $D_{th}$ legitimate?

**What's being tested:** The method is to compute $\mathrm{Excess} = D_{emp} - D_{th}$, and I want to confirm it is a real signal of rotation. Specifically, does additivity hold? Do noise-rotation and real-rotation combine by simple addition?

**Setup:** A Givens rotation rotates modes $i$ and $j$ by angle $\theta$, so the new eigenvector $i$ is $\cos(\theta)q_i + \sin(\theta)q_j$. That gives $D_{inject} = -\ln(\cos(\theta))/P$. What I want to see is $D_{emp} \approx D_{th} + D_{inject}(\theta)$, and where additivity breaks: it must, as $\ln\cos(\theta)$ diverges as $\theta \rightarrow \pi/2$.

**Verdict:**

| $\theta$ | $D_{inject}$ | predicted | $D_{emp}$ | ratio | recovery |
|---|---|---|---|---|---|
| 0.00 | — | 0.001930 | 0.001967 | 1.019 | — |
| 0.05 | 0.000417 | 0.002347 | 0.002395 | 1.021 | 1.115 |
| 0.10 | 0.001669 | 0.003599 | 0.003656 | 1.016 | 1.034 |
| 0.20 | 0.006712 | 0.008642 | 0.008710 | 1.008 | 1.010 |
| 0.30 | 0.015231 | 0.017160 | 0.017232 | 1.004 | 1.005 |
| 0.80 | 0.120464 | 0.122394 | 0.122257 | 0.999 | 0.999 |

$D_{emp} = D_{th} + D_{inject}$ to within 2% everywhere, tightening to 0.1% once the injected rotation exceeds the noise floor. The residual is a roughly constant absolute offset, about 3% of $D_{th}$.

**Power curve.** Detection threshold on a *single* pair of windows: 5% false positives (threshold = 95th percentile of the θ=0 distribution) at 80% power. Because additivity holds, the required injection is just the gap between the 95th and 20th percentiles of the null, so only the null needs simulating. N=40, P=3, Q=6, 700 trials per cell.

| T | $\nu=\infty$ | $\nu$=12 | $\nu$=8 | $\nu$=6 | $\nu$=6, standardised |
|---|---|---|---|---|---|
| 250 | 15.8° | 18.1° | 21.4° | 28.2° | **17.1°** |
| 500 | 11.2° | 12.7° | 14.6° | 17.8° | **11.6°** |
| 1000 | 7.8° | 8.9° | 9.7° | 12.2° | **8.2°** |

The Gaussian column scales as $\theta_{min}\sqrt{T} = 250, 249, 247$ degrees — clean $1/\sqrt{T}$, as expected from $D_{inject} \approx \theta^2/2P$ against a floor $\propto 1/T$. It sits about 35% above the 3.2/$\sqrt{T}$ I reported before, because that number came from scanning θ directly while this uses the percentile shortcut; the shortcut is the more conservative of the two and I am keeping it.

**Fat tails cost more than Eq (4.7) predicts.** Ratio to the Gaussian column against the predicted $\sqrt{\frac{\nu-2}{\nu-4}}$:

| T | $\nu$=12 | $\nu$=8 | $\nu$=6 |
|---|---|---|---|
| 250 | 1.143 | 1.354 | 1.778 |
| 500 | 1.136 | 1.312 | 1.595 |
| 1000 | 1.146 | 1.245 | 1.564 |
| **predicted** | **1.118** | **1.225** | **1.414** |

Measured exceeds predicted, and the gap widens as $\nu$ falls — 26% too high at $\nu$=6. The reason is structural: **Eq (4.7) rescales the mean of the null, but a detection threshold lives in its upper tail, and fat tails fatten the tail faster than the mean.** So multiplying $D_{th}$ by $\frac{\nu-2}{\nu-4}$ is the right correction for a reported ratio and the *wrong* one for a threshold. Anyone using the null to decide whether a single window pair has rotated needs the quantiles, not the mean.

Standardising at window=1 recovers it: 1.078, 1.041, 1.057 against Gaussian, versus 1.56–1.78 unstandardised. Same conclusion as regime 1.5 — the remedy is one knob and it works on the tail as well as the mean.

**So the honest headline is 11°, not 8°.** At two years of daily data on Gaussian returns I can detect an 11° rotation on a single pair; on ν=6 returns, 18°; standardised, back to 12°. Real panels sit near ν=5–7. Worth holding against regime 4's measured excess of about 18°.

## Stage 1, continued — real panels
*2026-08-02*

Yesterday I calibrated the instrument against worlds where I knew the answer. Today points it at real returns, and at [Figs. 7, 8 and 9](https://arxiv.org/abs/1203.6228) — the only published numbers I can check myself against. Four panels, built to the paper's own universes: SP500, Nikkei, DAX and CAC 40, all 2000–2010, all at the paper's window rule $T = N$.

### Regime 4.1 — the eigenvalue variogram

**What's being tested:** [Eq (4.8)](https://arxiv.org/abs/1203.6228) says that if $C$ never moved, two independent windows would disagree about its $i$-th eigenvalue by

$$\left\langle\left(\lambda_i^s - \lambda_i^t\right)^2\right\rangle_{|t-s|>T} \approx \frac{4\lambda_i^2}{T}$$

and nothing more. Regime 1.2 confirmed that synthetically. This is its real-data counterpart, and it is the right first thing to run because **it touches eigenvalues only** — no $P$, no $Q$, no subspace machinery, so it cannot be contaminated by the block-size question I have just reopened.

**Setup:** All four panels, window $T = N$, step $\max(5, T/25)$, modes 1–3. Correlation matrices throughout. Run twice, raw and after `standardise(window=1)`, because Eq (4.8) carries the same $\frac{\nu-2}{\nu-4}$ factor as Eq (4.7) and regime 1.5 says that factor is removable without fitting $\nu$. Short lags are kept deliberately.

**The panels:**

| | paper's N | mine | days | median kurtosis | zero returns | $\lambda_1/N$ |
|---|---|---|---|---|---|---|
| SP500 | 500 | 357 | 2761 | 11.7 | 1.4% | 0.316 |
| Nikkei | 204 | 132 | 2797 | 7.3 | 8.5% | 0.379 |
| DAX | 30 | **29** | 2832 | 10.3 | 3.5% | 0.383 |
| CAC 40 | 39 | 26 | 2842 | 8.4 | 3.2% | 0.425 |

The paper's Nikkei has $\lambda_1/N = 73/204 = 0.358$. Four independently-built panels land at 0.316–0.425 around it, which is the only external check I have that these universes are the right shape, and it passes.

**Verdict, part one: the windowing is correct.** The paper states the empirical curve *"starts from 0 for $\tau=0$ and increases to reach the stationary noise level at time $\tau = T$ ... simply due to the overlapping between the sliding periods"*. That is a known artifact with a known shape, and reproducing it is a free correctness check on the pairing code. Nikkei, mode 1:

| $\tau/T$ | 0.04 | 0.30 | 0.57 | 0.83 | 1.10 | 1.36 | 1.63 | 1.89 |
|---|---|---|---|---|---|---|---|---|
| ratio to null | 0.0× | 0.5× | **1.0×** | 1.6× | 2.1× | 2.4× | 2.7× | 3.0× |

Climbs from zero, crosses the null around $\tau \approx T/2$, flattens past $\tau = T$. Every panel does the same. If this had come out flat or started high I would have had a pairing bug quietly poisoning every regime downstream.

**Verdict, part two: the eigenvalues move, but only convincingly in one market.** Ratio to $4\lambda_i^2/T$ at the first lag past $T$:

| | $T=N$ | mode 1 raw | mode 1 std | mode 2 raw | mode 2 std |
|---|---|---|---|---|---|
| **SP500** | 357 | **10.6×** | **3.5×** | 10.4× | 7.0× |
| Nikkei | 132 | 2.0× | 0.7× | 3.2× | 2.7× |
| DAX | 29 | 0.6× | 0.3× | 1.3× | 0.7× |
| CAC 40 | 26 | 0.5× | 0.2× | 1.2× | 0.8× |

The S&P 500 is the paper's claim in its strong form — mode 1 sits at ten times the noise floor and is still climbing at 20× by $\tau = 2T$. Measurement noise cannot produce that.

**Standardising cuts every ratio by a factor of two to three**, uniformly across markets and modes. That is exactly what regime 1.5 predicts: part of what reads as eigenvalue evolution is fat tails inflating the floor rather than the level moving. It is also the first time that conclusion has been tested outside a synthetic world, and it holds. The cost is that **the Nikkei's mode-1 evidence does not survive it** — 2.0× becomes 0.7×, i.e. below the null. Only the S&P 500 clears decisively at 3.5×.

**Verdict, part three: the small markets sit *below* their own null, and I do not yet know why.** DAX 0.6× and CAC 0.5× raw, 0.3× and 0.2× standardised. Below 1.0× means the measured dispersion is *smaller* than sampling noise alone should produce, which is not a statement about the market — it is a statement that Eq (4.8) is wrong in this corner.

The obvious suspect is $T$. The paper's own rule puts DAX at a **29-day window**, and Eq (4.8) comes from perturbation theory that assumes $T$ large enough for the expansion to hold. Every calibration I have is at $T \geq 250$. The paper half-notices this — it observes that for DAX and CAC the top eigenvalue *"does not evolve too much during the following (non overlapping) period $\tau \in [T; 250]$ days"* — but reports it as a property of small markets rather than of small windows, and its Fig. 7 for those two indices is drawn on axes an order of magnitude below the Nikkei's.

Two readings: either 29 days is below the floor where any of this applies, in which case the small panels are unusable at $T=N$, or the effect is about $N$ rather than $T$, in which case running DAX at $T = 250$ should pull the ratio back above 1.

**One fix along the way.** `standardise` divided by zero on days where every name in the panel returned exactly zero — exchange holidays that leaked into the union date index on the Tokyo and Paris panels, which is the same phantom-date problem that inflates their zero-return fractions. Those days now hold unit scale instead of propagating NaN through the entire panel.

**To consider.** All four panels are conditioned on survival to 2026: the ticker lists are current index membership plus whatever historical names I could recover, so companies that vanished mid-decade are absent from the candidate list before any filter sees them. CAC 40 at 26 of 39 is the worst of it, mostly Yahoo having dropped delisted Paris lines entirely.

### Regime 4.2 — sliding window $T$: excess vs $\tau$

**What's being tested:** Is C genuinely evolving? Does the excess grow with $\tau$?

**Setup:** Slide a window of T days along the panel, and consider two windows $\tau$ days apart, then plot the excess rotation against $\tau$.

**Verdict:** $D_{emp}/D_{th}$, raw, $P=3$, $Q=6$, $T=N$.

| $\tau/T$ | S&P 500 ($T$=357) | Nikkei ($T$=132) | DAX ($T$=29) | CAC 40 ($T$=26) |
|---|---|---|---|---|
| 0.10 | 0.19× | 0.14× | — | — |
| 0.25 | 0.47× | 0.38× | 0.45× | 0.48× |
| 0.50 | 0.98× | 0.83× | 1.04× | 1.15× |
| 0.75 | 1.53× | 1.29× | 1.75× | 1.95× |
| 1.00 | 2.11× | 1.78× | 2.49× | 2.79× |
| 1.25 | 2.35× | 1.84× | 2.65× | 2.90× |
| 1.50 | 2.55× | 1.91× | 2.68× | 2.84× |
| 1.75 | 2.79× | 1.97× | 2.70× | 2.81× |
| 2.00 | 3.12× | 2.00× | 2.72× | 2.82× |

**S&P 500 is a clean result:**

| | intercept | slope /100d | $r$ | flat frac | reading |
|---|---|---|---|---|---|
| raw | −0.019 | +0.0172 | +0.996 | −0.09 | evolution |
| standardised | −0.029 | +0.0124 | +0.988 | −0.19 | evolution |

The windowing check passes on all four panels. $D_{emp}$ climbs from ~0, crosses $D_{th}$ at $\tau \approx T/2$, flattens past $\tau = T$

### Regime 4.3 — what window $T$ shows the rotation most clearly?

**What's being tested:** [Fig. 9](https://arxiv.org/abs/1203.6228) plots $D_{emp}(\tau{=}T)$ against $T$ and reads a **minimum** off it. Two effects set the shape. A short $T$ is a noisy estimate and the floor dominates, so $D_{emp}$ sits high. A long $T$ spans enough calendar time that the window's own contents move, so $D_{emp}$ climbs again. $T^*$ is the crossover — the paper's *"optimal time scale to measure the empirical eigenspaces"*. Their §6 gives one per index: Nikkei 600, SP500 700, DAX 450, CAC 40 400 days.

The ratio $D_{emp}/D_{th}$ is the wrong axis for this and I record it only because it falls out of the same run. $D_{th} \propto 1/T$ collapses while $D_{emp}$ saturates against the $D_{RMT}$ ceiling, so the ratio climbs monotonically on every panel for reasons that have nothing to do with the market — on the Nikkei $D_{emp}$ halves over the sweep while $D_{th}$ falls by 9×. Any $T^*$ read off a ratio is a statement about the denominator.

**Setup:** Sweep $T$ with $\tau$ tied to it, comparing two back-to-back windows $(\tau = T)$, at the paper's own blocks $P=5$, $Q=10$. Rank floor $T \geq N$ enforced: below it $q = N/T > 1$, the sample covariance is singular, and the Eq (6.1) sum over $j > Q$ runs across exactly-zero eigenvalues. Independent pair count $\mathrm{total}/2T$ recorded at every point.

**Verdict: the minimum is there on all four panels.**

| | their $N$ | mine | their $T^*$ | my $\arg\min D_{emp}$ | my $\min D_{emp}$ | indep pairs there |
|---|---|---|---|---|---|---|
| Nikkei | 204 | 132 | 600 | 793 | 0.1228 | 1.8 |
| SP500 | 500 | 357 | 700 | 999 | 0.0935 | 1.4 |
| CAC 40 | 39 | 26 | 400 | 529 | 0.1643 | 2.7 |
| DAX | 30 | 29 | 450 | 1032 | 0.1062 | 1.4 |

**The levels agree where I can read theirs off the page.** Their Nikkei minimum sits at about 0.128 against my 0.1228, and their flat $D_{th}$ line at about 0.145 against my 0.1399 at $T = 204$. Their $D_{th}$ benchmarks for the CAC 40 and the DAX rise before they fall, peaking near $T \approx 50$; mine do the same, which is the small-$N$ bulk edge and not something either of us put in by hand.

**Every minimum of mine sits later than theirs, and every panel of mine is smaller than theirs** — 132 against 204, 357 against 500, 26 against 39, 29 against 30. Fewer names is a noisier estimate at fixed $T$, so the noise-dominated arm runs longer before the evolution term overtakes it, which puts the crossover further out. The DAX is the extreme of it at 1032 against 450, and it is also the panel where I recover 29 of 30 names but the curve past $T \approx 800$ is visibly ragged.

**The existence of the U is solid; the location of its floor is not.** Independent pairs are below 3 at every one of my four minima, so $\arg\min$ is read off very little. The U itself is built from the whole curve rather than from its floor, and it survives on all four panels including two where regime 4.1 could not clear the eigenvalue null at all.

**Their fat-tail benchmark is a fitted $\nu$.** The Fig. 9 dotted blue line is $D_{th}$ under multivariate Student returns with $\nu$ *"chosen equal to 5.5 for the CAC40 and DAX indexes and to 18 for the Nikkei index"* — three panels, two hand-picked values, no procedure stated. That is the free parameter regime 1.5 removes: standardising at window=1 kills the $\nu$-dependence completely at every $N$ tested, with nothing to fit.

Plots: `results/regime4_3_fig9_all_panels_P5Q10.png` against `results/paper_1203.6228_fig9.png`.

### Regime 4.4 — Is the next tangent direction forecastable?

I sure hope so. 

**What's being tested:** The next stage needs a geometry for learning. I intially selected the Grassmann manifold $\mathrm{Gr}(N,P)$. If the eigenspace just moved in one direction, does its next movement tend to continue in that direction?

**Setup:** At the current subspace $Y_t$, construct 
$$H_t^-=-\mathrm{Log}_{Y_t}(Y_{t-h}),\qquad
H_t^+=\mathrm{Log}_{Y_t}(Y_{t+h}).$$
These are incoming and outgoing velocities in the same tangent space. That gives us:
- The learnability gate: cosine similarity between $H_t^-$ and $H_t^+$.
- The mandatory non-ML forecast: $\widehat Y_{t+h}=\mathrm{Exp}_{Y_t}(H_t^-)$.

**Verdict:** Yesterday’s rotation direction contains directional information about the next rotation. However, “Repeat yesterday’s entire rotation” is a terrible forecast everywhere. 

| Panel | Direction result | $\cos(H_t^-,H_t^+)>0$ | Repeat previous rotation |
|---|---:|---:|---:|
| S&P 500 | **exploratory** | 79.8% | **47.5% worse** |
| Nikkei | **$p=0.01$** | 72.2% | **67.1% worse** |
| DAX | **$p=0.01$** | 63.6% | **56.2% worse** |
| CAC 40 | **$p=0.02$** | 61.6% | **59.9% worse** |

### Regime 4.5 — Is this signal coherent across many assets?

**What is tested:** Is the movement shared across many assets, or merely the aggregate of independently changing company betas?

**Setup:** Extract asset-level tangent changes, destroy cross-asset timing independently, restore the correct geometric constraint, and compare the common component against 999 such histories.

**Verdict:** The observed tangent changes are more synchronised across companies than they would be if every company retained its own loading dynamics but changed at unrelated times.

| Panel | Common movement | Null expectation | Breadth |
|---|---:|---:|---|
| S&P 500 | 24.36% | 6.07% | about 125/357 names |
| Nikkei | 13.71% | 5.30% | about 41/131 |
| DAX | 17.56% | 8.63% | about 9/29 |
| CAC 40 | 14.03% | 10.10% | about 6/23 |

### Regime 4.6 — Are the rotations a variant of shrinkage?

**What is tested:** Does the temporal rotation signal only follow [ERSE's](https://arxiv.org/html/2507.01545) within-window correction? ERSE is developed for all-positive correlation matrices and uses eigenvector rotations whose recomputed pairwise Rayleigh eigenvalues move towards one another, making it a relevant shrinkage comparator.

**Setup:** Remove the ERSE direction and ask whether yesterday's residual direction still helps predict tomorrow's residual direction.

**Verdict:** Projecting out the ERSE direction removes almost nothing.

| Panel | Signal attributed to ERSE | Original cosine | After removing ERSE |
|---|---:|---:|---|
| S&P 500 | 0.05% | 0.2019 | 0.2021 |
| Nikkei | 0.06% | 0.1158 | 0.1161 |
| DAX | 0.92% | 0.0973 | 0.1009 |
| CAC 40 | 0.22% | 0.0830 | 0.0830 |

### Regime 4.7 — is the information contained in FLAG just as valid
**What is tested:** A flag contains three spaces: 
$$
Y_t^{(1)}\in\mathrm{Gr}(N,1),\qquad
Y_t^{(3)}\in\mathrm{Gr}(N,3),\qquad
Y_t^{(6)}\in\mathrm{Gr}(N,6),
$$
Such that
$$
\mathcal F_t=
\left(Y_t^{(1)},Y_t^{(3)},Y_t^{(6)}\right)
\in\mathrm{Flag}(N;1,3,6).
$$
It serves as a richer geometry that preserves how the market mode directions are organised. Does this new geometry satisfy regimes 4.4, 4.5 and 4.6? 

**Setup:** Reuse the exact 4.4-4.6 panels, horizons and nulls, computing the top 6 $\mathrm{Gr}(N,6)$. Slice into the three $Y$s, combine the results into one nested-projector flag tangent, and weigh each accordingly. Keep the Grassman logarithms to preserve lower-dim information. 

**Verdict:** 
| Panel | $\mathrm{Gr}(N,3)$ cosine | $\mathrm{Gr}(N,3)$ $p$ | Flag cosine | Flag $p$, raw / Holm |
|---|---:|---:|---:|---:|
| S&P 500 | 0.2019 | 0.0476 | 0.1545 | 0.0476 / 0.143 |
| Nikkei | 0.1158 | 0.010 | 0.0706 | 0.010 / 0.030 |
| DAX | 0.0973 | 0.010 | 0.0561 | 0.020 / 0.040 |
| CAC 40 | 0.0830 | 0.020 | 0.0622 | 0.020 / 0.040 |

The lower flag cosine is expected because it incorporates the weaker outer $Y^{(6)}$ movement alongside $Y^{(1)}$ and $Y^{(3)}$. Nonetheless, the representation gate is cleared.

## Stage 1, continued — 2 robustness checks.

*2026-08-03*

Last night, I posted a question to r/quant, wondering *"what is the strongest fair baseline: holding the eigenvectors fixed, EWMA, or a rotationally invariant estimator?"* in anticipation for stage 2. In a reply by *`u/Effective_Manager273`* two robustness checks were suggested alongside their response. I wrap up stage 1 by implementing and running the tests.

### Regime 4.8a — Does the signal require cross-sectional organisation?
*posed by `u/Effective_Manager273`:*

**What is tested:** Could the measured directional persistence arise from each asset’s individual distribution and autocorrelation, combined with rolling-window estimation, even when there is no organised cross-asset structure?

**Setup:** For each company independently, generate an IAAFT surrogate return history. then for each surrogate recompute the flag and observe mean cosine rotation. This follows the surrogate-testing framework of [Theiler et al.](https://www.sciencedirect.com/science/article/abs/pii/016727899290102S?via%3Dihub) and the iterative construction of [Schreiber–Schmitz](https://arxiv.org/abs/chao-dyn/9909041).

**Verdict:**
| Panel | Original cosine | After removal | Raw change | Coherence change |
|---|---:|---:|---:|---:|
| S&P | 0.1545 | 0.1551 | +0.4% | 13.05% → 15.64%, +19.8% |
| Nikkei | 0.0706 | 0.0786 | +11.3% | 6.95% → 7.14%, +2.8% |
| DAX | 0.0561 | 0.0657 | +17.1% | 10.98% → 11.42%, +4.0% |
| CAC | 0.0622 | 0.0637 | +2.4% | 10.36% → 8.92%, **−13.9%** |

Removing the market factor did not weaken the rotation signal at the complete flag level, but individual Flag layers changed substantially:
- S&P: market cosine gained 50%, top six lost 39%, while the $4{:}6$ block lost 24%.
- Nikkei: market cosine lost 21%, top three lost 15%, top six lost 21%.
- DAX: market lost 30%, top three lost 26%, while top six gained 56%.
- CAC: top six lost 66%, the $4{:}6$ block lost 40%, while top three gained 17%.
So the market removal redistributes where the persistence can be observed.

### Regime 4.8b — Remove the market factor and rebuild everything
*posed by `u/Effective_Manager273`:*

**What is tested:** Is the persistent Flag motion merely changing market beta or movement of the leading market factor?

**Setup:** Define a window where each companies beta is estimated, after removing hte average returns over the panel that day. Extract a new residual flag from this window, and run all the same tests. 

**Verdict:**
- Nikkei: **Pass**, persistence $p=0.01$, coherence $p=0.001$.
- DAX: **Pass**, persistence $p=0.01$, coherence $p=0.006$.
- CAC: **Pass**, persistence $p=0.02$, coherence $p=0.017$.
- S&P: **Pass**, persistence $p=0.01$, coherence $p=0.001$.

The signal **needs** real cross-sectional organisation and calendar structure.

## Stage 2 — family 1 benchmarks 

*2026-08-03*

To kickoff stage 2, I establish two familes of baslines to compare to: family 1 forecasts the Flag; family  2 forecasts the complete covariance matrix. 

### Benchmark Family 1 — forecast the future Flag
Every method receives the same current $\mathrm{Flag}(N;1,3,6)$ and predicts its position 42 trading days ahead.
| Benchmark | Forecast | Purpose |
|---|---|---|
| 1.1 Frozen Flag | predict no movement | mandatory zero-motion baseline |
| 1.2 Constant Velocity | repeat the previous full tangent step | mandatory naive continuation rule |
| 1.3 ERSE Direction | follow the published within-window ERSE rotation | checks whether temporal prediction reduces to ERSE |
| 1.4 HCAL Flag | use the current hierarchically filtered eigenspace | structural eigenvector-filter baseline |
| 1.5 BAHC Flag | use the bootstrapped hierarchical eigenspace | strongest recovered published eigenvector-filter baseline | 
| 1.6 Retained-Window Flag | use only the observations known to remain in the future rolling window | parameter-free rolling-composition forecast |
| 1.7 Stationary Roll-Forward | add a stationary fill for the 42 unseen observations | causal conditional rolling-window forecast |
| 1.8 RiskMetrics EWMA Flag | use the fixed $\lambda=0.94$ EWMA eigenspace | canonical short-memory financial forecast |
| 1.9 Validation-Geometric EWMA Flag | tune the EWMA half-life on validation Flag loss | objective-matched adaptive forecast |
| 1.10 Factor CM-IEWMA Flag | use the published multi-timescale factor covariance forecast | external financial forecast |

### Benchmark 1.1 — Frozen Flag

**What is tested:** How wrong am I if I pretend the Flag does not move?

**Setup:** At every forecast origin, predict
$$
\widehat{\mathcal F}_{t+42}=\mathcal F_t.
$$
Score the market, top-three, top-six and complete nested Flag against its actual position 42 trading days later.

**Verdict:** 
| Panel | Market | Block $2{:}3$ | Block $4{:}6$ | Top three | Top six | Complete Flag |
|---|---:|---:|---:|---:|---:|---:|
| S&P 500 | 0.00367 | 0.03828 | 0.09523 | 0.02594 | 0.04357 | 0.02439 |
| Nikkei | 0.00467 | 0.04470 | 0.20836 | 0.03074 | 0.10801 | 0.04781 |
| DAX | 0.00513 | 0.10787 | 0.22201 | 0.07255 | 0.09195 | 0.05654 |
| CAC 40 | 0.00487 | 0.03935 | 0.20328 | 0.02674 | 0.09966 | 0.04376 |

The market direction is already extremely stable, while the $4{:}6$ block leaves far more error available to remove.

Projector loss is bounded between 0 and 1 and consecutive rolling windows share most of their observations, so the numbers are small, and difficult to interpret. That motivated expressing the future results as skill relative to `Benchmark 1.1 — Frozen Flag`: 

$$
\mathrm{Skill} = 
100\left( 1-\frac{\mathrm{Loss}_{\mathrm{model}}}         
{\mathrm{Loss}_{\mathrm{Frozen}}} \right).
$$

### Benchmark 1.1-1.5 

In these results I take the equal-market average for benchmark $b$ and component $c$ across panels $m$:
$$
S_{b,c}^{\mathrm{combined}}
=
\frac14\sum_{m\in\{\mathrm{S\&P,Nikkei,DAX,CAC}\}}
\mathrm{Skill}_{b,m,c}.
$$

| Benchmark | What is tested |
|---|---|
| 1.1 — Frozen Flag | What happens if I pretend the Flag does not move? |
| 1.2 — Constant Velocity | What happens if I repeat the previous complete Flag rotation at full length? |
| 1.3 — ERSE Direction | Can ERSE’s within-window eigenvector correction serve as the next Flag forecast? | 
| 1.4 — HCAL Flag | Does replacing the current empirical Flag with a single hierarchically filtered correlation Flag improve prediction of the future empirical Flag? |
| 1.5 — BAHC Flag | Does bootstrap-averaging 1.4 make the eigenspace competitive.| 
| 1.6 — Retained-Window Flag | Does removing the observations known to expire improve the future Flag forecast? |
| 1.7 — Stationary Roll-Forward | Does a stationary forecast for the 42 unseen observations add to the retained-window forecast? |
| 1.8 — RiskMetrics EWMA Flag | Does the canonical fixed-decay financial forecast beat Frozen geometrically? |
| 1.9 — Validation-Geometric EWMA Flag | Can a validation-selected EWMA timescale beat Frozen geometrically? |
| 1.10 — Factor CM-IEWMA Flag | Does a published multi-timescale factor covariance forecast predict the future Flag? |

| Benchmark | Market skill | $2{:}3$ skill | $4{:}6$ skill | Top-three skill | Top-six skill | Complete-Flag skill | Worst panel, complete Flag |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen Flag | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| Constant Velocity | −59.9% | −53.4% | −50.6% | −53.7% | −54.6% | −53.7% | −57.4% |
| ERSE Direction | −3252% | −1.5% | −2.1% | −147.2% | −33.2% | −189.9% | −337.1% |
| HCAL Flag | −492.1% | −892.1% | −326.9% | −895.3% | −414.2% | −535.2% | −1001.6% |
| BAHC Flag | −391.9% | −761.4% | −304.3% | −762.0% | −355.8% | −458.0% | −881.0% |
| **Retained-Window Flag** | **+40.9%** | **+38.5%** | **+39.5%** | **+38.5%** | **+40.7%** | **+39.6%** | **+37.4%** |
| **Stationary Roll-Forward** | **+41.4%** | **+37.8%** | **+39.2%** | **+37.9%** | **+40.5%** | **+39.2%** | **+36.8%** |
| RiskMetrics EWMA Flag | −1869% | −891.6% | −371.3% | −903.9% | −476.8% | −629.7% | −1121.5% |
| **Validation-Geometric EWMA Flag** | **+22.2%** | **+25.1%** | **+19.1%** | **+24.9%** | **+17.6%** | **+20.1%** | **+16.2%** |
| Factor CM-IEWMA Flag | −90.2% | −90.7% | −68.6% | −91.2% | −62.4% | −71.6% | −187.8% |

Of the ten Family 1 baselines, only five are directly aligned with the 42 days ahead rolling Flag target. The remaining methods are established covariance estimators repurposed as external comparators.

| Benchmark | Market skill | $2{:}3$ skill | $4{:}6$ skill | Top-three skill | Top-six skill | Complete-Flag skill | Worst panel, complete Flag |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen Flag | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| Constant Velocity | −59.9% | −53.4% | −50.6% | −53.7% | −54.6% | −53.7% | −57.4% |
| **Retained-Window Flag** | +40.9% | **+38.5%** | **+39.5%** | **+38.5%** | **+40.7%** | **+39.6%** | **+37.4%** |
| **Stationary Roll-Forward** | **+41.4%** | +37.8% | +39.2% | +37.9% | +40.5% | +39.2% | +36.8% |
| Validation-Geometric EWMA Flag | +22.2% | +25.1% | +19.1% | +24.9%** | +17.6% | +20.1% | +16.2% |

## Stage 2 — alarms 

*2026-08-04*

### Alarm 1 — rolling-window deletion contamination

**What is tested:** Benchmarks 1.6 and 1.7 score +39.6% and +39.2% against
Frozen while forecasting nothing about markets. Why?

**Mechanism.** The Stage 2 target is the *rolling* Flag at $t+42$, re-estimated
from $[t+42-T+1,\;t+42]$. That window shares $T-42$ of its $T$ observations with
the current window — 83% at $T=250$, 88% at $T=357$. With fixed normalisation,

$$\hat C_{t+42}-\hat C_t=\tfrac1T\big(S_{\text{new}}-S_{\text{drop}}\big),$$

and $S_{\text{drop}}$, the scatter of the 42 departing observations, is **fully
known at the forecast origin**. Benchmark 1.6 simply deletes them. It is reading
the estimator's own bookkeeping schedule off the calendar.

**Setup.** Re-base the statistic to the *retained* Flag $\mathcal B_t$, putting
the deletion in the base point rather than the target. Add a volatility-matched
null: permute intact 42-day blocks within realised-volatility strata, rebuild the
entire pipeline. 99 replicates per panel.

**Verdict:** deletion accounts for **39.1% – 49.6%** of outgoing tangent energy
across all panels and components.

| Panel | complete Flag | market | $2{:}3$ | $4{:}6$ |
|---|---:|---:|---:|---:|
| S&P 500 | 0.451 | 0.453 | **0.496** | 0.453 |
| Nikkei | 0.392 | 0.447 | 0.442 | 0.394 |
| DAX | **0.391** | 0.453 | 0.447 | 0.399 |
| CAC 40 | 0.402 | 0.449 | 0.427 | 0.413 |

**The old calendar null was never a control for this.** Log realised-variance
lag-1 autocorrelation by non-overlapping aggregation scale, S&P:

| scale | observed | calendar-21 null | volatility-42 null |
|---|---:|---:|---:|
| 1d | +0.674 | +0.643 | +0.665 |
| 14d | +0.827 | +0.384 | +0.745 |
| 42d | +0.771 | **−0.016** | **+0.644** |
| 63d | +0.739 | +0.005 | +0.707 |

A 21-day block permutation annihilates volatility clustering beyond ~42 days —
precisely the scale that governs deletion-direction alignment. The
volatility-matched permutation retains ~83% of it. Under the corrected null,
only S&P survives Holm across four panels at the published window
(p = 0.01 / 0.05 / 0.35 / 0.52 for S&P / Nikkei / DAX / CAC).

**Novelty, stated honestly.** Allez & Bouchaud already knew. §6, on Fig. 7:

> "The fact that the empirical (red) curve starts from 0 for $\tau=0$ and
> increases to reach the stationary noise level at time $\tau=T$ **is simply due
> to the overlapping between the sliding periods.**"

and every quantitative statement in the paper is then confined to the
non-overlapping regime — §4: *"two non overlapping time periods, i.e. such that
$|t-s|>T$"*; §6: *"For times $s<t$ **with $|t-s|>T$**, we define the overlap
matrix $G^{s,t}$."*

My Stage 2 runs at $h=42$ with $T=250$–$357$, so $|t-s|\ll T$ throughout. **The
project spent its entire benchmark family inside the regime the source paper
explicitly fences off.** Alarm 1 is not a correction to the literature. It is the
discovery that I walked into a hole that was already signposted.

What remains genuinely new is narrower and should be claimed at that size: a
*quantification* inside the excluded regime (39–49%, four markets), and the
demonstration that a calendar block permutation fails as a control there while a
volatility-matched one succeeds. That is a methods note, not a headline.

### Alarm 2 — the window rule buys nothing

**What is tested:** every real-data script sets `T = max(N, 250)` (12 assignment
sites). On S&P that gives $T=N=357$, so $q\equiv N/T=1.000$ exactly — the
singular boundary. Median minimum eigenvalue is numerically zero and the
condition number is unbounded, against $\sim10^3$ / $87$ / $37$ on Nikkei / DAX /
CAC. The rule is inherited from the replication arm, where it was correct, and it
contradicts Regime 4.3's own measured optima (999 / 793 / 1032 / 529).

**Setup:** two-arm sweep on S&P, 99 volatility-null replicates per cell. Arm A
holds $h=42$; arm B holds the deletion fraction $h/T\approx0.112$.

**Verdict, arm B** (isolating $q$): every column monotone, excess grows 5.2×.

| $T$ | $q$ | null mean | observed | excess | $z$ |
|---:|---:|---:|---:|---:|---:|
| 357 | 1.000 | 0.0564 | 0.0977 | +0.0414 | 2.95 |
| 500 | 0.714 | 0.0460 | 0.1347 | +0.0887 | 4.49 |
| 750 | 0.476 | 0.0376 | 0.1923 | +0.1547 | 5.81 |
| 1008 | 0.354 | 0.0333 | 0.2480 | +0.2147 | 6.00 |

**But the decisive check kills the reading.** `addition_speed` is the norm of the
tangent from the retained Flag to the realised future Flag — the size of the
thing a model is asked to predict. Multiply it by the excess cosine:

| arm | $T$ | $h$ | addition speed | excess cosine | **product** |
|---|---:|---:|---:|---:|---:|
| A | 357 | 42 | 0.2249 | 0.0414 | **0.0093** |
| A | 500 | 42 | 0.1649 | 0.0593 | **0.0098** |
| A | 750 | 42 | 0.1095 | 0.0607 | **0.0066** |
| A | 1008 | 42 | 0.0926 | 0.1038 | **0.0096** |

The speed collapses 2.4× exactly as the cosine rises 2.5×. Along arm A the window
rule trades estimator noise against target amplitude and the explainable motion
is unchanged. The apparent 2.5× gain in significance was the denominator
shrinking. *(No uncertainty band on the product yet — four unbanded points. The
absence of a monotone trend is safe; the word "invariant" is not yet earned.)*

Arm B does raise the product, 0.0093 → 0.0391, but only by stretching the horizon
to 112 days. Different question, not a better answer.

**What survives:** the $q=1.000$ conditioning failure is real and
amplitude-independent — S&P's Gaussian NLL and GMV long/short numbers are
uninterpretable as they stand. The "5× stronger signal at a longer window"
reading does not survive. At $T=750$ all four panels show positive excess and two
survive four-panel Holm (0.04 / 0.04 / 0.056 / 0.056) instead of one.

**Why no $T$ escapes.** The target is the rolling Flag, which shares $(T-h)/T$ of
its observations with the base. As $T$ grows that overlap $\to1$ and the target
converges onto the thing being forecast *from*. Well-conditioned estimate, or a
target worth predicting. Not both.

## Stage 2, respecified — variance captured

### The move

Predict the same object; score it against realised returns. Forecasters still
emit an $N\times6$ orthonormal frame. Only the evaluation changes:

$$\text{capture}^{(d)}=\frac{\lVert\hat Y^{(d)\top}R_{\text{out}}\rVert_F^2}
{\lVert R_{\text{out}}\rVert_F^2}$$

*How much of next quarter's realised risk does my six-factor model span.*

Three independent parameters, never again conflated: $T_{\text{in}}=750$ chosen
for conditioning; $T_{\text{out}}=42$ chosen for economic relevance and **is**
the horizon; `step` $=14$. There is no $h/T$ in this design. The estimation and
target windows are disjoint by construction, so deletion contamination is
impossible rather than corrected.

Realised returns are standardised by **estimation-window** volatilities only.
`standardise(panel)` defaults to `window=21`, which reflect-pads and reads ±10
days ahead — a strictly-future perturbation moves a window's correlation by
5.5e−2 at `window=21` and 2.2e−16 at `window=1`. Always pass `window=1`.

### Benchmark — the single ladder

$T_{\text{in}}=750$, $T_{\text{out}}=42$, step 14. Origins: 429 / 426 / 414 / 418.
Paired mean (entrant − Frozen) at $d=6$, raw capture, t-statistic in brackets.

| Panel | $N$ | random $6/N$ | Frozen | in-sample ceiling | Retained Window | Constant Velocity | EWMA hl=252 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAC 40 | 23 | 0.261 | 0.638 | 0.797 | −0.0030 [−8.9] | +0.0004 [+1.0] | +0.0029 [+5.0] |
| DAX | 29 | 0.207 | 0.591 | 0.764 | −0.0021 [−9.1] | +0.0010 [+3.2] | +0.0015 [+3.3] |
| Nikkei | 131 | 0.046 | 0.488 | 0.650 | −0.0023 [−13.5] | +0.0003 [+1.5] | +0.0018 [+7.9] |
| S&P 500 | 357 | 0.017 | 0.425 | 0.582 | −0.0031 [−19.3] | +0.0011 [+7.7] | +0.0032 [+11.6]|

Sign consistency across 4 panels × 3 levels × 2 modes (raw and
market-neutralised), counting cells with $|t|>2$:

| entrant | significantly positive | significantly negative |
|---|---:|---:|
| Retained Window | 0 / 24 | **22 / 24** |
| Constant Velocity | 19 / 24 | 0 / 24 |
| EWMA hl=126 | 17 / 24 | 0 / 24 |
| **EWMA hl=252** | **24 / 24** | 0 / 24 |

**Retained Window went from +39.56% to losing significantly on 22 of 24 cells.**
The artifact is not corrected, it is gone, and a real estimator wins in its
place. That is the respecification working exactly as designed.

*Caveat: origins are pooled, no train/validation/test split, and the EWMA
half-lives were fixed by hand rather than selected on validation. The sign
pattern is the result; the t-statistics treat origins as independent and are
therefore too large. Must be redone split-clean with circular block intervals at
block length $\lceil(T_{\text{in}}+T_{\text{out}})/\text{step}\rceil=57$ origins,
~7 independent blocks over full history.*

### The ceiling is not free

The in-sample top-6 of $R_{\text{out}}$ *is* the exact attainable maximum. But
with $T_{\text{out}}=42$ observations and $N$ up to 357, it overfits the
realisation badly, so a large part of the skill denominator is noise.

**Test:** simulate the future from the estimation-window covariance itself — a
world where the true subspace does not move, so honest headroom is exactly zero.
Compute ceiling − frozen anyway:

| Panel | $N$ | $N/T_{\text{out}}$ | frozen | ceiling | headroom that is pure overfitting |
|---|---:|---:|---:|---:|---:|
| CAC 40 | 23 | 0.55 | 0.596 | 0.708 | **+0.112** |
| DAX | 29 | 0.69 | 0.530 | 0.649 | **+0.119** |
| Nikkei | 131 | 3.12 | 0.413 | 0.528 | **+0.115** |
| S&P 500 | 357 | 8.50 | 0.380 | 0.485 | **+0.106** |

Against a measured CAC headroom of 0.159, **~70% is fake**. Skill as originally
specified is ~3.4× too optimistic in its denominator. Same failure mode as
Alarm 1, reintroduced into the metric built to escape it.

Corrected:

| Panel | naive headroom | fake | **real headroom** | best entrant | naive skill | **honest skill** |
|---|---:|---:|---:|---:|---:|---:|
| CAC 40 | 0.159 | 0.112 | **0.047** | +0.0029 | 1.8% | **6.2%** |
| DAX | 0.173 | 0.119 | **0.055** | +0.0015 | 0.8% | 2.7% |
| Nikkei | 0.162 | 0.115 | **0.047** | +0.0018 | 1.1% | 3.8% |
| S&P 500 | 0.158 | 0.106 | **0.052** | +0.0032 | 2.0% | **6.1%** |

Two things worth recording. The bias is stable at 0.106–0.119 while $N$ varies
16×, so a single stationary-null correction is well behaved. And a split-half
cross-fit ceiling does **not** work — fitting top-6 on days 1–21 and scoring on
22–42 gives, under stationarity, a "ceiling" *below* Frozen (0.508 vs 0.588 on
CAC). It is a lower bound, not a ceiling. The fix is a per-origin stationary
ceiling null, symmetric with the Haar floor.

The real headroom is 0.047–0.055 on all four panels while $N$ runs 23 → 357.
Second near-invariance, alongside the flat product of Alarm 2. Both need error
bars before either is claimed.

### The rotationally-invariant estimator blind spot

Covariance-family entrants on the same ladder, $d=6$, minus Frozen:

| Panel | QIS | Ledoit–Wolf | OAS | EWMA hl=252 |
|---|---:|---:|---:|---:|
| CAC 40 | +0.0008 | +0.0008 | +0.0008 | +0.0032 |
| DAX | +0.0003 | +0.0003 | +0.0003 | +0.0020 |
| Nikkei | +0.0001 | +0.0001 | +0.0001 | +0.0021 |
| S&P 500 | −0.0002 | −0.0002 | −0.0002 | +0.0033 |

The three shrinkage estimators agree to four decimal places on every panel. Not a
bug. **Linear shrinkage toward a scaled identity leaves eigenvectors exactly
unchanged, and every rotationally-invariant estimator — QIS included — is defined
as optimal eigenvalue shrinkage with the sample eigenvectors held fixed.** On a
subspace metric the entire RIE class is pinned at zero by construction. The
residual is preprocessing noise.

The covariance-cleaning literature has not tried and failed to forecast
correlation geometry. The question is structurally outside its frame. EWMA wins
because it is the only entrant in the ladder that actually rotates the frame.

### What the number means

$1-\text{capture}$ is the share of next quarter's cross-sectional risk the
six factors do not see.

| Panel | $N$ | Frozen spans | **unspanned** | residual vol cut, perfect foresight | residual vol cut, best real model |
|---|---:|---:|---:|---:|---:|
| CAC 40 | 23 | 63.8% | 36.2% | 6.75% | **0.40%** |
| DAX | 29 | 59.1% | 40.9% | 6.90% | **0.18%** |
| Nikkei | 131 | 48.8% | 51.2% | 4.69% | **0.17%** |
| S&P 500 | 357 | 42.5% | **57.5%** | 4.63% | **0.28%** |

0.2–0.4% off residual volatility is below transaction costs and below the
estimation error in the volatility forecast sitting next to it. The oracle line
says the same from the other side: perfect foresight of the future Flag buys
+1.7% to +6.2% GMV long-only, and a real model reaches ~5% of that.

So the forecasting result is not the finding. The **level** is:

> A six-factor risk model does not span half of the S&P 500's next-quarter
> cross-sectional risk. Choosing its six directions with perfect hindsight, it
> still misses 52%. The forecastable part of that blindness is worth 0.3% of
> residual volatility.

The forecast result's job is to prove the metric is live. It does that — 24/24,
four markets, while simultaneously killing the artifact that dominated the old
leaderboard. Then it gets one paragraph.
---

## Model 4.1 — solving the first-order problem exactly, and losing anyway

### The objective is linear in the projector, so only one block of any correction can matter

The capture score is an inner product against a projector,

$$\text{capture}=\frac{\lVert\hat Y^\top R_{\rm out}\rVert_F^2}{\lVert R_{\rm out}\rVert_F^2}=\big\langle P,\;\tilde S_{\rm out}\big\rangle,\qquad P=\hat Y\hat Y^\top,\quad \tilde S_{\rm out}=\frac{R_{\rm out}R_{\rm out}^\top}{\operatorname{tr}R_{\rm out}R_{\rm out}^\top}.$$

Linear in $P$, so by linearity of expectation the Bayes-optimal frame is the top-6
eigenspace of $M_t=\mathbb E[\tilde S_{\rm out}\mid\mathcal F_t]$. The problem is not
"forecast eigenvectors"; it is *forecast one matrix, then diagonalise it*. The object
to forecast is the **trace-normalised** second moment, not the covariance — volatility
level cancels exactly, which is why the oracle line found marginal vol to dominate the
covariance problem while contributing nothing here.

Perturb a base estimator, $\hat M_t=\hat C_t+\varepsilon G$, with $\hat C_t$ having
eigenpairs $(\lambda_i,u_i)$, $U_6=[u_1..u_6]$, $U_\perp=[u_7..u_N]$. The Riesz
projector for the group $\{\lambda_1..\lambda_6\}$ has first variation

$$\delta P=\varepsilon\!\!\sum_{i\le6}\sum_{j>6}\frac{u_j^\top Gu_i}{\lambda_i-\lambda_j}\big(u_ju_i^\top+u_iu_j^\top\big)+O(\varepsilon^2),$$

the within-group terms cancelling identically. So $\delta P$ depends on $G$ **only**
through $G^{\rm oi}:=U_\perp^\top GU_6\in\mathbb R^{(N-6)\times6}$. The within-top-6
block, the within-complement block and the entire diagonal are invisible to the score,
and

$$\delta(\text{capture})=2\varepsilon\Big\langle \frac{G^{\rm oi}}{\Lambda},\;\Sigma^{\rm oi}_{\rm out}\Big\rangle,\qquad \Lambda_{ji}=\lambda_i-\lambda_j,\quad \Sigma^{\rm oi}_{\rm out}=U_\perp^\top\tilde S_{\rm out}U_6.$$

`src/coupling.py` implements this; `tests/test_coupling.py` checks it rather than
asserting it. A diagonal correction in the sample eigenbasis moves the projector by
$<10^{-9}$; edits confined to the top-6 or complement blocks move individual columns
but leave the projector fixed to $10^{-9}$ and the capture identical to $10^{-12}$; and
`score_gradient` matches a finite difference of realised capture with error falling
in proportion to $\varepsilon$.

### The RIE result, refined — it was the pipeline, not QIS

The structural claim is right and now measured: the **shrinkage step** of Ledoit–Wolf,
OAS and QIS has visible block $\lVert U_\perp^\top GU_6\rVert\sim10^{-16}$ on every
sampled origin of every panel. Linear shrinkage toward $\mu I$ and nonlinear eigenvalue
cleaning alike are pinned to Frozen by construction, not approximately.

But the reported ladder does not show three exact zeros. LW and OAS give $0.0000$;
QIS gives $-0.0002$ to $-0.0011$. The cause is not QIS. The ladder renormalises every
estimate to a correlation matrix, and $D^{-1/2}SD^{-1/2}$ is a **congruence, not a
similarity**: it preserves eigenvectors only when $D$ is a scalar. Shrinking a
constant-diagonal matrix toward a multiple of the identity leaves the diagonal
constant, so LW and OAS pass through it untouched. QIS cleans each eigenvalue
differently, its diagonal is not constant, and the renormalisation rotates the frame —
measured visible block $2.5\times10^{-2}$, $\lVert\delta P\rVert=0.04$ on S&P.

So the earlier reading "the residual is preprocessing noise" was correct, and is now
identified rather than assumed. Both stages are recorded per panel as
`rie_shrinkage_invisible` and `rie_invisible_after_renormalisation`.

### The split-clean ladder, and what it costs the old t-statistics

`scripts/stage2_capture_ladder.py` regenerates the ladder reproducibly: half-life
selected on validation, comparison on test only, circular-block intervals at
$\lceil(T_{\rm in}+T_{\rm out})/\text{step}\rceil=57$ origins. Calibration against the
uncommitted original run reproduces it — CAC Frozen 0.6411 / ceiling 0.7971 against a
recorded 0.638 / 0.797 — once the realised block is scaled by estimation-window
per-name volatilities **without** day-flattening. That convention matters: flattening
days moves CAC Frozen from 0.641 to 0.564, so it is a flag in the code, not a silent
default.

| Panel | $N$ | test origins | random $6/N$ | Frozen | ceiling | ceiling bias | **real headroom** |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAC 40 | 23 | 121 | 0.261 | 0.618 | 0.784 | 0.078 | **0.088** |
| DAX | 29 | 120 | 0.207 | 0.579 | 0.749 | 0.083 | **0.087** |
| Nikkei | 131 | 114 | 0.046 | 0.473 | 0.647 | 0.089 | **0.086** |
| S&P 500 | 357 | 118 | 0.017 | 0.442 | 0.614 | 0.078 | **0.093** |

The near-invariance of real headroom survives the split-clean redo and tightens:
0.086–0.093 while $N$ runs 23 → 357. (It sits above the 0.047–0.055 recorded earlier
because that figure pooled all origins and this one is test-only; the *invariance*, not
the level, is the reproducible part.)

The predeclared worry about pooled t-statistics was justified, and partially. Best
EWMA against Frozen, split-clean:

| Panel | best EWMA | paired improvement | 95% circular-block CI | excludes zero |
|---|---|---:|---|:--:|
| CAC 40 | hl=126 | +0.0043 | [−0.0041, +0.0126] | **no** |
| DAX | hl=252 | +0.0082 | [+0.0058, +0.0109] | yes |
| Nikkei | hl=252 | +0.0081 | [+0.0065, +0.0098] | yes |
| S&P 500 | hl=126 | +0.0145 | [+0.0115, +0.0175] | yes |

So the 24/24 sign pattern survives and the *effect* is larger than the pooled estimate,
not smaller — but CAC, the panel with the smallest $N$ and the 24/24 headline, loses
significance once its origins are correctly treated as ~2 independent blocks. Three of
four panels stand.

### Model 4.1 — five parameters, every one of them visible

$$\hat M_t=\underbrace{\hat C_t(\theta)}_{\text{EWMA correlation}}+\varepsilon\sum_{m=1}^{3}\beta_m\Big(U_\perp A^{(m)}_tU_6^\top+U_6A^{(m)\top}_tU_\perp^\top\Big)$$

with $A^{(1)}$ the fast/slow EWMA difference projected into the visible block,
$A^{(2)}$ the realised Grassmann log at the base frame, $A^{(3)}=z_tA^{(1)}$ with $z_t$
a causally standardised log realised variance. Three $\beta$, one $\varepsilon$, one
$\theta$. The family contains Frozen ($\varepsilon=0,\theta\to\infty$), every EWMA
($\varepsilon=0$) and momentum ($\beta=e_2$) exactly, so the comparison is nested.
$\beta$ is fitted closed-form on train; $\varepsilon$, $\theta$ and the fit are chosen
on validation by **realised capture of the exactly corrected frame** — the perturbation
theory selects what to fit, it never does the scoring. The eigengap floor never bound
on any panel (0.000), so the $1/\Lambda$ weight is reporting data rather than a clip.

**Result. The predeclared stopping rule was cleared on 3 of 4 panels and the gain did
not survive to test on any of them.**

| Panel | selected $\theta$ | selected $\varepsilon$ | validation gate | gate CI low | passed | **test vs EWMA** | 95% CI |
|---|---:|---:|---:|---:|:--:|---:|---|
| CAC 40 | 252 | 0.020 | +0.0004 | +0.0004 | yes | **−0.0002** | [−0.0018, +0.0013] |
| DAX | ∞ | 0.200 | +0.0010 | +0.0010 | yes | **−0.0029** | [−0.0046, −0.0014] |
| Nikkei | 252 | **0.000** | 0.0000 | 0.0000 | **no** | 0.0000 | [0.0000, 0.0000] |
| S&P 500 | 252 | 0.100 | +0.0001 | +0.0001 | yes | **−0.0001** | [−0.0002, +0.0000] |

Nikkei is the cleanest line in the table. Offered a free amplitude, validation chose
$\varepsilon=0$ — the correction was switched off and the model collapsed onto plain
EWMA. DAX is the sharpest: the model is significantly *worse* than the EWMA it
contains, its interval excluding zero on the wrong side. The two remaining panels are
indistinguishable from their own baseline.

The validation gains were real but of order $10^{-4}$, and a gate that only asks
"is the improvement consistent across blocks" will pass a consistent $10^{-4}$. That is
a lesson about the gate, and it is recorded rather than patched: the rule was
predeclared, it was applied verbatim, and no feature hunting followed.

### Optimising the risk is not optimising the geometry — and it does not matter

A model can be trained to predict *where the subspace goes* or to maximise *how much
realised variance it spans*. These coincide only if realised variance is isotropic
across the complement, which it is not. Four fits were run, differing only in the loss
and never in the parameter count: `gradient` (risk loss, first order), `ridge`
(gap-weighted geometry), `geometric` (pure geometry, $1/\Lambda$ dropped) and `direct`
(exact capture search over the sphere).

| Panel | cos(geometric, risk) | cos(ridge, risk) | cos(direct, risk) | spread in validation capture |
|---|---:|---:|---:|---:|
| CAC 40 | **−0.16** | +0.45 | +0.92 | 0.00458 → 0.00493 |
| DAX | +0.77 | +0.73 | +1.00 | 0.00233 → 0.00256 |
| Nikkei | +0.87 | +0.21 | +0.99 | all 0.00338 |
| S&P 500 | +0.97 | +0.54 | +1.00 | 0.00201 → 0.00209 |

Two things. The distinction is real and $N$-dependent: on the smallest panel the pure
geometry loss picks a direction essentially orthogonal to — and slightly opposed to —
the risk-optimal one, while on S&P the two agree at 0.97. And `direct` agrees with
`gradient` at 0.92–1.00 everywhere, so the first-order surrogate is not misleading;
the linearisation is doing its job.

But the fourth column is the finding. **All four losses score within $4\times10^{-4}$
capture of each other.** The objective is flat in $\beta$: you can rotate the fitted
direction by 90 degrees and lose almost nothing. That is a stronger statement than the
headroom argument, because it says the failure is not "we picked the wrong loss" or
"we picked the wrong features" — there is no direction in this feature space that the
metric rewards appreciably.

### What this closes

The first-order condition was solved exactly. Every parameter was spent on the only
block the metric can see, the RIE class was shown to be structurally pinned rather than
merely unlucky, the label was constructed rather than approximated, four different
objectives were compared, and a predeclared stopping rule was honoured. The result is
that a metric-aware model **does not beat the EWMA it contains** on any of four
markets, and on one of them is significantly worse.

That converts the earlier claim. Not "we tried EWMA and nothing else worked", but:

> The realised-variance capture of a six-factor risk model is not meaningfully
> forecastable from the correlation geometry. The problem's first-order structure
> admits exactly one $(N-6)\times6$ block of influence; a five-parameter model
> targeting that block directly, fitted four ways, cannot improve on a
> one-parameter exponential kernel. The ceiling is not a limitation of the model
> class — the objective is flat.

The level result from the respecified Stage 2 is unchanged and remains the finding:
a six-factor model does not span half of the S&P 500's next-quarter cross-sectional
risk, and with perfect hindsight of its own six directions it still misses 39%.
The forecastable part of that blindness is worth 0.5–1.2% of residual volatility,
essentially all of which a plain EWMA already collects.
