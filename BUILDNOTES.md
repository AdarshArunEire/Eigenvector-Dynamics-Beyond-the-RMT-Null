# Build notes — eigenvector dynamics beyond the RMT null

## References

Allez & Bouchaud wrote this twice and arXiv does not link the two, because they are
separate submissions rather than versions of one. I worked from the short one for most of
both dates without knowing the long one existed.

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
