# Reading list — eigenvector overlap and subspace rotation across fields

*Assembled 2026-08-02 from a search sweep. Triaged 2026-08-02 against every abstract.*

**The rule from here on:** anything still on this list is committed to a deep read. Anything that
could not name a question in BUILDNOTES it would answer has been moved to [Cut](#cut) with the
reason, so the next search sweep does not re-find it and re-triage it from scratch.

Status markers: **[read]** gone through; **[abstract read]** abstract confirmed at source, full
text not read; **[deep read pending]** committed, not yet done.

Thirteen entries survive, in three tiers. The tiers are a reading order, not a quality ranking —
tier A changes what I build next, tier C changes how I defend what I have built.

---

> **2026-08-02, after deep reads.** Two papers were missing from this list entirely and both belong
> in tier A. See `PRIOR_ART.md` for the full write-up. In short: the null I am using is the
> *perturbative* 2012 form, and an **exact** version has existed since 2016; and CFM published a
> competing non-stationarity test in 2022 that I had never seen. Both are added below and marked
> **[read]**.

## Tier A — read before building anything else

**Bun, Bouchaud & Potters, *Overlaps between eigenvectors of correlated random matrices*,
[arXiv:1603.04364](https://arxiv.org/abs/1603.04364), Phys. Rev. E 98, 052145 (2018)** — **[read]**
**The paper 2509.25076 actually builds on; Allez–Bouchaud 2012 is cited there only as where the
subject was "first discussed".** General, exact, closed-form mean squared overlaps
$\Phi(\lambda,\lambda') = N\mathbb{E}(u_\lambda\cdot u'_{\lambda'})^2$ between eigenvectors of two
sample covariances on non-overlapping windows — Eq (6), self-overlap Eq (9), additive case Eqs (8)
and (10) — and **none of them involve $C$**. They propose the resulting statistical test outright.
Two things keep my ground: they explicitly **exclude the top eigenvectors** ("governed by non
trivial dynamics"), which is the block I work on; and they read the bulk departure as *spectroscopy
of the true spectrum* (inverse-Wishart fit, $\kappa\approx0.7$ on 450 US stocks 2005–2012), not as
non-stationarity. **Same measurement, opposite interpretation — that is the objection to prepare
for.** Next action is not a re-read but a rebuild: derive $D_{th}$ from their exact $\Phi$ instead of
Eq (6.1)'s perturbative sum and check against regime 1.1's 0.25%.

**Bouchaud, Mastromatteo, Potters & Tikhonov, *Excess Out-of-Sample Risk and Fleeting Modes*,
[arXiv:2205.01012](https://arxiv.org/abs/2205.01012), Wilmott 2022** — **[read]** **Was not on this
list and should have been.** CFM's *alternative* to the eigenvector-overlap test, described in their
own intro as "simpler and more transparent": the spectrum of
$\mathbb{D} = \mathbb{E}_{in}^{-1/2}\mathbb{E}_{out}\mathbb{E}_{in}^{-1/2}$, whose eigenvalues are
**exactly** $C$-independent by a characteristic-polynomial argument, against a Wishart ×
inverse-Wishart null (Eq 6, edges Eq 7). They standardise by intraday Garman–Klass volatility for
the same reason I do. Finite-$N$ shifts the edge by $\approx$ 17% at $N=98$ — precedent for taking
small panels seriously. **Strategic significance: CFM moved off the overlap route for detection.
Mine is the branch that supports a forecast; theirs is not extrapolable.**

**Riabov, Tikhonov & Bouchaud, *Eigenvector overlaps of sample covariance matrices with
intersecting time periods*, [arXiv:2509.25076](https://arxiv.org/abs/2509.25076)** —
**[read]** Intersection enters through one scalar $t = T_B\sqrt{q\tilde q}/N \in [0,1]$; central
result Eq (28) reduces to Bun–Bouchaud–Potters at $t=0$. **`min_lag = T` can go.** They detect
non-stationarity on real US equities (Fig. 3) — qualitatively, by eye, with no excess statistic,
threshold or power curve. **Their closing paragraph states that heavy-tail effects mimicking
non-stationarity "remains an open problem" — regimes 1.5 and 3.1 answer exactly that.**

*Original annotation, kept for the record:* *"We compute exactly the overlap between the eigenvectors
of two large empirical covariance matrices computed over intersecting time intervals, generalizing
the results obtained previously for non-intersecting intervals."* Girko linearisation plus extended
local laws, checked numerically, applied to financial data. Six pages plus supplementary.

This is the direct modern successor to Eq (6.1), by Bouchaud himself, and it covers the case
`regime4.py` currently throws away via `min_lag = T`.

*The deep read must answer:* (1) does their formula reduce to Eq (6.1) in the zero-intersection
limit, and if so does it reproduce my measured $D_{th}$ to the 0.25% that regime 1.1 holds to?
(2) can `min_lag = T` be dropped, which would recover every window pair at $\tau < T$ — the region
regime 4.1 currently uses only as a windowing sanity check? (3) do they hit the same small-$T$
breakdown that puts DAX and CAC *below* their own null in regime 4.1, or does an exact
(non-perturbative) calculation make that go away?

**Allez & Bouchaud, *Eigenvector dynamics under free addition*,
[arXiv:1301.4939](https://arxiv.org/abs/1301.4939)** — **[abstract read] [deep read pending]**
**Promoted from "found, unread" — this was badly under-rated.** The abstract: evolution of a given
eigenvector under addition of a GOE matrix, overlap with the eigenvectors of the initial matrix, a
"Cauchy-flight" regime identified, and the local density of that vector in the initial matrix's
eigenvalue space. Critically: *"Our results are obtained in a non perturbative setting"*, and it
closes by giving *"a robust derivation of a result obtained in [Allez & Bouchaud, Phys. Rev. E 86,
046202 (2012)]"* — which is the published 1203.6228.

So this is a **non-perturbative rederivation of the formula the whole project rests on.** Regime 2.2
and regime 4.1 are both stories about where the perturbative expansion stops working — the bulk edge
climbing toward $\lambda_P$, and a 29-day window putting DAX below its own null. A non-perturbative
version is the correct tool for exactly those two failures, and I had it filed as a follow-up
curiosity.

*The deep read must answer:* what does the non-perturbative form predict at $T = 29$ and at the bulk
geometries in the regime 2.2 grid where the substitution bias hit +18%? If it tracks my measurements
where perturbation theory does not, the DAX/CAC result stops being "Eq (4.8) is wrong in this corner"
and becomes a quantitative correction.

**Allez & Bouchaud, *Eigenvector dynamics: general theory and some applications*,
[arXiv:1203.6228](https://arxiv.org/abs/1203.6228)** — **[read]** The spine. Four indices, $T = N$,
Eq (6.1) the two-window null, §5 the EWMA Langevin treatment, §6.2 the market-mode tilt toward the
uniform vector.

Not a first read but a **targeted re-read**, and BUILDNOTES already names the three places:
(1) confirm the §4 equation numbering against the PDF — the two §4 entries in the reference table are
cited by section because the numbering could not be recovered from extracted text, and they are
quoted in stage 1; (2) §5 Eq (5.1), before building the EWMA instrument; (3) §6.2, because the
market-mode tilt is the dynamic version of the prior ERSE imposes statically.

Note the 4-page letter [arXiv:1108.4258](https://arxiv.org/abs/1108.4258) is a separate submission,
not a version — Nikkei only, different equation numbers, and a $T^*$ claim the full paper
contradicts. The map between the two lives in BUILDNOTES. Cite the full paper.

## Tier B — direct comparators and next claims

**Liu & Liu, *Covariance Matrix Estimation for Positively Correlated Assets*,
[arXiv:2507.01545](https://arxiv.org/abs/2507.01545)** — **[§1–§5 read]**
ERSE: pairwise Givens rotations of eigenvectors linked to **weak factors**, preserving
orthogonality, under a positive-comovement prior. **10.52% out-of-sample variance reduction against
linear shrinkage, 12.46% against nonlinear.**

Two things the abstract corrects in my earlier note. First, the empirical work is on
**factor-sorted portfolios from the Ken French library at monthly frequency** — not an equity
cross-section at daily frequency. That is a different $N$, a different $T$, and a different kurtosis
regime from all four of my panels, so the 12.46% does not transfer to my setting unexamined. Second,
their Proposition 5 says ERSE's pairwise rotation recomputes two Rayleigh
eigenvalues that move toward one another while preserving their sum. A Givens
rotation of modes $i,j$ by $\theta$ is the same geometric operator used for the
Regime 3.1 injection, but the full ERSE algorithm is not identical to one
injection: it chooses pairs and angles iteratively, recomputes eigenvalues and
uses the rotated basis.

Regime 4.6 implements that algorithm directly. On the four full-history panels,
ERSE explains only 0.05–0.92% of outgoing tangent energy at $\delta=0.25$ and
residual directional persistence remains significant against the matched null.
The strict all-positive assumption holds in only 0–36.6% of rolling windows,
although 97.4–98.7% of individual off-diagonal correlations are positive, so
the empirical comparison is a deliberate extrapolation rather than an
unqualified application of the paper's theorem.

**Residual limitation.** ERSE targets weak, near-degenerate factors, which is exactly where Eq
(6.1)'s $(\lambda_i-\lambda_j)^{-2}$ blows up and where regime 2.2 showed the bulk-edge substitution
bias is worst. Some measured "rotation" there may be cross-sectional estimation error, not temporal
movement. The direct ERSE-attribution threat is now rejected for the leading
$P=3$ tangent, but cross-sectional estimation error remains relevant to weaker
modes and should not be declared solved by Regime 4.6.

**Ledoit & Wolf, *Shrinkage Estimation of Large Covariance Matrices: Keep it Simple, Statistician?*,
[SSRN 3421503](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3421503), J. Multivariate
Analysis 186 (2021) 104796** — **[abstract read] [deep read pending]** **Reclassified.** I had this
filed as "the competing school, and the honest question of whether the sophistication pays". That
undersells it badly. The abstract states a key ingredient of the methodology is **a new estimator of
the angle between sample and population eigenvectors, without making strong assumptions on the
population eigenvalues.**

That is an estimator of the same quantity my instrument measures, built in the opposite tradition,
and free of the spectral assumptions that regime 2.2 showed are what actually drive my substitution
bias — a factor of 17 at fixed $N$ and $q$, tracking only how far the sampled bulk edge climbs
toward $\lambda_P$.

*The deep read must answer:* does their angle estimator, applied to my two windows instead of to
sample-vs-population, give a second independent route to $D$ that does not need the true spectrum?
If it does, it is a check on $D_{emp}/D_{th}$ that shares none of my failure modes.

**Blocker, Raja, Fessler & Balzano, *Dynamic Subspace Estimation with Grassmannian Geodesics*,
[arXiv:2303.14851](https://arxiv.org/abs/2303.14851)** — **[abstract read] [deep read pending]**
A **geodesic model for time-varying subspaces**, a (non-convex) objective, an algorithm with a
monotonically non-increasing objective, and parameter estimation from data under
Grassmannian-constrained optimisation. Demonstrated on synthetic, video, and dynamic fMRI data.

This is the closest existing thing to what forecasting $\phi_1^{t+\tau}$ would require: not
tracking a subspace, but **fitting an explicit dynamical model to a moving one and recovering its
parameters.** If the machinery already exists, it is here, and reinventing it would be waste.

*The deep read must answer:* is a geodesic (constant-velocity) model rich enough for what I measure?
Regime 4's excess is roughly 18° — a single geodesic direction and rate is a strong assumption, and
the test for it is whether measured $D(s,t)$ grows linearly in $|t-s|$ once $D_{th}$ is subtracted.
That is a plot I can make from existing outputs before reading a line of their algorithm.

**Oriol, *Asymptotic non-linear shrinkage and eigenvector overlap for weighted sample covariance*,
[arXiv:2410.14420](https://arxiv.org/abs/2410.14420)** — **[abstract read] [deep read pending]**
**Demoted from "second priority", and my note on it was wrong.** The "eigenvector overlap" in the
title is $|u_i^{*}v_j|^2$ between the **sample and the true** eigenvectors — the Ledoit-Péché
$\Theta^g$ functional. It is **not** the two-window temporal overlap, and it is **not** a modern null
for the §5 EWMA instrument. I had written that it was.

What it actually delivers, from the text: asymptotic non-linear shrinkage formulas for the
**exponentially-weighted** sample covariance, with an explicit closed form for $\check\Theta^{(1)}$
under an $\alpha$-exponential weight law, reducing the whole estimation problem to estimating
$\check m$. Plus heavy-tail experiments at $\nu \in \{12, 3, 2.5\}$ finding the theory survives on
bounded fourth moments rather than the twelfth its proofs assume.

Still worth reading, for a narrower reason: if I build the §5 EWMA instrument, $D_{th}$ needs
estimated eigenvalues of an **EWMA** sample covariance, and regime 2.2 showed the substitution bias
is governed entirely by where the sampled bulk edge sits. This paper is the only thing that says
where the EWMA bulk edge sits asymptotically.

*The deep read must answer:* what does the $\alpha$-exponential bulk edge do as $\alpha$ grows, and
does regime 2.2's warning get better or worse under EWMA weighting than under a flat window of the
same effective length?

## Tier C — read before defending what I have

**Bun, Bouchaud & Potters, *Cleaning large correlation matrices: tools from random matrix theory*,
[arXiv:1610.08104](https://arxiv.org/abs/1610.08104), Physics Reports** — **[abstract read]
[deep read pending, scoped]** The RIE review, 165 pages. *"Special care is devoted to the statistics
of the eigenvectors of the empirical correlation matrix, which turn out to be crucial for many
applications."* The framework this project is implicitly arguing with: RIE keeps sample eigenvectors
because it assumes no prior on where the true ones point.

**Scope the read** — this is not a cover-to-cover item. Two sections only: the eigenvector-overlap
material, and the Appendix on **additively** (rather than multiplicatively) corrupted noisy
matrices, which is the $E = C + \mathcal{E}$ setting of Eq (4.1) that the entire null descends from.

**Saad-Falcon, Ancelin & Romberg, *Subspace Tracking with Dynamical Models on the Grassmannian*,
[arXiv:2402.10352](https://arxiv.org/abs/2402.10352)** — **[abstract read] [deep read pending]**
Regularized least squares algorithms built from Grassmann manifold operations and explicit dynamical
models, demonstrated on narrowband beamforming where *"the dynamics of multiple signals of interest
are captured by motion on the Grassmannian."*

Companion to 2303.14851 and the cheaper of the two: it is a menu of dynamical models with
regularisation, where the other is one model fitted properly. Read this second, for the model
choices. It also carries enough array-processing vocabulary to make the DOA literature navigable,
which is why the tutorial survey is no longer on this list.

**Lin & Pan, *Eigenvector overlaps in large sample covariance matrices and nonlinear shrinkage
estimators*, [arXiv:2404.18173](https://arxiv.org/abs/2404.18173)** — **[abstract read] [deep read
pending]** Convergence in probability of sample-vs-true eigenvector overlaps toward their
deterministic counterparts **with explicit convergence rates**, under $M \propto N$, for general
deterministic $D_k$ with bounded operator norm. Plus a sharper characterisation of the loss of
Ledoit-Wolf nonlinear shrinkage.

**Correction:** I had these filed as sample-vs-true only. Riabov et al. cite this paper (their note
[13]) as the first two-resolvent local law **for sample covariance matrices**, "suboptimal and
restricted to non-overlapping samples" — so it does cover the two-window case, at $t=0$, with worse
error terms. Promote accordingly. Kept mainly for **the explicit rates.** Regime 2.2 measured the estimated-spectrum substitution decaying as $T^{-1.34}$
against the $T^{-3/2}$-correction argument's implied $T^{-1/2}$, and I concluded the exponent
probably describes one spectrum at one $N$ rather than a law. This is the paper that can say whether
that is right.

**Attal & Allez, *Eigenvector Overlaps of Random Covariance Matrices and their Submatrices*,
[arXiv:2501.08768](https://arxiv.org/abs/2501.08768)** — **[abstract read] [deep read pending]**
**Reframed.** Singular vectors of any $m \times n$ submatrix of an $M \times N$ Gaussian matrix, and
their asymptotic overlaps with those of the full matrix, in the regime where $N/M$, $m/M$ and $n/N$
all converge. Explicit limiting rescaled mean squared overlaps for both left and right singular
vectors in the bulk, for arbitrary initial matrix $A$; when $A$ is null this is Marchenko-Pastur and
the formulas become Cauchy-like.

I had justified keeping this because it uses Brownian trajectories, like regime 4.4's Brownian
rotation on $O(N)$. That is a coincidence of method, not a shared question, and is not why it stays.
It stays because on an $N \times T$ panel a submatrix is **$m$ assets over $n$ days**, so their
result is the overlap between a sub-panel's covariance eigenvectors and the full panel's — the
**nested** window case (a corollary of 2509.25076 in the $n$ direction), and, in the $m$ direction,
a direct handle on the survivorship problem BUILDNOTES states and does not fix. CAC 40 at 26 names
of 39 is exactly an $m < M$ submatrix.

*The deep read must answer:* how much apparent eigenvector movement does dropping $M - m$ names
manufacture, at my worst ratio ($m/M = 26/39$)? That converts "not fixed, stated" into a number.

**Lamrani, Bongiorno & Potters, *Optimal Data Splitting for Holdout Cross-Validation in Large
Covariance Matrix Estimation*, [arXiv:2503.15186](https://arxiv.org/abs/2503.15186)** —
**[abstract read] [deep read pending]** Closed-form expected estimation error for a white inverse
Wishart population, and the finding that **the optimal train-test split scales as the square root of
the matrix dimension.** For general populations the error is connected to the variance of the
eigenvalue distribution, with approximations.

Relevant to $T^*$ from a different direction, and more concretely than I gave it credit for: it is a
published scaling law relating an optimal window length to $N$, which is the shape of answer the
$T^*$ scan is looking for. The paper's rule and the paper's $T = N$ rule disagree, and that
disagreement is testable.

**Balzano, Nowak & Recht, *Online Identification and Tracking of Subspaces from Highly Incomplete
Information* (GROUSE), [arXiv:1006.4046](https://arxiv.org/abs/1006.4046)** — **[abstract read]
[deep read pending]** Incremental gradient descent on the Grassmannian, linear-time subspace updates.
The canonical online subspace tracker; kept as the single representative of that block (GRASTA,
PETRELS, PAST, Oja are variants addressing robustness and missing data, neither of which is my
problem).

Read for the Grassmannian gradient step only, and for one structural point worth stating before
borrowing anything from this literature: **GROUSE and its descendants assume a true subspace exists
and is being observed through noise, and they track it. That assumption is my null hypothesis.**
A tracker will happily report motion on data where nothing moved, for the same reason regime 1.1
measured $D = 0.0019$ on a world that never changed. Any borrowed tracker needs $D_{th}$ bolted to
it before its output means anything.

---

## Cut

Removed from the reading list. Recorded so a later sweep does not re-triage them.

**Bun, Allez, Bouchaud & Potters, *Rotational invariant estimator for general noisy matrices*
([CFM PDF](https://www.cfm.com/wp-content/uploads/2022/12/241-2015-Rotational-invariant-estimator-for-general-noisy-matrices.pdf))** — I had this as a "practitioner-facing version" of the review; it is
actually a distinct Replica-method paper covering additive and multiplicative rotational-invariant
perturbations. Cut anyway: its additive-noise content is what the Physics Reports Appendix already
covers, and reading both is duplicated effort. If the 1610.08104 Appendix turns out to be thin,
promote this back.

**Schmidt, MUSIC (1986), and Jeong, Son & Lee, *Asymptotic Performance Analysis of the MUSIC
Algorithm* ([ETRI PDF](https://ksp.etri.re.kr/ksp/article/file/62008.pdf), Applied Sciences 2020)** —
The ETRI paper does derive a closed-form MSE from how sample-covariance eigenvectors relate to true
ones, so my description of it was accurate. Cut because what it produces is a **DOA-specific
azimuth MSE for a uniform linear array**, one signal at a time — first-order eigenvector perturbation
in a form specialised away from anything I can use. My null is already validated to 0.25% (regime
1.1) and to 0.5% (regime 2.1); a second derivation of the same first-order object buys nothing, and
there is no two-window subspace distance anywhere in it.

**Direction of Arrival Estimation: A Tutorial Survey,
[arXiv:2508.11675](https://arxiv.org/abs/2508.11675)** — A beginner tutorial with Python
implementations, covering beamforming, MUSIC/ESPRIT, ML and sparse methods. No new result, and the
vocabulary I wanted from it comes free with 2402.10352, which is itself a beamforming paper. Cut as
redundant.

**GRASTA / GREAT / PETRELS / PAST / Oja** ([Balzano's
page](https://web.eecs.umich.edu/~girasole/?page_id=190), [De Groat et al., *Subspace
Tracking*](https://dsp-book.narod.ru/DSPMW/66.PDF)) — Variants of the same online-tracking idea,
addressing outlier robustness ($\ell_1$), missing data, and linear time-varying systems. None of
those is my problem. GROUSE stands in for the block; the links stay here if the tracking route is
ever actually taken.

**Anderson orthogonality catastrophe** — Already understood via 1203.6228 §2, and the only thing it
decides is that $Q > P$ is necessary rather than a convenience, which is a settled design choice
that no open question depends on. Nothing to read.

**Zanardi & Paunković, *Ground state overlap and quantum phase transitions*,
[quant-ph/0512249](https://arxiv.org/abs/quant-ph/0512249)** — Characterises quantum phase
transitions by the overlap between ground states at two parameter values, on the Dicke and XY models,
and explicitly connects to Anderson orthogonality and the Loschmidt echo. Structurally the same
object as $\langle\phi^s|\phi^t\rangle$, and that is the whole of the relationship: the models share
no structure with a sample covariance matrix and there is no formula that transfers. The analogy is
already stated on p.5 of 1203.6228 and does not need a second source.

**García-Mata & Wisniacki, *Quantum analogues of exponential sensitivity: from Loschmidt echo to
Krylov complexity*, [arXiv:2604.12707](https://arxiv.org/abs/2604.12707)** — A textbook chapter for
the Quantum Chaos volume of Elsevier's *Comprehensive Quantum Mechanics*, surveying Loschmidt echo,
OTOCs and Krylov complexity. The Loschmidt echo is the overlap of one state evolved under two
Hamiltonians; my $D$ is the overlap of two bases **estimated from finite samples**. The entire
difficulty of this project is the estimation noise, which has no counterpart there — no null, no
estimator, nothing that crosses over.

**Level repulsion and avoided crossings** — Cut, and worth recording *why*, because it stops a
wasted experiment. The open question I had written down was "a market's spectrum sits somewhere on
the Wigner-to-Poisson axis and nobody seems to have asked where." The reason nobody has asked is
that the answer is mostly predetermined: the spacings I could measure are those of a **sample**
covariance matrix, whose bulk shows Wigner-Dyson repulsion because the *sampling* is random,
largely independently of what $C$ is. I would be measuring my own estimator, not the market. The
pseudo-collision argument of §2 stands as the justification for using blocks rather than single
vectors, and needs no citation beyond 1203.6228.

---

## What survives, and what it changes

1. **The temporal question is narrower than it was, not closed.** 2509.25076 answers the
   overlapping-window case exactly, and it is Bouchaud's own group, so the two-window overlap
   *distribution* is not open territory. What remains open is forecasting: nobody in that line
   predicts $\phi^{t+\tau}$.
2. **The machinery for the forecast exists in signal processing**, in 2303.14851 and 2402.10352
   specifically, and it fits dynamical models to moving subspaces rather than merely tracking them.
   But every algorithm in that literature assumes motion and would report it on static data, so
   none of it is usable without $D_{th}$ attached.
3. **Two papers may already answer questions I have open**, and neither is the pair I named
   yesterday. 2509.25076 on overlapping windows still stands. Its partner is **1301.4939**, not
   2410.14420 — a non-perturbative treatment is what the small-$T$ and bulk-edge failures need, and
   2410.14420 turned out to be about sample-vs-true overlaps under EWMA weighting, which is a
   different object than I recorded.
4. **ERSE and regime 3.1 share a geometric operator.** Both use pairwise Givens
   rotations, but ERSE is an iterative estimator with its own pair selection,
   stopping rule and Rayleigh-eigenvalue update. Regime 4.6 implements it and
   rejects direct ERSE attribution of the temporal tangent signal.
5. **Gap I have not filled.** Nothing on this list is a *two-sample hypothesis test for principal
   subspaces* from the statistics literature, which is formally what $D_{emp} - D_{th}$ is. The RMT
   line gives exact distributions under a null of no change; the statistics line gives tests with
   stated size and power. Regime 3.1's power curve was built by simulation because I do not know
   whether that test already exists. Worth one search sweep before claiming the instrument is novel.

   Two leads surfaced while triaging, **titles only, abstracts not yet checked** — do not cite
   either until read:
   - *Estimating eigenvectors and eigenspaces of covariance matrices: optimal bounds and conditions
     for consistency*, [arXiv:2607.23964](https://arxiv.org/abs/2607.23964). If this is what the
     title suggests, it is the statistics-side answer to when a measured eigenspace means anything,
     which is the question regime 3.1's power curve answers by brute force.
   - Bun, Bouchaud & Potters, *On the overlaps between eigenvectors of correlated random matrices*,
     Phys. Rev. E **98**, 052145 ([CFM PDF](https://www.cfm.com/wp-content/uploads/2022/12/212-2016-On-the-overlaps-between-eigenvectors-of-correlated-random-matrices.pdf)).
     Two *correlated* matrices rather than one sample and its population — possibly the missing
     link between the RIE review and Eq (6.1), and possibly the thing 2509.25076 generalises.
