# Prior art — what has been done, and where this project actually sits

*2026-08-02. Deep reads, not abstract scans. Read at source: full text of 2509.25076 (incl.
supplementary), 1301.4939, 2205.01012, 1603.04364/PRE 98 052145, and operative §1–§5 of 2507.01545.
The Stage 2 geometry sweep additionally read the operative sections of the primary papers, including
their definitions, objectives, algorithms, experiments and stated limitations. Abstract-only items are
marked as such throughout — I have not padded this with things I did not read.*

---

## The headline question

> *Has anyone used ML to predict eigenvector rotation, and used it — most notably to add a rotation
> factor to RIE?*

**Not in that precise combination. But the old answer — an unqualified “no” — was wrong.** Three
neighbouring problems have already been solved:

1. Signal processing has explicitly predicted temporally correlated subspaces on the Grassmann
   manifold since at least 2011.
2. Modern ML now performs supervised subspace regression, including prediction of eigenspaces.
3. Financial ML has forecast the complete realised covariance matrix on the SPD manifold, thereby
   forecasting its eigenvectors implicitly even though it never isolates or evaluates their motion.

What I have **not** found is the conjunction that defines this project: estimate a financial leading
eigenspace from noisy finite windows; use an RMT null to separate sampling rotation from latent
motion; learn the residual trajectory; and show that the predicted basis improves a cleaned
covariance out of sample. That is the defensible gap. “Nobody has predicted eigenvectors” is not.

One important branch of the ML-*cleaning* literature has converged on a shared design choice,
stated openly in each paper: **learn the eigenvalue map, keep the empirical basis.**

| Paper | What it learns | Eigenvectors |
|---|---|---|
| Manolakis, Bongiorno & Mantegna, [2601.07687](https://arxiv.org/abs/2601.07687) (Jan 2026) | nonlinear map from empirical singular values to cleaned singular values | *"operates in the empirical singular-vector basis"* — untouched |
| Bongiorno, Manolakis & Mantegna, [2507.01918](https://arxiv.org/abs/2507.01918) | lag-transform of returns + eigenvalue regularisation + marginal vols, GMV loss end-to-end | explicitly a **rotation-invariant** network |
| Bongiorno, Challet & Loeper, [2111.13109](https://arxiv.org/abs/2111.13109) | a set of **time-independent** eigenvalues encoding the average influence of the future on present eigenvalues | untouched |

Read those three together and the shape of the gap is exact. 2111.13109 is the most pointed: it is
a paper explicitly about **non-stationary** covariance, by people who know the non-stationarity is
real, and its entire answer is *"stop letting the eigenvalues depend on the input."* It does not
ask whether the eigenvectors move. 2601.07687 goes further — it finds that analytical cleaners fail
out-of-sample precisely because *"dependence structures drift over time"* — diagnoses drift as the
problem, and then fixes it with a learned **eigenvalue** map.

Read those as the immediate baselines, not the limits of ML. The broader literature below crosses
the eigenvector line. The honest statement of the gap is therefore:

> The covariance-*cleaning* papers that diagnose drift repair it on the eigenvalue axis, while
> geometric ML can predict subspaces and financial geometric ML can predict whole covariance
> matrices. No paper found connects those pieces through an RMT-calibrated, explicitly evaluated
> forecast of the financial eigenbasis.

Two caveats, because a negative result deserves them.

**First, "rotation factor added to RIE" is close to a contradiction in terms, and that is not an
accident.** RIE is *defined* by rotational invariance — no privileged direction, therefore the
estimator must be $\hat{C} = \sum_k \xi(\lambda_k)\, u_k u_k^\top$ with the sample $u_k$. The
moment you rotate, you have left the class and lost its optimality guarantee, which is optimality
*within* that class. Anything you build has to be justified on its own terms — out-of-sample
performance — not by inheriting RIE's theory. That is a real cost and worth being clear-eyed about.

**Second, ERSE already broke rotational invariance, and its own result argues the rotation may be
doing less than it looks.** See below.

---

## The geometry question has an answer — and it changes Stage 2

The state space is not an open problem. If $U_t\in\mathbb{R}^{N\times P}$ is any orthonormal basis
for the leading $P$-dimensional eigenspace, the object of interest is the equivalence class
$[U_t]=\{U_tR:R\in O(P)\}$, i.e. a point $Y_t$ on the Grassmann manifold
$\mathrm{Gr}(N,P)$. Learning the columns of $U_t$ with an ordinary Euclidean loss is wrong because
sign flips and rotations within the same subspace change the matrix without changing the object.

The clean one-step geometry is already standard. At the current subspace $Y_t$, define

$$H_t^-=-\mathrm{Log}_{Y_t}(Y_{t-h}),\qquad
H_t^+=\mathrm{Log}_{Y_t}(Y_{t+h}).$$

Both are elements of the **same** tangent space $T_{Y_t}\mathrm{Gr}(N,P)$. $H_t^-$ is the previous
velocity parallel-transported to the present; $H_t^+$ is the future velocity to be predicted. This
immediately gives the non-ML persistence test and the mandatory forecasting baseline:

$$\cos_t=\frac{\langle H_t^-,H_t^+\rangle}
{\lVert H_t^-\rVert_F\lVert H_t^+\rVert_F},\qquad
\widehat Y_{t+h}^{\rm cv}=\mathrm{Exp}_{Y_t}(H_t^-).$$

The second expression is constant-velocity geodesic extrapolation. Inoue & Heath derived this as
one-step Grassmannian prediction for $P=1$ in 2011. Saad-Falcon, Ancelin & Romberg give the
general-$P$ position/velocity formulation and the useful identity above; their first-order chordal
version is simply a second difference of projectors. Therefore **parallel transport, logarithms,
exponentials, a persistence score, and a no-ML predictor are not research contributions here.** They
are machinery to use and baselines to beat.

### What the ML model should predict

The smallest defensible Stage 2 model is not a large network that emits $NP$ unrelated numbers. It
is a geometry-aware autoregression in the current tangent space:

$$\widehat H_t=\sum_{j=1}^{L}a_j(x_t)\,
\Gamma_{t-j\rightarrow t}H_{t-j},\qquad
\widehat Y_{t+h}=\mathrm{Exp}_{Y_t}(\widehat H_t),$$

where the transported past velocities supply the admissible directions and a small model learns the
scalar weights $a_j$ from information available at $t$. This is the data-scarce version of the
Riemannian AR construction: it respects the changing tangent spaces, cannot manufacture an invalid
subspace, and nests the constant-velocity baseline as $L=1,a_1=1$. Only after this model establishes
held-out gain is a free-form tangent network justified.

For a direct network output, predict any full-rank matrix $A_\theta(x_t)\in\mathbb{R}^{N\times Q}$,
orthonormalise it by QR, and train on the resulting subspace rather than on its chosen basis. The
stable containment loss for a future $P$-space inside a predicted $Q$-space is

$$L_{P\subset Q}=P-\left\lVert
U_{t+h}^{(P)\top}\widehat U_{t+h}^{(Q)}\right\rVert_F^2
=\sum_{i=1}^{P}\sin^2\theta_i.$$

This is the projector/chordal loss in Fanaskov et al., *Deep Learning for Subspace Regression*
([arXiv:2509.23249](https://arxiv.org/abs/2509.23249), ICLR 2026). They prove that allowing
$Q>P$ can reduce the derivative of a smooth Grassmann-valued target and show empirically that the
redundancy can improve accuracy and generalisation. That gives the existing $P,Q$ design a new and
very concrete interpretation: **$Q$ is not only a collision buffer for measurement; it can be a
learnability buffer for prediction.** Their conclusion is also a warning: exact subspace regression
was too complicated in many of their applications even with a specialised loss, and the larger
containing space was what made it tractable.

$D(P,Q)$ and $L_{P\subset Q}$ use the same principal angles but should not do the same job.
$L_{P\subset Q}$ is bounded and supplies well-behaved training gradients. The log-determinant

$$D(P,Q)=-\frac1P\sum_{i=1}^{P}\log\cos\theta_i$$

heavily penalises the worst missed direction and diverges at orthogonality. Retain it as the
RMT-calibrated scientific score and an evaluation metric; do not make it the first training loss.

### Grassmann, flag, or SPD?

There are three different targets, and conflating them creates a fake geometry problem.

| Scientific target | Correct space | What it identifies |
|---|---|---|
| leading $P$-dimensional span only | $\mathrm{Gr}(N,P)$ | a subspace, not an ordered basis |
| nested market/core/buffer spaces | partial flag manifold | $\mathcal U_1\subset\mathcal U_P\subset\mathcal U_Q$ |
| the complete covariance forecast | SPD manifold | eigenvalues and eigenvectors jointly |

A partial flag is the closest geometry to what this project actually measures. It preserves the
market direction separately, treats near-degenerate modes as blocks rather than unstable named
vectors, and retains the $Q$-dimensional collision buffer. The natural first signature for the
current experiments is $\mathrm{Flag}(N;1,P,Q)$, with a weighted sum of projector losses at its
three nested levels. Szwagier & Pennec's *Nested Subspace Learning with Flags*
([arXiv:2502.06022](https://arxiv.org/abs/2502.06022), JMLR 2026) supplies the nested-projector
construction; Jin & Coulson's *Online Subspace Learning on Flag Manifolds for System
Identification* ([arXiv:2511.06416](https://arxiv.org/abs/2511.06416), L4DC 2026) shows that flags
can be updated from streaming data and used in an adaptive prediction pipeline.

Regime 4.7 now implements the first empirical version of that proposal as the
nested-projector embedding of $\mathrm{Flag}(N;1,3,6)$. It extracts one top-six
frame per window, retains its cumulative dimensions 1, 3 and 6, computes a
Grassmann log at every level, and combines the tangent tuple with
inverse-dimension weights. This is basis-invariant and faithful to the nested
information, but it should be named precisely: it is not an implementation of
the intrinsic quotient-manifold flag logarithm. The gate validates the target;
Stage 2 will still need a constraint-preserving update or retraction for flag
predictions.

The flag does **not** magically solve covariance reconstruction. A $Q$-space containing the future
$P$-space does not say which directions inside it receive distinct eigenvalues. The principled
resolution is to choose flag blocks at reproducible spectral gaps, shrink eigenvalues within a
near-degenerate block toward a common value, and predict only the orientation between blocks. Inside
an exactly degenerate block the basis is unidentifiable and economically irrelevant; ERSE's
$(\lambda_i-\lambda_j)$ factor says approximately the same thing near degeneracy. Whether empirical
gap-based blocks are stable enough to improve realised risk is a project question, not a settled
geometric theorem.

### The closest precedents, and the exact remainder

| Paper | What the old wizard already built | What remains here |
|---|---|---|
| Inoue & Heath, [1105.5782](https://arxiv.org/abs/1105.5782) (2011) | log/exp, parallel transport, direction/magnitude split and one-step prediction | $P=1$ communications channel; no noisy covariance estimate or RMT null |
| Yang & Hospedales, [CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Yang_Multivariate_Regression_on_CVPR_2016_paper.html) | kernel and extrapolative regression from covariates to future Grassmann points | parametrised visual domains, not a stochastic financial subspace time series |
| Blocker et al., [2303.14851](https://arxiv.org/abs/2303.14851) (2023) | batch estimation of a geodesic dynamic subspace from noisy observations | fits/denoises observed data; does not test an out-of-sample subspace forecast |
| Saad-Falcon et al., [2402.10352](https://arxiv.org/abs/2402.10352) (2024) | general-$P$ constant-position/velocity dynamics and cheap projector approximation | tracking under an assumed motion model; no test that measured finance motion exceeds sampling noise |
| Figueras & Persson, [2509.24767](https://arxiv.org/abs/2509.24767) (2025) | AR($N$) processes on Stiefel/Grassmann manifolds with estimated orthogonal-group actions | a global $O(N)$ transition is far too parameter-heavy for $N\approx500$ and single-digit independent windows |
| Fanaskov et al., [2509.23249](https://arxiv.org/abs/2509.23249) (ICLR 2026) | neural eigenspace regression, invariant losses, and larger containing outputs | i.i.d. parametric problems with clean targets, not temporal covariance eigenspaces corrupted by Wishart noise |
| Bucci, Palma & Zhang, [2412.09517](https://arxiv.org/abs/2412.09517) (2024) | financial realised-covariance forecasting with an SPD network and geometric loss | forecasts the whole matrix but never isolates eigenvector skill, subtracts an RMT null, or compares a predicted basis with RIE |
| Szwagier–Pennec and Jin–Coulson (2026) | nested and online subspace learning on flag manifolds | no spectral-gap/RMT-calibrated financial flag forecast |

The geometry is therefore **not** the research question. The remaining research question is:

> After removing finite-window overlap noise, is there enough transported tangent-direction signal
> in financial leading eigenspaces to beat static and constant-velocity Grassmann baselines, and
> does resolving only economically identifiable flag blocks improve covariance risk beyond
> eigenvalue shrinkage?

That is narrower than the original claim and materially stronger. A negative result is still a result:
it would say that the correct geometry and published predictors fail once estimation noise is
calibrated, not merely that an arbitrary neural architecture failed.

---

## The one that matters most: ERSE is an explicit comparator, not an automatic collapse

[arXiv:2507.01545](https://arxiv.org/abs/2507.01545), Liu & Liu. Operative
sections §1–§5 read in full.

The prior is **Perron–Frobenius**, not "positive comovement" loosely. Under an all-positive
correlation matrix, the dominant eigenvector aligns with the uniform vector $\mathbf{1}/\sqrt{n}$
and every other eigenvector, being orthogonal to it, is pushed toward the uniform vector's null
space. They prove the dominant eigenvector's *deviation* from that null space is much larger than
the others', and by more than the eigenvalue gap would suggest. ERSE then **imposes a floor on that
deviation**: any eigenvector deviating too little gets rotated, pairwise (their PER technique),
until every eigenvector clears the threshold. At most $n-1$ iterations.

Which eigenvectors deviate too little? The **weak factors** — the near-degenerate ones.

Their Proposition 5 gives the precise scope of the shrinkage statement. After a
pairwise rotation, the two recomputed Rayleigh eigenvalues narrow their gap while
maintaining their sum. That is an exact identity for the pair's updated scalar
eigenvalue estimates. The full ERSE output also uses the rotated eigenvectors,
so it is too strong to say that the entire estimator “collapses into” ordinary
eigenvalue shrinkage in the original basis.

**Read that as a warning about your own headline.** A Givens rotation of the pair $(q_i, q_j)$ by
$\theta$ produces, in the original basis, a $2\times2$ block

$$\begin{pmatrix} \lambda_i\cos^2\theta + \lambda_j\sin^2\theta & (\lambda_i-\lambda_j)\sin\theta\cos\theta \\ (\lambda_i-\lambda_j)\sin\theta\cos\theta & \lambda_i\sin^2\theta + \lambda_j\cos^2\theta \end{pmatrix}$$

The diagonal is exactly linear shrinkage of $(\lambda_i,\lambda_j)$ toward each other, sum
preserved. The information genuinely new to the rotation lives **only in the off-diagonal term**,
and that term carries the factor $(\lambda_i - \lambda_j)$ — *it vanishes as the two modes become
degenerate.* Which is precisely where ERSE chooses to act.

So: **on near-degenerate modes, the eigenvalue effect of rotating and shrinking
is hard to distinguish, but the basis effect remains a concrete geometric
object.** The right confound test is therefore not a verbal equivalence and not
a blind subtraction. Build ERSE itself, form its tangent direction at the
current subspace, and ask how much of the temporal tangent lies along it.

Regime 4.6 now does exactly that on the four full-history panels. At Liu & Liu's
primary $\delta=0.25$, ERSE accounts for only 0.05–0.92% of outgoing tangent
energy; after projecting it out, the persistence cosine is unchanged and still
beats the matched block-shuffle null on every panel. The measured temporal
signal is therefore **not ERSE rearranged**. A separate top-to-complement
covariance-transition share did not exceed its null, so incremental covariance
risk value is not yet established and remains an honest Stage 2 test.

Regime 4.7 then asks whether this conclusion survives after replacing the lone
top-three Grassmann state by the nested market/core/buffer flag. It does:
complete-flag persistence beats its Holm-adjusted calendar null on Nikkei, DAX
and CAC, while S&P is strongly separated but resolution-limited by only 20
calendar nulls; complete-flag coherence passes on all four panels. ERSE explains
only 0.11–1.32% of outgoing complete-flag tangent energy at $\delta=0.25$, and
the residual persists. The important qualification is component-specific: the
outer six-space is weak on DAX and borderline on CAC, so it is supported as a
containing buffer, not as six equally predictable directions.

Regime 4.9 later narrows that temporal interpretation. The earlier tangent
combined the deterministic deletion of 42 known observations with the addition
of 42 unseen returns. Once every transition is based at its retained-observation
Flag, deletion accounts for a mean 39–45% of per-origin outgoing tangent energy. Incoming-block
persistence survives both calendar and volatility-matched return nulls on S&P
and in the equal-market aggregate; Nikkei is borderline before multiplicity
correction, while DAX and CAC fail. The literature comparison still supports
the Flag representation and the distinction from ERSE, but no longer supports
a universal four-market forecastability claim.

Also correcting the record on their empirics: **19 Ken French factor-sorted datasets, monthly,
July 1969 – June 2024 (660 months), $N$ from 30 to 654, rolling window 120 months.** Average
pairwise correlation 0.56–0.83. So $q = N/T$ runs up to $654/120 \approx 5.5$. That is not your
regime in any respect — different frequency, different $q$, and a correlation level no equity
cross-section reaches. The 12.46% does not transfer.

---

## The exact null already exists, and it is not Eq (6.1)

This is the biggest correction to the reading list. **[arXiv:2509.25076](https://arxiv.org/abs/2509.25076)
cites Allez–Bouchaud 2012 only as where the subject was *"first discussed"*.** The paper it builds
on is:

**Bun, Bouchaud & Potters, *Overlaps between eigenvectors of correlated random matrices*,
[arXiv:1603.04364](https://arxiv.org/abs/1603.04364), Phys. Rev. E 98, 052145 (2018).** Read in
full, including appendices.

They give **general, exact, closed-form** mean squared overlaps
$\Phi(\lambda,\lambda') = N\,\mathbb{E}(u_\lambda \cdot u'_{\lambda'})^2$ between the eigenvectors
of two sample covariance matrices measured on two **non-overlapping** windows — their Eq (6), with
the self-overlap at Eq (9). Additive-noise versions at Eqs (8) and (10). The remarkable property,
which they emphasise: **the formulas do not involve $C$ at all.** They depend only on the empirical
Stieltjes transform. And they say outright:

> *"This leads us to propose a statistical test based on these overlaps, that allows one to
> determine whether two realisations of the random matrix $S$ and $S'$ indeed correspond to the very
> same underlying 'true' matrix $C$."*

This is exact but it is **not yet the block log-determinant null used here**. For
$G=U_P^\top U'_Q$, the BBP formulas determine pairwise second moments
$\mathbb E|G_{ij}|^2$ and therefore control Frobenius/trace overlap quantities. The project scores

$$D(P,Q)=-\frac{1}{2P}\log\det(GG^\top),$$

which depends on the **joint law of all singular values of $G$**, including their correlations and
the lower tail near a missed direction. Pairwise means do not determine that law, and
$\mathbb E\log\det(GG^\top)\neq\log\det\mathbb E(GG^\top)$. What is missing at the annotation is
therefore a finite-$N$ joint block-overlap law, or a controlled closure that is accurate for this
nonlinear statistic. BBP supplies indispensable inputs and a competing interpretation of the data;
it does not by itself give the expected $D(P,Q)$.

**That is your instrument, proposed in 2016.** Two things save the project's ground, and they are
both real:

1. **They explicitly exclude the top eigenvectors.** Their Fig. 1 covers bulk eigenvalues only,
   with the aside *"the top eigenvectors are governed by non trivial dynamics, see e.g. [16]"* —
   where [16] is Allez–Bouchaud 2012. Your instrument is a **top-block** statistic ($P=3$, $Q=6$).
   The exact theory and your measurement address disjoint parts of the spectrum, and the exact
   theory defers the part you work on back to the perturbative paper.
2. **They use the departure as a spectroscopy of $C$, not as a non-stationarity test.** Their real-data
   conclusion from 450 US stocks (2005–2012, 1800 days split into two non-overlapping $T=900$
   halves, 100 bootstraps) is that the bulk overlaps depart from the null because the true spectrum
   has structure — they fit an inverse-Wishart prior, $\kappa \approx 0.7$. They do not claim
   the market moved. **Same measurement, opposite reading.** Anyone assessing your work will ask why
   your excess is evolution rather than mis-specified $C$, and BBP is the reason they will ask.

Their closing line is the setup for the 2025 paper: *"The multiplicative model is also interesting
since it describes the case of correlation matrices measured on overlapping periods... This case
turns out to be more subtle and is the subject of ongoing investigations."* Nine years later:

**Riabov, Tikhonov & Bouchaud, 2509.25076.** Read in full. They handle intersecting windows by
Girko linearisation of a block matrix plus two-resolvent local laws. The intersection enters through
a single scalar

$$t := \frac{T_B\sqrt{q\tilde q}}{N} \in [0,1], \qquad t = \mathrm{Corr}(XX^\top + BB^\top,\; \tilde X\tilde X^\top + BB^\top)$$

where $T_B$ is the shared block length. Their central result is Eq (28), and **at $t = 0$ it reduces
exactly to BBP 2018.** So `min_lag = T` can be dropped: $t$ is the knob, and every window pair at
$\tau < T$ becomes usable rather than being thrown away.

Three things from their empirics you should know:

- **They see what you see.** Fig. 3, real returns, $N=300$ US stocks 2004–2013, bootstrapped at
  $T=600$: *"the displacement... is due to the strong non-stationarity of the underlying population
  covariance on a timescale comparable to $N/q = 600$."* Non-stationarity detected via eigenvector
  overlaps on real equity data, by Bouchaud, in 2025. Your regime 4 headline is **not** new as a
  qualitative claim.
- **But it is entirely qualitative.** They eyeball a discrepancy in a scatter plot. There is no
  excess statistic, no null-subtracted quantity, no detection threshold, no power curve, no angle.
  You have all five.
- **They hand you your own contribution, in writing.** Their closing paragraph:

  > *"finite-size effects may also mimic non-stationarity and are expected to be more pronounced in
  > the presence of heavy tails, as is often the case for financial data... Determining whether our
  > results extend to such settings remains an open problem."*

  **Regime 1.5 and regime 3.1 are an answer to that sentence.** You have the $\frac{\nu-2}{\nu-4}$
  scaling verified to 2.5% across $\nu = 6..\infty$; you have the finding that Eq (4.7) is the right
  correction for a *mean* and the wrong one for a *threshold* because tails fatten the upper tail
  faster than the mean; and you have a remedy — standardise at window=1 — that removes the
  $\nu$-dependence completely at every $N$ tested, with no $\nu$ to fit. That is a direct, dated,
  quantitative response to a stated open problem in a September 2025 CFM paper. **Lead with it.**

---

## Where your $D$ actually comes from, and the one thing genuinely still open

**[arXiv:1301.4939](https://arxiv.org/abs/1301.4939), Allez & Bouchaud, *Eigenvector dynamics under
free addition*.** Read in full. **This was the single most under-rated item on the list and §7 is
why.**

Section 7 constructs your statistic explicitly. Overlap matrix $G_t(ij) = \langle\psi^t_i|\phi_j\rangle$
of dimensions $Q \times P$; the volume of the projected parallelepiped
$v(t) = (\det(G^\dagger G))^{1/2}$; and

$$D(V_0, V_1^t) = -\ln\left(\det(G_t^\dagger G_t)\right)^{1/2P} = -\frac{1}{P}\sum_{k=1}^{P}\ln s_k$$

They note it *"already appeared in the literature on the Anderson orthogonality catastrophe."* Their
$\delta$ is a margin widening the second interval specifically *"to truncate the singularity induced
by pseudo collisions at the edge of the intervals"* — the same object as your $Q > P$ gap, and the
same sensitivity regime 2.2 measured as a factor of 17 driven by bulk-edge position.

**What they prove:** Eq (7.1) — the perturbative form — holds in the **semi-perturbative** regime
($t_N \to 0$, $N \to \infty$, *without* requiring $Nt_N \to 0$). That is a genuine extension over the
2012 paper, and the mechanism is that off-diagonal entries of $G^\dagger G$ stay negligible so
$\det$ factorises into the diagonal product.

**What they explicitly do not prove — the last paragraph of the paper:**

> *"The reader may wonder how to extend formula (7.1) in the non perturbative regime, i.e. for
> arbitrary values of $t$. This question is clearly more difficult as one would need to understand
> the convergence of the non diagonal terms of the matrix $G^\dagger_t G_t$ in the large $N$ limit,
> which are no longer negligible in the determinant expansion."*

So the state of play is precise, and better for you than I expected:

| object | non-perturbative result | status |
|---|---|---|
| single-eigenvector overlap $\mathbb{E}\langle\psi^t_i\|\phi_j\rangle^2$ | **yes** — their Thm 4.3, Eq (4.5); Cauchy-flight form Eq (6.6) | solved 2013 |
| two-window spectral overlap $\Phi(\lambda,\lambda')$ | **yes** — BBP 2018, exact and $C$-free; overlapping case 2509.25076 | solved |
| **block subspace distance $D(P,Q)$** | **no** — blocked by off-diagonal $G^\dagger G$ | **open since 2013** |

**Your DAX and CAC results sit exactly on that open problem.** Regime 4.1 has both panels *below*
their own null at $T = 29$ — 0.6× and 0.5× raw, 0.3× and 0.2× standardised — and you wrote that this
means Eq (4.8) is wrong in that corner without a mechanism. Allez & Bouchaud name the mechanism:
when the perturbative regime is left, the off-diagonal terms of $G^\dagger G$ stop being negligible
in the determinant. Your $D_{th}$ is built from the diagonal-product approximation. At $T = 29$ you
are outside its stated domain of validity, and that is a citable statement rather than an anomaly.

Caveat on the mapping. 1301.4939 is **additive** GOE perturbation, $M(t) = A + H(t)$; their $t$ is
noise amplitude, not calendar time. Your setting is multiplicative (sample covariance from finite
windows). The correspondence runs through $E = C + \mathcal{E}$, Eq (4.1) of 1203.6228, with
$t \leftrightarrow$ something of order $1/T$. **Establish that mapping explicitly before quoting any
of their formulas** — it is the step most likely to embarrass you.

---

## The branch CFM took instead — and why you should know it

**Bouchaud, Mastromatteo, Potters & Tikhonov, *Excess Out-of-Sample Risk and Fleeting Modes*,
[arXiv:2205.01012](https://arxiv.org/abs/2205.01012), Wilmott 2022.** Read in full. This was not on
your list at all and it is the closest thing to a competitor.

The intro is blunt about the lineage. Ref [14] is BBP 2018 — *"a non parametric method based on the
overlap of the eigenvectors of $\mathbb{E}_{in}$ and those of $\mathbb{E}_{out}$... did not require
the knowledge of $C$, only that it was time independent."* Then:

> *"In this note, we want to propose an **alternative** non parametric test, **simpler and more
> transparent**."*

**CFM moved off the eigenvector-overlap route for detection.** What they moved to:

$$\mathbb{D} := \mathbb{E}_{\text{in}}^{-1/2}\,\mathbb{E}_{\text{out}}\,\mathbb{E}_{\text{in}}^{-1/2}$$

The characteristic polynomial of $\mathbb{D}$ equals that of $W_{in}^{-1}W_{out}$, so **its
eigenvalues are exactly independent of $C$** — no asymptotics, no approximation, a two-line
argument. The null is a Wishart × inverse-Wishart (Jacobi-type) density, their Eq (6), with closed-form
edges Eq (7). Eigenvalues above $\lambda_{max}$ are portfolios that over-realise their risk; the
corresponding eigenvectors are the **fleeting modes**.

Their empirics, and how they line up with yours:

- $N=98$ futures 2006–2022, $N=300$ US stocks 2002–2022. $q_{in} = 1/4$, $q_{out} = 4$ —
  deliberately asymmetric: long in-sample, short out-of-sample, tuned to catch abrupt shifts.
- **They standardise the same way you do, for the same stated reason:** each daily return divided by
  its own intraday volatility (Garman–Klass on OHLC), *"in order to get rid of any spurious
  volatility fluctuations and only focus on correlations."* Your regime 1.5 / 2.3 remedy is house
  practice at CFM. **What you have that they do not is the derivation** — that fat tails and vol
  clustering are one law, $\mathbb{E}[c^2]/\mathbb{E}[c]^2$, and that standardising removes both
  with no free parameter. They do it because it works; you can say why.
- **Finite-$N$ corrections are large and they say so:** the upper edge shifts by
  $\Delta_N = (cN)^{-2/3}$ with $c \approx 2.7\times10^{-3}$, giving $\Delta_N \approx 2.4$ at
  $N=98$ against $\lambda_{max} = 13.97$. A 17% correction to the threshold from finite $N$ alone.
  Precedent for taking your small-panel problems seriously rather than as a defect.
- Result: significant departures in both universes; excess risk loads on **low** in-sample risk modes
  for futures, **high** for equities; **momentum** identified as a source of equity excess risk.

**Why this matters strategically.** The fleeting-modes statistic answers *"is there excess risk, and
in which portfolio directions"*. Yours answers *"has the leading subspace rotated, and by how many
degrees, against a calibrated detection floor"*. Those are different questions and yours is the one
that supports a forecast — you cannot extrapolate $\mathbb{D}$'s top eigenvector forward in any
natural way, but a subspace with a measured angular velocity is exactly what a Grassmannian geodesic
model takes as input. **That is the cleanest statement of the project's position I can give you:
the overlap route is the one you can forecast with, and it is the one CFM set down in 2022 because
it was harder.**

---

## Eigenvector filtering already exists, and it is the same group

*2026-08-02, read in full including SI.*

**Bongiorno & Challet, *Covariance matrix filtering with bootstrapped hierarchies*,
[arXiv:2003.05807](https://arxiv.org/abs/2003.05807).** Same Bongiorno as 2601.07687 and
2507.01918 at the top of this document.

The narrower claim at the top — that one branch of ML *cleaning* learns the eigenvalue map and
keeps the empirical basis — is right about those three papers and wrong as a statement about either
ML generally or the group. BAHC bootstraps
the returns, applies hierarchical clustering with average linkage to each copy, and averages
the filtered matrices. They say outright that eigenvectors are the harder axis and then work
on it anyway:

> *"Eigenvector filtering is more complex. However, ansätze for the shape of the true
> correlation matrix impose constraints on the structure of the eigenvectors and of the
> eigenvalues."*

The measured consequence is **eigenvector persistence**: BAHC eigenvectors overlap the
out-of-sample ones better than the sample eigenvectors do, and that is their stated mechanism
for beating RIE/QuEST/Ledoit-Wolf on realised minimum-variance risk whenever $q = N/T \gtrsim
1/3$. Their Appendix B builds the Oracle estimator $\Xi^{in} = U^{in} Z^{in} U^{in\dagger}$
with $Z^{in} = \mathrm{diag}(U^{in\dagger} C^{out} U^{in})$, notes it equals $C^{out}$ exactly
when $U^{in} = U^{out}$, and therefore reads $\|C^{out} - \Xi^{in}\|_F$ as an in/out
eigenvector overlap. **That is a sibling instrument to $D(P,Q)$** — Frobenius over the whole
spectrum rather than a log-determinant over the top block, but the same quantity in spirit.

Two things follow.

1. **Narrow the novelty claim.** "Nobody touches the eigenvectors" is false. BAHC improves
   persistence by filtering toward a structural ansatz *within one window*. Outside finance,
   Grassmannian prediction explicitly uses the **time sequence** of subspaces. What survives is
   the RMT-calibrated financial conjunction stated in the geometry section above.
2. **Take their protocol.** 10,000 simulations, each drawing $n=100$ random assets and a
   random window from 1992–2018 US equities, $t^{out} = 42$ days, scored on realised
   minimum-variance risk. That is how this literature manufactures evaluation mass from one
   dataset, it is published and reproducible, and it is the benchmark any stage 2 estimator
   has to beat anyway.

---

## The econometrics of time-varying loadings — the case they exclude

*2026-08-02. Both read in full by subagent, chapter by chapter.*

This is an entire literature the document did not touch, and it asks my question in different
language: a factor model's loading matrix $\Lambda_t$ spans a subspace, and whether it moves
is exactly whether the leading eigenvectors move.

**Bates, Plagborg-Møller, Stock & Watson, *Consistent Factor Estimation in Dynamic Factor
Models with Structural Instability*, J. Econometrics 177(2) 2013.** Writes instability as
$\Lambda_t - \Lambda_0 = h_{NT}\xi_t$ and asks when PCA still recovers the factor **space**.
Theorem 1 gives a rate $R_{NT}$ built from three envelopes $Q_1, Q_2, Q_3$; consistency needs
$Q_1 = O(N)$.

**Mikkelsen, *Time-Varying Loadings in Factor Models*, Aarhus 2016.** Three papers: a two-step
PCA→Kalman MLE for loadings following stationary VARs, an LM test for time-varying loadings,
and an FX application. Assumption D.1 is
$\sup_{s,t}\sum_{i,j=1}^{N}|E(\xi_{isp}\xi_{jtq}F_{sp}F_{tq})| = O(N)$ — **the same rate, by a
different route.**

**The whole literature rests on one condition, and it is the condition my project denies.**
Both consistency results require the loading drift to be cross-sectionally **weak**. Bates et
al. state the mechanism plainly:

> *"the reason that the principal component estimator can handle such large changes in the
> coefficients is that, **if these shifts have limited dependence across series**, their effect
> can be reduced, and eliminated asymptotically, by averaging across series."*

A coherent rotation shared across a non-vanishing fraction of names makes $\sum_{i,j}$ scale as
$O(N^2)$, not $O(N)$, and the theorems stop applying. Bates et al. draw the boundary to the
exponent: a perfectly correlated $O(1)$ shift is tolerated for at most $O(N^{1/2})$ series —
**about 22 names out of 500** — and beyond that their own §4.3 simulations show the trace $R^2$
frozen at 0.92 from $T=50$ to $T=400$, no convergence. Mikkelsen excludes the non-stationary
case outright: *"With non-stationary loadings the principal components estimator cannot
consistently estimate the factor space."*

So the correct framing is **not** that I contradict them. It is that my claim lives in the
regime they set aside, and their own boundary is the argument for why a rolling-window
eigenbasis is not good enough there.

**The threat, and it is now the sharpest one in this document.** Mikkelsen's Chapter 2 runs his
LM test on 100 Fama-French size/BM portfolios, $T = 636$ monthly, and rejects constant loadings
for **80–87%** of them, with the estimated factors nearly identical to Fama-French (squared
canonical correlations 0.993, 0.951, 0.917). Time-varying betas in equity data are established.
A referee will say the excess $D$ is just that. **The distinction I have to demonstrate is
coherence: a common rotation of the subspace, against the sum of $N$ independent $\beta$
wiggles — which is precisely what every model in that dissertation assumes.** This outranks the
ERSE-attribution threat and the BBP reading, because it does not attack the measurement, it attacks the
interpretation with an established alternative already fitted to equity data.

**What they hand me.**

- **Mikkelsen Ch. 2's LM test** — per series, regress squared PC residuals on the squared PCs,
  statistic $T R_i^2$, null $\chi^2(r)$, invariant to factor rotation, with a Breitung-Tenhofen
  GLS fix for serial correlation. A cheap independent screen on daily returns. Caveat: power is
  0.37 against small loading variance at $T=400$, so a non-rejection means little.
- **Mikkelsen Ch. 1's two-step PCA→Kalman** estimator fits an AR(1) to each loading path, and
  the AR matrix $B_i(L)$ is **rotation-invariant** even though the mean and variance are only
  identified up to $H$. That is the "predicted eigenbasis" building block, sitting unused —
  no chapter of the dissertation forecasts anything out of sample.
- **Procrustes alignment** $A^* = VU'$ from the SVD of $\mathrm{corr}(F,\tilde F)$, for fixing
  signs and rotations when comparing eigenvector sets across windows.
- **Bates et al.'s trace $R^2 = \hat{E}\|P_F\hat F\|^2/\hat{E}\|\hat F\|^2$** — a normalised
  sum of squared principal cosines, i.e. another cousin of $D$.

**A warning for regime 4.4.** Mikkelsen needs $T \geq 200$ before the AR persistence of a
loading path is estimated with under 10% bias, and his test's power collapses for small
loading variance. Direction persistence may be genuinely hard to detect at my window lengths
even if it is there — an inconclusive 4.4 will need to be reported as inconclusive rather than
as absence.

---

## Surrogate-data testing — what Regime 4.8A can and cannot establish

**Theiler, Eubank, Longtin, Galdrikian & Farmer, *Testing for nonlinearity in
time series: the method of surrogate data*, Physica D 58 (1992),
[DOI](https://doi.org/10.1016/0167-2789(92)90102-S)** introduced the operative
logic: specify a null, generate constrained surrogate histories under it, and
compare a discriminating statistic with that ensemble. **Schreiber & Schmitz,
*Improved surrogate data for nonlinearity tests*,
[arXiv:chao-dyn/9909041](https://arxiv.org/abs/chao-dyn/9909041)** proposed the
iterative amplitude-adjusted Fourier construction used in Regime 4.8A to match
the observed marginal distribution and autocorrelation more closely.

The transfer to this project is methodological, not a financial precedent.
Independent IAAFT surrogates preserve each asset's univariate marginal exactly
and its linear spectrum approximately, but deliberately destroy contemporaneous
market and sector organisation. They therefore test whether univariate dynamics
plus overlapping covariance estimation can manufacture Flag persistence. They
do **not** provide a matched null for a realistic financial covariance process,
do not preserve nonlinear volatility clustering, and cannot replace the intact
multivariate calendar-block null. A pass says real cross-sectional organisation
is necessary; it does not by itself identify the economic mechanism.

## Where that leaves the project

**Taken, and you should stop claiming it.** Detecting non-stationarity via two-window eigenvector
overlaps: BBP 2018 proposed the test, 2509.25076 extended it to overlapping windows and showed the
displacement on real US equities. Eigenvector overlap formulas, exact and $C$-free: solved.
The idea that eigenvector movement is the interesting axis: CFM, 2012–2025, continuously.

**Yours, on the evidence.**

1. **Heavy tails in the overlap null — answering a stated open problem.** 2509.25076's closing
   paragraph asks the question; regimes 1.5 and 3.1 answer it, including the mean-versus-threshold
   asymmetry that nobody appears to have noted. Strongest card.
2. **The block $D(P,Q)$ null needs more than the published pairwise formulas.** BBP gives exact
   non-perturbative mean-squared overlaps, but $D$ needs the joint singular-value law of the overlap
   block or a validated finite-$N$ closure. The precise missing object is stated above; this is not a
   claim that block overlap quantities in general are unknown.
3. **Quantification.** Additivity of $D_{emp} = D_{th} + D_{inject}$ to 0.1%, a detection floor of
   11° at $T{=}500$ Gaussian and 18° at $\nu{=}6$, and the $1+\mathrm{CV}^2$ law. The published work
   in this line reports curves and discrepancies, not thresholds and power.
4. **The RMT-calibrated financial forecast route is unoccupied on the evidence gathered** — the
   generic ML and geometric machinery is not. See the geometry section above.

**Threats, in order of severity.**

0. **Coherence.** Time-varying betas in equity data are established — Mikkelsen rejects constant
   loadings for 80–87% of 100 Fama-French portfolios. Unless the measured rotation is shown to
   be a *common* movement of the subspace rather than $N$ independent loading wiggles, the
   excess has a ready-made conventional explanation with a literature behind it. This now sits
   above everything below it, and the test is cheap: decompose the rotation into common and
   idiosyncratic parts. It is also the same quantity that decides whether Bates et al.'s
   consistency theorem applies to my panels, so one experiment answers both.

1. **ERSE attribution.** Rotation of near-degenerate modes narrows the pair's
   recomputed Rayleigh eigenvalues, while the full ERSE estimator also changes
   the basis. This is now tested directly rather than inferred: ERSE explains
   0.05–0.92% of outgoing tangent energy and residual persistence survives on all
   four panels. The redundancy threat is rejected; covariance-risk value remains
   open.
2. **BBP's reading of the same measurement.** They attribute bulk overlap departure to structure in
   the true spectrum, not to evolution. You need an argument that separates the two, and
   "I work on the top block, they excluded the top block" is a start but not a finish.
3. **Small panels are outside the theory's domain**, now with a named mechanism. DAX and CAC at
   $T = N$ are probably unusable; the $T = 250$ run you proposed is the right test.
4. **Survivorship**, unchanged and unfixed.

**What I would do next, in order.** The same-tangent-space persistence,
coherence, ERSE-attribution and partial-flag representation gates are now run.
The target is frozen as $\mathrm{Flag}(N;1,3,6)$, with separately reported
market/core/buffer losses. Start the strictly chronological baseline ladder:
holding still, full constant velocity, validation-fitted damping, and then a
transported tangent autoregression. Use the bounded projector loss for learning
and keep $D(P,Q)$ as the RMT-calibrated score. The sub-universe coherence sweep
is still required for a literal scaling claim but need not block model fitting.
BBP block-null closure and the
overlapping-window $t$ correction remain necessary calibration work, but neither
should postpone the direct held-out test of the learnability premise.

---

## Reading log — what I actually read

**Full text:** 2509.25076 (+ supplementary) · 1301.4939 · 2205.01012 · 1603.04364 (+ appendices) ·
2410.14420 · CFM RIE preprint (Bun, Allez, Bouchaud, Potters) — intro only.

**Full text, added 2026-08-02:** 2003.05807 (BAHC, + SI) · 2402.10352 (Grassmannian subspace
tracking) · 1203.6228 (the full paper itself, figures read off the PDF) · Bates,
Plagborg-Møller, Stock & Watson 2013 (by subagent) · Mikkelsen 2016 dissertation, all three
chapters (by subagent).

**Stage 2 geometry sweep, primary full text with operative sections read:** 1105.5782
(Grassmannian predictive coding) · 2303.14851 (dynamic subspace geodesics) · 2402.10352
(position/velocity RLS) · 2509.24767 (Grassmann/Stiefel AR processes) · 2509.23249 / ICLR 2026
(deep subspace regression) · 2412.09517 (financial realised-covariance SPD network) · 2011.13699
(Grassmann handbook, projector and logarithm sections) · 2511.06416 / L4DC 2026 (online flag
subspace learning). Yang–Hospedales CVPR 2016 and Szwagier–Pennec JMLR 2026 were read at the
primary proceedings source for their formulation, construction and scope; they were not treated as
cover-to-cover reads.

**Machinery now used for the Stage 1 gates:** the Grassmannian log maps and
same-basepoint construction of 2402.10352, whose Eq (6) —
$H_{t+1} = \log_{Y_{t+1}}(Y_{t+2})$ and
$\Gamma_{Y_t}^{H_t}H_t = -\log_{Y_{t+1}}(Y_t)$ — removes the need to implement transport at
all, since both vectors already sit in the tangent space at $Y_{t+1}$. Their chordal
first-order form, the second difference of projectors
$\|Y_{t+2}Y_{t+2}^\top - 2Y_{t+1}Y_{t+1}^\top + Y_tY_t^\top\|_F$, needs no SVD. Reference for
implementation is their [16], Bendokat, Zimmermann & Absil, *A Grassmann manifold handbook*
(2024). Exponential/retraction and explicit transport remain Stage 2 machinery;
Regime 4.7 adds the nested-projector flag embedding without claiming the
intrinsic flag exponential.

**Operative sections read:** 2507.01545 §1–§5, including the PER algorithm,
ERSE construction, Proposition 5's pairwise Rayleigh-eigenvalue identity and
the empirical setup.

**Abstract / summary only, flagged as such:** 2601.07687 · 2507.01918 · 2111.13109 ·
1610.08104 · 2404.18173 · 2501.08768 · 2503.15186 · 1006.4046 · SSRN 3421503.

**Not read:** 1203.6228 §5 (EWMA Langevin) and §6.2 (market-mode tilt). You have read this paper; I
have not, and every reference to it here is via other papers' citations of it. **The §4 equation-numbering
check BUILDNOTES flags is still outstanding.**

**Searches run for the novelty claim** (so you can judge how strong the negative is): ML/DL
eigenvector rotation prediction; beyond-RIE and non-rotationally-invariant estimators; deep learning
covariance cleaning and forecasting; Bongiorno/Challet/Potters non-stationary filtering; autoencoder
eigenvector denoising; geometric/SPD-manifold covariance forecasting; end-to-end NN portfolio
covariance; eigenvector forecasting for portfolios; Grassmann-valued time-series prediction;
Riemannian autoregression; neural subspace regression; and online flag-manifold learning. The
negative claim is now deliberately restricted to the RMT-calibrated financial conjunction above.
The literature most likely to hide a counterexample is proprietary and unpublished.
