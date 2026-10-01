# 🎲 Finding Structure with Randomness
### A seminar guide to Halko, Martinsson & Tropp (2011) — *Probabilistic Algorithms for Constructing Approximate Matrix Decompositions*

> **Who this is for:** a CS BSc student who knows a bit of linear algebra, calculus, and algorithms, and has to *present* this paper without drowning.
> **What you'll get:** the background you're missing, the paper's ideas in plain language, the algorithms, the experiments, the theory (with the proofs unpacked), and a toolkit for the actual talk (slide plan, Q&A cheat sheet).

---

## 📑 Table of Contents

0. [The 60-second version](#0-the-60-second-version)
1. [The problem: why anyone cares](#1-the-problem-why-anyone-cares)
2. [Background crash course](#2-background-crash-course)
3. [The big idea: two stages](#3-the-big-idea-two-stages)
4. [Stage A in detail: finding the range](#4-stage-a-in-detail-finding-the-range)
5. [Stage B in detail: from basis to factorization](#5-stage-b-in-detail-from-basis-to-factorization)
6. [Costs: where randomized wins (and where it doesn't)](#6-costs-where-randomized-wins-and-where-it-doesnt)
7. [The experiments](#7-the-experiments)
8. [The theory, unpacked](#8-the-theory-unpacked)
9. [Historical context: who thought of what](#9-historical-context-who-thought-of-what)
10. [A critical look](#10-a-critical-look)
11. [Seminar toolkit](#11-seminar-toolkit)
12. [Notation cheat sheet & glossary](#12-notation-cheat-sheet--glossary)

---

## 0. The 60-second version

Big data often lives in a big matrix $A \in \mathbb{R}^{m\times n}$. Very often that matrix is **secretly simple**: it has a few important directions and a lot of near-redundancy ("low numerical rank"). Classical algorithms (SVD, pivoted QR) find that structure but are expensive: $O(mnk)$ operations for rank $k$, and many passes over the data.

The paper's trick is almost insultingly simple:

1. **Poke the matrix with random vectors.** Compute $Y = A\Omega$, where $\Omega$ is a random $n \times (k+p)$ matrix.
2. **Orthonormalize.** The columns of $Y$ are random samples of the range of $A$. A QR factorization gives an orthonormal basis $Q$ for them.
3. **Compress.** $A \approx Q(Q^*A)$. Now the big matrix has been reduced to the small matrix $B = Q^*A$, and you run an ordinary deterministic SVD on $B$.

The paper shows that this works, and that it is:

- **accurate**: error within a small factor of the theoretical optimum $\sigma_{k+1}$, with failure probability you can make like $10^{-15}$;
- **fast**: $O(mn\log k)$ for dense matrices (using a structured random matrix), instead of $O(mnk)$;
- **robust and parallel-friendly**: it is all matrix–matrix multiplications;
- **pass-efficient**: a constant number of passes over data instead of $O(k)$. Sometimes one pass suffices.

> **One-sentence thesis for your talk:** *Randomness isn't a desperate last resort for matrix computations. It's a precise tool that, combined with classical linear algebra, gives provably accurate, fast and robust low-rank approximations.*

---

## 1. The problem: why anyone cares

### 1.1 Matrices that are "secretly small"

A $1000\times 1000$ matrix has a million numbers. But if every column is (nearly) a combination of just 10 fixed columns, there are really only about $10\cdot 1000 + 10\cdot 1000 = 20{,}000$ numbers worth of information. The rest is redundancy.

The factorization the whole paper revolves around is

$$
\underbrace{A}_{m\times n} \;\approx\; \underbrace{B}_{m\times k}\;\underbrace{C}_{k\times n}, \qquad k \ll \min(m,n). \tag{1.1}
$$

(The paper's equation 1.1.) If you have it, you can:

- **store** $A$ with $k(m+n)$ numbers instead of $mn$;
- **multiply** $A$ by a vector in $O(k(m+n))$ time instead of $O(mn)$ (compute $C x$ first, then $B(Cx)$);
- **interpret** the data (principal components, topics, "eigenfaces").

### 1.2 Where these matrices show up (from the paper, §1.1)

| Application | What the low-rank structure means |
|---|---|
| **PCA** (statistics/data mining) | Directions of maximal variance = a low-rank approximation of the (centered) data matrix |
| **Dimensionality reduction / manifold learning** | Data has fewer degrees of freedom than its ambient dimension |
| **Least squares** | Huge, nearly dependent systems; good factorizations give good solvers |
| **PDE solvers / fast multipole / $\mathcal H$-matrices** | Interactions between far-apart objects are smooth ⇒ low rank |
| **Multiscale modeling** | The input-to-output map is approximately low rank |

### 1.3 Why classical algorithms are no longer enough (paper §1)

The authors list four reasons. They are really a lecture on *how computing changed*:

1. **Matrices are stupendously big** (web graphs, image databases). Classical algorithms weren't designed for that.
2. **Data is noisy anyway.** Computing a decomposition to 16 digits when your data has 3 digits of accuracy is "profligate".
3. **Data movement is the real cost.** Moving a matrix from disk to RAM can cost far more than the arithmetic on it. So *passes over the data* matter more than FLOPs.
4. **Hardware evolves** (multicore, GPUs). Algorithms must parallelize well.

> 💡 **Key mindset shift to convey:** in the classical cost model you count floating-point operations. In the paper's world you also count *how many times you have to read the matrix* and *how parallelizable the work is*.

---

## 2. Background crash course

You don't need more than this section to follow the rest. It's the part to read slowly. Everything is stated for real matrices, where the paper's $A^*$ (conjugate transpose) is just the transpose $A^\top$. I'll keep the paper's $A^*$ so you can compare with it.

### 2.1 Matrices as machines

A matrix $A\in\mathbb{R}^{m\times n}$ is a machine that eats vectors $x\in\mathbb{R}^n$ and spits out $Ax \in \mathbb{R}^m$.

- **Range** (column space): all possible outputs, $\mathrm{range}(A)=\{Ax\}$. It's the span of the columns.
- **Null space:** inputs that get killed, $Ax = 0$.
- **Rank:** the dimension of the range = the number of linearly independent columns.

**Key idea for this paper:** if $\mathrm{rank}(A)=k$ then *every* output $Ax$ lives in a $k$-dimensional subspace. Find that subspace, and you've captured what $A$ "does".

### 2.2 Orthonormal bases, QR, and projections

- Vectors $q_1,\dots,q_k$ are **orthonormal** if $q_i^\top q_j = 0$ for $i\ne j$ and $\|q_i\| = 1$.
- A matrix $Q$ with orthonormal columns satisfies $Q^*Q = I$ and preserves lengths: $\|Qx\| = \|x\|$.
- **Gram–Schmidt / QR factorization:** any matrix $Y$ with independent columns can be written $Y = QR$, where $Q$ has orthonormal columns spanning $\mathrm{range}(Y)$ and $R$ is upper triangular. This is how we "orthonormalize" the columns.
- **Orthogonal projector:** $P_Y = QQ^*$ takes any vector and drops it onto the closest point in $\mathrm{range}(Y)$ (its "shadow"). $I - P_Y$ gives what's left over (the part perpendicular to the range).

The paper measures quality by how much of $A$ is *left over* after projecting its columns onto $\mathrm{range}(Q)$:

$$
\|A - QQ^*A\| = \|(I-P_Y)A\|.
$$

Small leftover = $Q$ captures the "action" of $A$.

### 2.3 Norms: how to measure "how big is this matrix?"

- **Spectral (operator) norm:** the maximum stretch factor,
$$\|A\| = \max_{x\neq 0}\frac{\|Ax\|}{\|x\|}.$$
- **Frobenius norm:** the "vector norm of all entries",
$$\|A\|_F = \Big(\sum_{j,k}|a_{jk}|^2\Big)^{1/2}.$$

Useful facts used throughout: $\|A\|^2 = \|A^*A\| = \|AA^*\|$, and both norms are **unitarily invariant**: rotating inputs/outputs (multiplying by orthonormal matrices) doesn't change them. This lets the proofs "rotate" into convenient coordinates.

### 2.4 The Singular Value Decomposition (SVD), the star of the show

Every matrix has an SVD:

$$
A = U\Sigma V^*, \qquad \Sigma=\mathrm{diag}(\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r > 0),
$$

with $U$ and $V$ having orthonormal columns (**left/right singular vectors**).

**Geometric picture** (memorize this, it's your best slide): take the unit sphere in $\mathbb{R}^n$. Apply $A$. You get an **ellipsoid** whose semi-axis lengths are the singular values $\sigma_j$ and whose axis directions are the columns of $U$. A matrix of low numerical rank is a **pancake**: a few long axes, the rest nearly flat.

```
   unit sphere  --A-->  ellipsoid
      ○                  ⬭  (σ1 long, σ2 shorter, σ3 tiny ...)
```

**The punchline theorem (Eckart–Young–Mirsky; paper eq. 1.4 / 3.1):** the best rank-$k$ approximation of $A$ is the **truncated SVD** $A_{(k)}=\sum_{j\le k}\sigma_j u_jv_j^*$, and

$$
\min_{\mathrm{rank}(X)\le k}\|A-X\| = \sigma_{k+1}, \qquad
\min_{\mathrm{rank}(X)\le k}\|A-X\|_F = \Big(\sum_{j>k}\sigma_j^2\Big)^{1/2}.
$$

So **$\sigma_{k+1}$ is the yardstick** for the whole paper: no rank-$k$ method can beat it. Every error bound in the paper reads "our error $\le$ (small factor) $\times\ \sigma_{k+1}$".

**Numerical rank.** Real data is never exactly low rank. We say the numerical rank is $k$ if $\sigma_{k+1}$ is below our tolerance. (Singular values that "decay quickly" ⇒ easy problem; "decay slowly / flat spectrum" ⇒ hard problem. This matters a lot later.)

### 2.5 Classical algorithms and their price tags

| Method | What it does | Cost |
|---|---|---|
| Full SVD / full QR | Everything | $O(mn\min\{m,n\})$ |
| Rank-revealing QR (Businger–Golub, Gu–Eisenstat) | Pivoted QR, picks the "important" columns first | $O(kmn)$ |
| Krylov methods (Lanczos/Arnoldi) | Uses only matrix–vector products $x\mapsto Ax$ | $\sim k\,T_{\text{mult}}+k^2(m+n)$ |

Here $T_{\text{mult}}$ is the cost of one matrix–vector product (about $mn$ for a dense matrix, maybe $m+n$ for a sparse or structured one).

Two things to note about these:

- They need lots of **random access** to the matrix, and at least $k$ **passes** over it, a disaster when the matrix lives on disk.
- Krylov methods build the subspace $\mathrm{span}\{\omega, A\omega, A^2\omega,\dots\}$. They're excellent but **numerically unstable** unless you add restarts, reorthogonalization and other sophisticated surgery (the paper quotes *Numerical Recipes* basically saying "don't write this yourself").

### 2.6 Cost models: FLOPs vs. passes

- **FLOP count**: arithmetic operations. This is the usual "$O(\cdot)$" you learned.
- **Pass-efficiency**: how many times you must stream the whole matrix through fast memory. When the data doesn't fit in RAM, this dominates everything and FLOP counts become "largely irrelevant".
- **BLAS-3 / blocking:** multiplying a matrix by $k$ vectors *at once* is much faster than doing $k$ separate matrix–vector products (better cache use, easy parallelism), even though the FLOP count is the same.

This explains why randomized algorithms can win *even with the same or more FLOPs*.

### 2.7 Just enough randomness

- A **standard Gaussian** random variable has mean 0, variance 1. A **Gaussian matrix** has independent standard Gaussian entries.
- **Rotation invariance (crucial!):** if $G$ is Gaussian and $U, V$ are orthonormal, then $U^*GV$ is *also* a standard Gaussian matrix. A Gaussian is "equally random in every direction". It has no preferred coordinate system. The paper's proofs exploit this constantly.
- **Expectation** $\mathbb{E}$ = average over the randomness.
- **Concentration of measure:** nice functions of many independent Gaussians rarely stray far from their average (the probability of deviating by $t$ standard deviations decays like $e^{-t^2/2}$). That's why a statement about the *average* error is also a statement about the *typical* error.
- **Monte Carlo reputation (Remark 1.1).** Classical Monte Carlo integration, $\int f\,d\mu\approx\frac1m\sum f(X_i)$, has error $\sim m^{-1/2}$ and no certificate of success. That gives randomized algorithms a bad name in scientific computing. The authors want you to know that *their* algorithms are different: they're insensitive to the quality of the random number generator, and failure probability is a user-controlled parameter you can push below $10^{-15}$.

### 2.8 The Johnson–Lindenstrauss idea (the spiritual ancestor)

**JL lemma (1984):** $N$ points in a high-dimensional Euclidean space can be mapped by a *random* linear map into dimension $O(\log N)$ while approximately preserving all pairwise distances. Random maps preserve geometry.

The paper's algorithm is the same spirit: a random map $\Omega$ squeezes the $n$-dimensional input space down to $k+p$ dimensions, and (with high probability) loses nothing important about the range of a low-rank $A$.

---

## 3. The big idea: two stages

### 3.1 Splitting the problem

**Stage A: find a low-dimensional subspace that captures the action of $A$.**
Find $Q\in\mathbb{R}^{m\times \ell}$ with orthonormal columns such that

$$
A\approx QQ^*A. \tag{1.2}
$$

*(This is the only randomized part.)*

**Stage B: use $Q$ to compute any standard factorization.**
Since $A\approx Q(Q^*A)$, we have reduced $A$ to the small $\ell\times n$ matrix $B=Q^*A$. Compute a deterministic factorization of $B$ and lift it back.

For the SVD:

1. Form $B=Q^*A$.
2. Compute the SVD of the small matrix: $B=\widetilde U\Sigma V^*$.
3. Set $U=Q\widetilde U$.

Then $A\approx QB = Q\widetilde U\Sigma V^*=U\Sigma V^*$. 🎉

> 🧠 **Why this decoupling is elegant:** the "scary" randomness is confined to finding a subspace. Everything afterwards is textbook, well-understood, numerically stable linear algebra on a *small* matrix. Randomness helps to *shrink the problem*, not to solve it.

### 3.2 Two problem formulations

- **Fixed-precision problem:** given tolerance $\varepsilon$, find $Q$ with the fewest columns such that $\|A-QQ^*A\|\le\varepsilon$. Optimal answer: the number of singular values exceeding $\varepsilon$.
- **Fixed-rank problem:** given target rank $k$ and oversampling $p$, find $Q$ with $k+p$ columns such that $\|A-QQ^*A\|\approx\sigma_{k+1}$ (eq. 1.5).

The paper mostly analyzes fixed-rank (cleaner) and shows how to convert to fixed-precision by building $Q$ incrementally while estimating the error cheaply (§4.3–4.4).

### 3.3 Intuition for the random sampling (§1.3.2)

**Case 1: $A$ has exact rank $k$.**
Draw a random vector $\omega$ and compute $y=A\omega$. This is a random sample from the range of $A$. Repeat $k$ times: $y^{(i)}=A\omega^{(i)}$. Because the $\omega^{(i)}$ are random they're in "general position": linearly independent and none (almost surely) lands in the null space. So the $y^{(i)}$ are independent, i.e. they span the whole $k$-dim range. Orthonormalize them, done.

**A toy example.** Let
$$A=\begin{bmatrix}1&2\\2&4\\3&6\end{bmatrix}$$
(rank 1: the second column is twice the first). The range is the line spanned by $(1,2,3)^\top$, and the null space is the line spanned by $(2,-1)^\top$. Take *any* $\omega$ not on that null line, say $\omega=(1,-1)^\top$. Then $A\omega=(-1,-2,-3)^\top$, which is a multiple of $(1,2,3)^\top$. One random poke found the whole range. The only unlucky $\omega$'s sit on a line, a set of probability zero.

**Case 2: $A=B+E$** (rank-$k$ "signal" $B$ plus small "noise" $E$).
Now $A\omega = B\omega+E\omega$, and the noise term nudges every sample out of $\mathrm{range}(B)$. With exactly $k$ samples, unlucky noise could make us miss part of the signal. So take $k+p$ samples: **oversampling**. The paper's remarkable claim is that the failure probability decays *superexponentially* in $p$, so $p=5$ or $10$ typically gives superb results.

> 🎣 **Analogy:** you want to learn the shape of a dark room. Throwing exactly as many balls as there are "important" walls works only if each bounce is clean. Throw a few extras and you're safe even if a couple bounce weirdly.

### 3.4 The Proto-Algorithm

> **Proto-Algorithm (fixed-rank).** Given $A\in\mathbb{R}^{m\times n}$, target rank $k$, oversampling $p$:
> 1. Draw a random $n\times(k+p)$ test matrix $\Omega$.
> 2. Form $Y=A\Omega$.
> 3. Construct $Q$ whose columns are an orthonormal basis of $\mathrm{range}(Y)$.

Three lines. The rest of the 74 pages is about answering:

- Which $\Omega$? How much oversampling? *(§4.1–4.2, 4.6, 10, 11)*
- $Y$ is ill-conditioned, so how do we orthonormalize stably? *(§4.1)*
- What if $k$ is unknown? *(§4.3–4.4)*
- What if the spectrum decays slowly? *(§4.5, power iteration)*
- How do we get SVD/ID/eigendecomposition from $Q$? *(§5)*
- What are the costs? *(§6)*  Does it work? *(§7)*  What can we *prove*? *(§8–11)*

The authors note this is basically the first step of **subspace iteration** with a random start, an old idea. Their novelty is the observation that the starting subspace should be a bit *larger* ($k+p$ dimensions) than the subspace you want, and that this single step often suffices.

---

## 4. Stage A in detail: finding the range

### 4.1 Algorithm 4.1: Randomized Range Finder

> 1. Draw an $n\times\ell$ Gaussian matrix $\Omega$.
> 2. $Y=A\Omega$.
> 3. $Q$ = orthonormal basis of $\mathrm{range}(Y)$ (e.g. QR: $Y=QR$).

**Cost:**
$$T_{\text{basic}}\sim \ell n\,T_{\text{rand}}+\ell\,T_{\text{mult}}+\ell^2 m$$
(generating the random numbers, $\ell$ matrix–vector products, and the QR). The middle term usually dominates. And it is **embarrassingly parallel**: the $\ell$ products $A\omega^{(i)}$ are independent.

**Practical notes from the paper:**

- Performance depends *very little* on the quality of the random number generator.
- **Orthonormalization must be stable.** The columns of $Y$ are almost linearly dependent (because $A$ amplifies the dominant directions and crushes the rest), so naive Gram–Schmidt loses orthogonality. The authors recommend Gram–Schmidt with *double orthogonalization*, or Householder reflections / Givens rotations.
- **Pivoting buys nothing:** the columns of $Y$ are i.i.d. samples, so no column is "more important" than another.

### 4.2 How much oversampling?

Define $p=\ell-k$. Rules of thumb:

- Gaussian $\Omega$: **$p=5$ or $10$** is typically enough. Rarely any gain from $p>k$.
- Faster-decaying singular values ⇒ less oversampling. Exact rank $k$ ⇒ $p=0$ works.
- Very large matrices may need more.
- Structured $\Omega$ (§4.6) needs more, but is cheaper per sample.
- Since the bottleneck is $A\Omega$, which blocking/parallelism accelerates, it is often fine to draw *more* samples than strictly necessary. The QR step notices the numerical rank and ignores the excess.

### 4.3 A posteriori error estimation: *how do I know I'm done?*

We'd like to know $\|(I-QQ^*)A\|$ without computing it (that's as expensive as the original problem). **Trick:** the norm of a matrix can be estimated by probing it with random vectors.

**Lemma 4.1.** For a real matrix $B$, draw $r$ independent standard Gaussian vectors $\omega^{(i)}$. Then for any $\alpha>1$,
$$
\|B\|\;\le\;\alpha\sqrt{\tfrac{2}{\pi}}\;\max_{i=1,\dots,r}\|B\omega^{(i)}\|
$$
except with probability at most $\alpha^{-r}$.

**Why plausible?** Let $v$ be the top right singular vector of $B$. Then $\|B\omega\|\ge\sigma_1|v^*\omega|$, and $v^*\omega$ is a one-dimensional standard Gaussian whose expected absolute value is $\sqrt{2/\pi}\approx0.8$. So a typical probe returns about $0.8\,\|B\|$ or more. Taking the max over several probes and multiplying by a safety factor $\alpha$ gives an upper bound with overwhelming probability. (This is a heuristic explanation, not the paper's proof, which is in Woolfe et al.)

Apply it with $B=(I-QQ^*)A$, $\alpha=10$ and $r=10$:

$$
\|(I-QQ^*)A\|\le 10\sqrt{\tfrac2\pi}\max_{i}\|(I-QQ^*)A\omega^{(i)}\| \quad\text{with prob.}\ge 1-10^{-r}. \tag{4.3}
$$

It costs only $r$ extra matrix–vector products.

### 4.4 Algorithm 4.2: Adaptive Randomized Range Finder ("error estimation almost for free")

Build $Q$ **one column at a time**:

```
for i = 1, 2, ...
    draw Gaussian ω(i);   y(i) = A ω(i)
    q̃(i) = (I − Q(i−1) Q(i−1)*) y(i)      # remove what's already captured
    q(i) = q̃(i) / ‖q̃(i)‖;   Q(i) = [Q(i−1)  q(i)]
```

The vectors $\tilde q^{(i)}$ are exactly the quantities appearing in the estimator! So **stop when $r$ consecutive $\|\tilde q^{(i)}\|$ are all smaller than $\varepsilon/(10\sqrt{2/\pi})$.** The estimate costs nothing extra. The algorithm guarantees $\|(I-QQ^*)A\|\le\varepsilon$ with probability at least $1-\min\{m,n\}\,10^{-r}$. (The failure bound comes from a crude union bound and is pessimistic; $r=10$ works well in practice.)

In practice, process samples in **blocks** (e.g. 32, then 64, ...) for BLAS-3 speed. Doubling the guess keeps the asymptotic cost.

### 4.5 Slowly decaying spectra: the power scheme

**The problem.** If the singular values of $A$ decay slowly (typical for images, networks, text), the tail $\Sigma_2$ is large, so the noise terms $E\omega$ contaminate the samples. The basic algorithm can then be mediocre.

**The fix (Rokhlin–Szlam–Tygert).** Apply the same algorithm to
$$
B=(AA^*)^qA, \qquad\text{whose SVD is } B=U\Sigma^{2q+1}V^*.
$$
Same singular *vectors*, but the singular values become $\sigma_j(B)=\sigma_j(A)^{2q+1}$. This *widens the gaps* between big and small values.

> 🔊 **Analogy:** a spectrum with $\sigma_1=1$, $\sigma_{11}=0.5$ looks "flat-ish". Raise to the 5th power ($q=2$): $1$ vs $0.03$. You've turned up the contrast knob, so the dominant directions drown out the rest.

**Algorithm 4.3 (naive):** $Y=(AA^*)^qA\Omega$, then QR.

**Algorithm 4.4 (what you should actually use): Randomized Subspace Iteration.** Re-orthonormalize between every multiplication:

```
Y0 = AΩ,            Y0 = Q0 R0
for j = 1..q:
    Ỹj = A* Q(j-1),   Ỹj = Q̃j R̃j
    Yj = A Q̃j,        Yj = Qj Rj
Q = Qq
```

**Why re-orthonormalize?** In floating point, with machine precision $\mu\approx10^{-16}$, naive powering wipes out all information about singular values smaller than $\mu^{1/(2q+1)}\|A\|$. For $q=2$ that's $10^{-16/5}\approx 6\times 10^{-4}$, so you'd be blind to everything below about $0.06\%$ of $\|A\|$. Interleaving QR factorizations fixes this at nearly no cost. The two algorithms are algebraically equivalent in exact arithmetic.

**Price and payoff:**

- Cost: $2q+1$ times as many matrix–vector products as Algorithm 4.1.
- Passes: $2(q+1)$ over the matrix.
- Heuristic from the paper: *if the basic scheme is within factor $C$ of optimal, the power scheme is within $C^{1/(2q+1)}$.* The gap shrinks to 1 **exponentially fast in $q$**. Example: $C=100$ gives $4.6,\,2.5,\,1.9$ for $q=1,2,3$.
- In practice $q=1$ or $2$ usually suffices, and the paper says it is rare for more than five passes to be needed.

### 4.6 The accelerated technique: structured random matrices (SRFT)

**Bottleneck:** with a Gaussian $\Omega$, computing $A\Omega$ costs $O(mn\ell)$, the same as classical rank-revealing QR. To beat that, we need a random $\Omega$ whose product with $A$ is *fast*.

**Subsampled Random Fourier Transform (SRFT):**
$$
\Omega=\sqrt{\tfrac n\ell}\;D\,F\,R,
$$
where

- $D$: $n\times n$ diagonal with i.i.d. random unit-modulus entries (random "phases", i.e. random signs in the real case);
- $F$: the $n\times n$ unitary **discrete Fourier transform**, $f_{pq}=n^{-1/2}e^{-2\pi i(p-1)(q-1)/n}$;
- $R$: an $n\times\ell$ matrix that picks $\ell$ random coordinates.

Computing $A\Omega$ applies a (subsampled) **FFT** to each row of $A$, costing $O(mn\log\ell)$ instead of $O(mn\ell)$.

> 🧩 **Why this works intuitively:** think of sampling $\ell$ random coordinates of a vector. That's bad if the vector's energy is concentrated in a few coordinates ("spiky") because you'd likely miss it. So first **scramble** with $D$ (random phases), then **spread** the energy with the FFT ($F$), so no coordinate is special, then **sample** with $R$. It's a cheap imitation of a dense random rotation.

**Algorithm 4.5:** the same as 4.1 but with an SRFT $\Omega$. Total cost
$$T_{\text{struct}}\sim mn\log\ell+\ell^2n.$$

**Caveats the authors state openly:**

- Needs more oversampling in theory ($\ell\sim(k+\log n)\log k$), though empirically $\ell=k+10$ or $k+20$ is almost always enough.
- Doesn't adapt easily to fixed-precision (samples are computed in bulk). So just double $\ell$ until it works.
- It's *not known* whether the power-iteration refinement can be accelerated this way (Remark 4.7).
- Even stronger practical variants exist: chains of random Givens rotations (eq. 4.8, Remark 4.6).

---

## 5. Stage B in detail: from basis to factorization

Given $Q$ with $\|A-QQ^*A\|\le\varepsilon$. We want an SVD, eigenvalue decomposition, ID, etc.

### 5.1 Direct SVD (Algorithm 5.1), the accuracy champion

$B=Q^*A$ → SVD of $B$ → $U=Q\widetilde U$.

**Error doesn't degrade:** $\|A-U\Sigma V^*\|=\|A-QQ^*A\|\le\varepsilon$, since $U\Sigma V^*=QB=QQ^*A$ exactly. Cost is dominated by $Q^*A$: $O(kmn)$ for dense $A$, but it's a single big matrix–matrix product (BLAS-3, parallel) and it needs only a fast way to apply $A^*$. Remaining steps cost $O(k^2n)$ and $O(k^2m)$.

**Truncation (Remark 5.1):** if $Q$ has more columns than the desired rank $k$, zero out all but the top-$k$ entries of $\Sigma$. Worst case, this adds $\sigma_{k+1}$ to the error (Theorem 9.3), but empirically the penalty is usually absent.

### 5.2 Interpolative decomposition (ID) and row extraction (Algorithm 5.2): fast but lossy

The **ID** picks $k$ actual columns of $A$ such that
$$A=A_{(:,J)}X,$$
where $X$ contains a $k\times k$ identity (in the columns $J$) and all entries satisfy $|x_{ij}|\le2$. "Every column is a combination of $k$ chosen columns, with bounded coefficients." Finding the *best* column subset is NP-hard (maximum volume), but allowing the bound 2 makes it efficient. This is a neat "relax a hard problem slightly" moment worth mentioning.

**Row extraction:** compute an ID of $Q$ itself, $Q=XQ_{(J,:)}$. Then $A\approx XA_{(J,:)}$, so a factorization is obtained **without any matrix–matrix product** (cost $O(k^2(m+n))$).

**Lemma 5.1:** the price is
$$\|A-U\Sigma V^*\|\le\Big[1+\sqrt{1+4k(n-k)}\Big]\varepsilon.$$
The proof is a neat three-line chain: write $\hat A=QQ^*A$; then $\hat A=X\hat A_{(J,:)}$, and
$$\|A-XA_{(J,:)}\|\le\|A-\hat A\|+\|X\|\,\|A_{(J,:)}-\hat A_{(J,:)}\|\le(1+\|X\|)\varepsilon,$$
because a submatrix of $A-\hat A$ can't have a larger norm than $A-\hat A$. Then bound $\|X\|\le\sqrt{1+4k(n-k)}$ using $|x_{ij}|\le 2$.
Use it when speed matters and $\varepsilon$ is tiny; otherwise prefer Algorithm 5.1.

### 5.3 Hermitian (symmetric) matrices (Algorithm 5.3)

If $A=A^*$, $Q$ is a good basis for both row and column spaces: $A\approx Q(Q^*AQ)Q^*$ with error $\le2\varepsilon$. Form the small $B=Q^*AQ$, diagonalize it ($B=V\Lambda V^*$), set $U=QV$. (Algorithm 5.4 is the row-extraction analogue.)

### 5.4 Positive semidefinite matrices: Nyström (Algorithm 5.5)

For PSD $A$, use
$$A\approx(AQ)(Q^*AQ)^{-1}(AQ)^*=FF^*,\qquad F=(AQ)(Q^*AQ)^{-1/2}.$$
Implementation: $B_1=AQ$, $B_2=Q^*B_1$, Cholesky $B_2=C^*C$, $F=B_1C^{-1}$, SVD of $F$. Its spectral error never exceeds $\|A-QQ^*A\|$ and is often much smaller. You get "one free step of subspace iteration".

### 5.5 Single-pass algorithms: when you can only look once

If the matrix is a *stream* you cannot revisit. For Hermitian $A$ we already know $\Omega$, $Y=A\Omega$ and $Q$. Let $B=Q^*AQ$ (unknown). Since $AQQ^*\approx A$,
$$
BQ^*\Omega=Q^*AQQ^*\Omega\approx Q^*A\Omega=Q^*Y.
$$
So solve the least-squares problem $B\,(Q^*\Omega)\approx Q^*Y$ for $B$. Everything on both sides is available *without touching $A$ again* (Algorithm 5.6).

For non-Hermitian $A$: sketch both $Y=A\Omega$ and $\tilde Y=A^*\tilde\Omega$ in one pass, then solve two coupled relations for $B=Q^*A\tilde Q$.

**Caveat (Remark 5.4):** single-pass is less accurate. The matrix $Q^*\Omega$ is typically ill-conditioned, so the error can be inflated by $1/\tau_{\min}$ (its smallest singular value). Oversampling makes the system overdetermined and helps.

---

## 6. Costs: where randomized wins (and where it doesn't)

The paper evaluates three "computing environments" (§1.4, §6). This is a great slide.

| Environment | Classical | Randomized | Verdict |
|---|---|---|---|
| **Dense matrix in RAM** | RRQR: $O(kmn)$ | SRFT + row extraction: $O\big(mn\log k+k^2(m+n)\big)$ | Several times faster, even for $m,n\sim10^3$, $k\sim10^2$ |
| **Fast matvec (sparse/structured)** | Krylov: $\sim k\,T_{\text{mult}}+k^2(m+n)$ | $2(k+p)T_{\text{mult}}+O(k^2(m+n))$ with Gaussian $\Omega$ (more with power iteration: $(2q+2)(k+p)T_{\text{mult}}$) | Same ballpark in flops. Krylov is *sometimes more accurate*; randomized is *more robust, simpler, parallel* |
| **Out-of-core / streaming** | $\ge k$ passes | $2(q+1)$ passes; $q=0$ with direct SVD is 2 passes; single-pass variants exist | Huge win |

Notable quotes/ideas for the talk:

- In the sparse case, it is usually *much* faster to apply a matrix to $k$ vectors **at once** than to do $k$ sequential matvecs. So "a few steps of the power method can be done faster than $k$ steps of a Krylov method".
- The error rules of thumb: dense SRFT route $\|A-U\Sigma V^*\|\lesssim n\,\sigma_{k+1}$ (worst case); Gaussian route $\lesssim\sqrt{kn}\,\sigma_{k+1}$; with power iteration $(kn)^{1/(2(2q+1))}\sigma_{k+1}$. Actual errors are usually much smaller.
- **Honest message:** randomized does *not* always beat Krylov in accuracy per flop. Its selling points are robustness, theory, flexibility and hardware friendliness. The paper even suggests hybrid schemes could be best. In fact Algorithm 4.3 is conceptually a *block Lanczos method with a random starting matrix*.

---

## 7. The experiments

Four sets of experiments (§7). Here is the story each one tells.

### 7.1 Rapidly decaying spectra (Algorithm 4.2)

- **Matrix A:** a $200\times200$ discretization of a Laplace single-layer integral operator, with singular values decaying exponentially down to machine precision.
- **Matrix B:** a $1596\times532$ *implicit* matrix (each matvec solves a sparse $140{,}000\times140{,}000$ linear system on a resistor network).
- **Findings:**
  - The actual error $e_\ell$ is "remarkably close" to the theoretical minimum $\sigma_{\ell+1}$.
  - The error estimator $f_\ell$ is always an upper bound, pessimistic by about $10\times$ (the built-in safety factor).
  - Over **2000 trials** the errors concentrate tightly around their mean (the algorithm isn't just good on average; it's *reliably* good).
  - The basis reaches full double-precision accuracy.

### 7.2 Graph Laplacian of an image (slowly decaying spectrum)

- A $9025\times9025$ sparse matrix built from similarity of $5\times5$ patches in a $95\times95$ image. Eigenvectors give "coordinates" for denoising.
- **Finding:** $q=0$ (basic algorithm) is "miserable". $q=1$ is already a *tremendous* improvement and the top eigenvalues become accurate; $q=2,3$ give little more.

### 7.3 Eigenfaces

- FERET face database: $98{,}304\times7254$ dense matrix (about $5.4$ GB in double precision, which doesn't fit in the RAM of many machines). Implemented in a pass-efficient way. Errors estimated with a power-iteration-based estimator (Remark 4.1) since the true error is too expensive to compute.
- **Finding:** same pattern. $q=0$ is poor, big gains by $q=1$, essentially nothing after 3–5 passes.

### 7.4 Structured random matrices: speed and accuracy

*Methods compared for an $n\times n$ Gaussian test matrix:*

| name | Stage A | Stage B |
|---|---|---|
| `direct` | Rank-revealing QR | Alg. 5.1 |
| `gauss` | Alg. 4.1 with Gaussian | Alg. 5.1 |
| `srft` | Alg. 4.1 with modified SRFT (4.8) | Alg. 5.2 |
| `svd` | Full SVD (LAPACK), truncated | — |

**Speed (Table 7.1):** for $n=4096$, $\ell=160$: `direct` 24.6 s, `gauss` 11.6 s, `srft` 3.94 s, about a $6\times$ speedup for SRFT. A full SVD for $n=4096$ takes 690 s. For large $\ell$ (e.g. 640) SRFT's advantage disappears and it becomes slower than `direct`, a nice reminder that asymptotics have constants. The Gaussian method gives a steady $\sim2\times$ gain, "primarily because matrix–matrix multiplication is faster than QR".

**Accuracy (Fig 7.8):** 100,000 trials per distribution (Gaussian, random orthonormal, SRFT, modified SRFT). The SRFT is *as good as or slightly better than* Gaussian, and the error distributions are tight. (Orthonormal beats Gaussian only when $\ell$ is nearly $n$, because a nearly square Gaussian matrix is badly conditioned.)

> **Experimental setup honesty:** timings come from a 3.2 GHz Pentium IV, Fortran 90. Absolute numbers are dated. The *ratios* and the qualitative story are what matter.

---

## 8. The theory, unpacked

This is where seminars get scary, so I'll take it slowly. The proofs are less scary than they look. The structure is:

> **Part 1 (deterministic):** a pure linear-algebra bound that holds for *any* test matrix $\Omega$.
> **Part 2 (probabilistic):** plug in the specific random distribution and bound the random terms.

(The authors say this "bipartite" proof style is common in the literature on randomized linear algebra and is closest in spirit to Boutsidis–Drineas–Mahoney.)

### 8.1 Setup (§9.1)

Write the SVD of $A$ with a split at index $k$:

$$
A=U\begin{bmatrix}\Sigma_1&\\&\Sigma_2\end{bmatrix}\begin{bmatrix}V_1^*\\V_2^*\end{bmatrix},
$$

where $\Sigma_1$ holds the top $k$ singular values ("signal") and $\Sigma_2$ the rest ("tail", with $\|\Sigma_2\|=\sigma_{k+1}$). Rotate the test matrix into the singular-vector coordinates:

$$
\Omega_1=V_1^*\Omega\ \ (k\times\ell),\qquad \Omega_2=V_2^*\Omega\ \ ((n-k)\times\ell).
$$

Then the sample matrix is

$$
Y=A\Omega=U\begin{bmatrix}\Sigma_1\Omega_1\\\Sigma_2\Omega_2\end{bmatrix}.
$$

**Reading it:** top block = what we *want* to capture (signal, scaled by big $\sigma$'s). Bottom block = contamination from the tail. We succeed if the signal part isn't drowned: $\Omega_1$ must be "well-behaved" (full row rank, not nearly singular) and $\Omega_2$ not too big.

### 8.2 Tools: the linear-algebra toolbox (§8)

You don't need to reprove these, but know what's in the box:

| Tool | Statement | Role |
|---|---|---|
| **Conjugation rule** (8.1) | $M\succeq0\Rightarrow A^*MA\succeq0$; $M\preceq N\Rightarrow A^*MA\preceq A^*NA$ | Lets us push inequalities between PSD matrices through multiplications |
| **Perturbation of inverses** (8.2) | $I-(I+M)^{-1}\preceq M$ | Bounds the top-left block of the projector |
| **Block norm bound** (8.3) | For PSD $\begin{bmatrix}A&B\\B^*&C\end{bmatrix}$: norm $\le\|A\|+\|C\|$ | Kills the off-diagonal block |
| **Projector facts** (8.4–8.5) | Unitary conjugation of projectors; if $\mathrm{range}(N)\subset\mathrm{range}(M)$ then $\|(I-P_M)A\|\le\|(I-P_N)A\|$ | Bigger range ⇒ smaller error |
| **Prop. 8.6** | $\|PM\|\le\|P(MM^*)^qM\|^{1/(2q+1)}$ | The engine of the power-iteration analysis |

Here $M\preceq N$ means "$N-M$ is positive semidefinite" (PSD: symmetric with nonnegative eigenvalues).

### 8.3 Theorem 9.1: the deterministic error bound

> **Theorem 9.1.** If $\Omega_1$ has full row rank, then
> $$
> \|(I-P_Y)A\|^2\;\le\;\|\Sigma_2\|^2+\big\|\Sigma_2\Omega_2\Omega_1^\dagger\big\|^2 .
> $$
> (valid in both spectral and Frobenius norm; $\dagger$ = pseudoinverse)

**Interpretation:** the first term is the unavoidable tail error. The second is the *price of randomness*: how badly the contamination $\Omega_2$, amplified by $\Omega_1^\dagger$ (which is large if $\Omega_1$ is nearly rank-deficient), distorts the sample.

**Proof walk-through (spectral norm):**

1. **Get rid of $U$.** By unitary invariance, replace $A$ by $\tilde A=U^*A=\Sigma V^*$ and $Y$ by $\tilde Y=\begin{bmatrix}\Sigma_1\Omega_1\\\Sigma_2\Omega_2\end{bmatrix}$. Same error.
2. **Build a "flattened" comparison matrix.** Define
$$
Z=\tilde Y\,\Omega_1^\dagger\Sigma_1^{-1}=\begin{bmatrix}I\\F\end{bmatrix},\qquad F=\Sigma_2\Omega_2\Omega_1^\dagger\Sigma_1^{-1}.
$$
(We used $\Omega_1\Omega_1^\dagger=I$, valid when $\Omega_1$ has full row rank.) Since $Z$ is $\tilde Y$ times something, $\mathrm{range}(Z)\subset\mathrm{range}(\tilde Y)$, so by Prop 8.5
$$\|(I-P_{\tilde Y})\tilde A\|\le\|(I-P_Z)\tilde A\|.$$
We've *replaced the messy range by a clean one*: $Z=\begin{bmatrix}I\\F\end{bmatrix}$ is the identity plus a perturbation $F$. If $F=0$, its range is exactly the top-$k$ coordinates, which is what we want to capture.
3. **Square it and use $\tilde A=\Sigma V^*$:**
$$\|(I-P_Z)\tilde A\|^2=\|\Sigma^*(I-P_Z)\Sigma\|.$$
4. **Compute the projector.** With the explicit formula $P_Z=Z(Z^*Z)^{-1}Z^*$ one gets
$$
I-P_Z=\begin{bmatrix}I-(I+F^*F)^{-1}&-(I+F^*F)^{-1}F^*\\-F(I+F^*F)^{-1}&I-F(I+F^*F)^{-1}F^*\end{bmatrix}\preceq\begin{bmatrix}F^*F&\mathcal B\\\mathcal B^*&I\end{bmatrix},
$$
using Prop 8.2 for the top-left block and the conjugation rule for the bottom-right ($\mathcal B$ denotes the off-diagonal block).
5. **Conjugate by $\Sigma$** (conjugation rule):
$$\Sigma^*(I-P_Z)\Sigma\preceq\begin{bmatrix}\Sigma_1^*F^*F\Sigma_1&\ast\\\ast&\Sigma_2^*\Sigma_2\end{bmatrix}.$$
Here the "magic cancellation" occurs: $F\Sigma_1=\Sigma_2\Omega_2\Omega_1^\dagger$, since the $\Sigma_1^{-1}$ in $F$ cancels the $\Sigma_1$.
6. **Block norm bound (Prop 8.3):** the norm of a PSD block matrix is at most the sum of the norms of its diagonal blocks:
$$\|\Sigma^*(I-P_Z)\Sigma\|\le\|F\Sigma_1\|^2+\|\Sigma_2\|^2=\|\Sigma_2\Omega_2\Omega_1^\dagger\|^2+\|\Sigma_2\|^2.\ \blacksquare$$

**The paper says** this sharpens the earlier bound of Boutsidis–Drineas–Mahoney (the earlier one lacked the squares).

### 8.4 Gaussian test matrices (§10)

**Why Gaussians are analyzable:**

- **Rotation invariance:** $V^*\Omega$ is again a standard Gaussian matrix. Since $\Omega_1$ and $\Omega_2$ are disjoint row-blocks of $V^*\Omega$, they are **independent Gaussian matrices**. That lets us handle $\Omega_2$ by conditioning on $\Omega_1$.
- Full row rank of the $k\times(k+p)$ matrix $\Omega_1$ holds with probability one.

**Needed facts (Props 10.1–10.2):** for fixed matrices $S,T$ and Gaussian $G$,

$$
\big(\mathbb E\|SGT\|_F^2\big)^{1/2}=\|S\|_F\|T\|_F,\qquad
\mathbb E\|SGT\|\le\|S\|\|T\|_F+\|S\|_F\|T\|
$$

(the first is a direct calculation, since every entry has variance 1; the second is due to Gordon's work on Gaussian processes), and for a $k\times(k+p)$ Gaussian $G$,

$$
\big(\mathbb E\|G^\dagger\|_F^2\big)^{1/2}=\sqrt{\tfrac{k}{p-1}},\qquad
\mathbb E\|G^\dagger\|\le\frac{e\sqrt{k+p}}{p}.
$$

The first one comes from $\|G^\dagger\|_F^2=\mathrm{tr}\,(GG^*)^{-1}$, the trace of an **inverse Wishart** matrix, with a known expectation.

> 🔑 **Intuition for the $1/(p-1)$:** a square Gaussian matrix is nearly singular (its smallest singular value is tiny, about $1/\sqrt n$), so its pseudoinverse is huge. A *fat* Gaussian ($k\times(k+p)$) has smallest singular value roughly $\sqrt{k+p}-\sqrt k$, which grows with $p$. So oversampling directly fights the instability of $\Omega_1^\dagger$. That's the mathematical reason for $p$.

**Theorem 10.5 (average Frobenius error).** For $k\ge2$, $p\ge2$:
$$
\mathbb E\|(I-P_Y)A\|_F\le\Big(1+\frac{k}{p-1}\Big)^{1/2}\Big(\sum_{j>k}\sigma_j^2\Big)^{1/2}.
$$

*Proof sketch:* by Theorem 9.1 and Hölder, $\mathbb E\|\cdot\|_F\le(\|\Sigma_2\|_F^2+\mathbb E\|\Sigma_2\Omega_2\Omega_1^\dagger\|_F^2)^{1/2}$. Condition on $\Omega_1$ and apply $\mathbb E\|SGT\|_F^2=\|S\|_F^2\|T\|_F^2$ to $\Omega_2$: get $\|\Sigma_2\|_F^2\,\mathbb E\|\Omega_1^\dagger\|_F^2=\|\Sigma_2\|_F^2\cdot\frac{k}{p-1}$. Done.

**Theorem 10.6 (average spectral error).**
$$
\mathbb E\|(I-P_Y)A\|\le\Big(1+\sqrt{\tfrac{k}{p-1}}\Big)\sigma_{k+1}+\frac{e\sqrt{k+p}}{p}\Big(\sum_{j>k}\sigma_j^2\Big)^{1/2}.
$$

*Proof sketch:* same, but use Gordon's bound $\mathbb E\|\Sigma_2G\Omega_1^\dagger\|\le\|\Sigma_2\|\|\Omega_1^\dagger\|_F+\|\Sigma_2\|_F\|\Omega_1^\dagger\|$.

*Interesting phenomenon:* the **spectral** error also involves the **Frobenius tail** $\big(\sum_{j>k}\sigma_j^2\big)^{1/2}$: a flat tail hurts even the spectral norm. With $p\sim k$ the first term has constant factor and the second has order $k^{-1/2}$.

**Theorem 1.1** (the introduction's showcase) is a simplified consequence:
$$
\mathbb E\|A-QQ^*A\|\le\Big[1+\frac{4\sqrt{k+p}}{p-1}\sqrt{\min\{m,n\}}\Big]\sigma_{k+1}.
$$
It comes from bounding $(\sum_{j>k}\sigma_j^2)^{1/2}\le\sqrt{\min\{m,n\}-k}\;\sigma_{k+1}$, which is the worst case of a perfectly *flat* tail.

> ⚠️ **Be honest in your talk about what this number means.** For $k=p=10$, $\min\{m,n\}=10^4$ the bracket in Theorem 1.1 is about $200$. That is a worst-case guarantee for flat spectra. Realistic spectra decay and then the second term of Theorem 10.6 is tiny, so errors are typically within a small constant of $\sigma_{k+1}$ (as the experiments show). Also: the *power iteration* (below) is exactly the cure for the flat-tail case.

**Probability bounds (§10.3).** Averages could in principle hide rare disasters. Concentration of measure shows they don't:

- **Prop 10.3:** for an $L$-Lipschitz function $h$ of a Gaussian matrix, $\mathbb P\{h(G)\ge\mathbb E\,h(G)+Lt\}\le e^{-t^2/2}$.
- **Prop 10.4** (tail bounds for $\|G^\dagger\|$, based on Chen–Dongarra), and the new Theorem A.6 (appendix, long and "brute force") for $\|G^\dagger\|_F$.
- **Result (eq. 1.9):** with probability at least $1-6p^{-p}$,
$$
\|A-QQ^*A\|\le\Big[1+11\sqrt{k+p}\sqrt{\min\{m,n\}}\Big]\sigma_{k+1}.
$$
For $p=5$: failure $\le6/5^5\approx0.2\%$. For $p=10$: $\le6\cdot10^{-10}$. This is the "superexponential decay in $p$" promised in the intro. Proof idea for the spectral bound: define the "good event" $E_t$ where $\|\Omega_1^\dagger\|$ and $\|\Omega_1^\dagger\|_F$ are controlled, apply Gaussian concentration to $\Omega_2$ conditional on it, then un-condition with a union bound.

### 8.5 Power scheme theory (§9.3, §10.4)

**Theorem 9.2:** with $B=(AA^*)^qA$ and $Z=B\Omega$,
$$\|(I-P_Z)A\|\le\|(I-P_Z)B\|^{1/(2q+1)}.$$
It's an immediate consequence of **Prop 8.6**, $\|PM\|\le\|P(MM^*)^qM\|^{1/(2q+1)}$ for any orthogonal projector $P$, a matrix version of the scalar fact $|px|^q\le|p||x|^q$ for $|p|\le1,q\ge1$. Its proof uses the SVD to reduce to $\|RDR\|^t\le\|RD^tR\|$ for a projector $R$ and diagonal $D\ge0$, which follows from **Jensen's inequality** (convexity of $z\mapsto z^t$).

Combine with Theorem 9.1 applied to $B$ (whose singular values are $\sigma_j^{2q+1}$):
$$
\|(I-P_Z)A\|\le\Big(1+\|\Omega_2\Omega_1^\dagger\|^2\Big)^{1/(4q+2)}\sigma_{k+1}.
$$
The "suboptimality factor" is raised to the power $\frac1{4q+2}$, so it **goes to 1 exponentially fast in $q$**.

**Corollary 10.10 / Theorem 1.2** (for Gaussian $\Omega$, with $\ell=2k$ columns):
$$
\mathbb E\|A-U\Sigma V^*\|\le\Big[1+4\sqrt{\tfrac{2\min\{m,n\}}{k-1}}\Big]^{1/(2q+1)}\sigma_{k+1}.
$$
With $k=10$, $\min\{m,n\}=10^4$ the bracket is $\approx190$, and the bound becomes about $5.7$ for $q=1$, $2.9$ for $q=2$, $2.1$ for $q=3$. By $q\sim\log\min\{m,n\}$ the factor is essentially 1. The paper stresses *"This result is new."*

*(Limitation, Remark 10.1: there's no Frobenius-norm analogue because Prop. 8.6 fails for the Frobenius norm.)*

### 8.6 Truncated SVD (§9.4, Theorem 9.3 — argument due to Ming Gu)

If $\hat A_{(k)}$ is the best rank-$k$ approximation of $P_ZA$, then
$$\|A-\hat A_{(k)}\|\le\sigma_{k+1}+\|(I-P_Z)A\|.$$
*Proof:* triangle inequality: $\|A-\hat A_{(k)}\|\le\|A-P_ZA\|+\|P_ZA-\hat A_{(k)}\|$, and the second term is at most $\|P_ZA-P_ZA_{(k)}\|\le\|A-A_{(k)}\|=\sigma_{k+1}$ since $\hat A_{(k)}$ is the *best* rank-$k$ approximation of $P_ZA$ (hence better than the particular rank-$k$ matrix $P_ZA_{(k)}$) and projectors don't increase norms.

### 8.7 SRFT theory (§11)

**Theorem 11.1 (SRFT preserves geometry).** For a fixed $n\times k$ orthonormal $V$ and an SRFT with
$$4\big[\sqrt k+\sqrt{8\log(kn)}\big]^2\log k\le\ell\le n,$$
we have $0.40\le\sigma_k(V^*\Omega)$ and $\sigma_1(V^*\Omega)\le1.48$ with failure probability $O(k^{-1})$. So $\Omega_1$ has full row rank and $\|\Omega_1^\dagger\|^2\le 1/0.4^2<7$. (The proof follows Tropp's analysis of the subsampled randomized Hadamard transform and is omitted in the paper.)

**Theorem 11.2 (error for SRFT).** With the same $\ell$,
$$
\|(I-P_Y)A\|\le\sqrt{1+7n/\ell}\;\sigma_{k+1},\qquad
\|(I-P_Y)A\|_F\le\sqrt{1+7n/\ell}\Big(\sum_{j>k}\sigma_j^2\Big)^{1/2}
$$
with failure probability $O(k^{-1})$.
*Proof:* Theorem 9.1 gives error $\le\|\Sigma_2\|\big[1+\|\Omega_1^\dagger\|^2\|\Omega_2\|^2\big]^{1/2}$. Theorem 11.1 gives $\|\Omega_1^\dagger\|^2<7$. And $\|\Omega_2\|\le\|\Omega\|=\sqrt{n/\ell}$ deterministically.

**Why the $\log k$ is necessary (Remark 11.2)** is a lovely **coupon collector** argument. Take $n=k^2$ and let $V$ pick every $k$-th coordinate of the identity (a "spiky" subspace). After applying $D$ and $F$, $V^*DF$ is $k$ copies of a $k\times k$ DFT side by side. The SRFT then samples $\ell$ columns of this. To get full rank $k$ you need at least one copy of *each* of the $k$ distinct columns. That's the coupon collector's problem, which needs $\Theta(k\log k)$ draws. You can't beat that with a sampling-based test matrix.

**Comparison of failure probabilities:** SRFT fails with probability about $1/k$, Gaussian with probability about $e^{-(\ell-k)}$. This isn't a flaw of the analysis: *discrete sampling schemes inherently fail more often*, though empirically $\ell=k+20$ suffices.

### 8.8 One-page "proof map" (print this)

```
                  Theorem 9.1 (deterministic)
        err² ≤ σ_{k+1}² + ‖Σ2 Ω2 Ω1†‖²      [needs Ω1 full row rank]
                         │
        ┌────────────────┴──────────────────┐
   Gaussian Ω                          SRFT Ω
 Ω1, Ω2 independent Gaussians     Thm 11.1: σ_min(Ω1) ≥ 0.4
 E‖Ω1†‖_F² = k/(p−1)               ‖Ω2‖ ≤ √(n/ℓ)
 Gordon: E‖SGT‖ ≤ …               ⇒ Thm 11.2
 ⇒ Thm 10.5, 10.6, 1.1
 concentration ⇒ Thm 10.7–10.8, eq (1.9)
                         │
              Prop 8.6 / Thm 9.2 (power scheme)
     err ≤ (err for (AA*)^q A)^{1/(2q+1)}  ⇒  Cor 10.10, Thm 1.2
```

---

## 9. Historical context: who thought of what

The paper's §2 is a literature survey. It's good for a "related work" slide. The cleanest way to remember it: **three families of randomized low-rank approximation** (plus two cousins), all following *preprocess → sample → postprocess*.

| Family | Idea | Key names/dates |
|---|---|---|
| **Sparsification / quantization** | Replace $A$ by a sparser/coarser random surrogate with $\mathbb E[\tilde A]=A$ | Achlioptas–McSherry; Arora–Hazan–Kale; Karger (graphs) |
| **Column (row) selection** | A few columns carry most of the action. Sample columns with probability $\propto$ squared norm or **leverage scores** | Frieze–Kannan–Vempala (1998); Drineas–Kannan–Mahoney; Rudelson–Vershynin; Deshpande et al. (volume sampling) |
| **Dimension reduction** (this paper's family) | Random linear map compresses the rows/range; no precomputed sampling probabilities | Papadimitriou et al. 1998 (latent semantic indexing); Sarlós 2006; Woolfe et al. 2008 (fast SRFT); Martinsson–Rokhlin–Tygert 2006 (Gaussian + oversampling); Rokhlin–Szlam–Tygert 2009 (power iteration) |
| Submatrix methods | ID / CUR / skeleton decompositions | Goreinov et al.; Cheng et al.; Drineas–Mahoney (CUR) |
| Compressive sensing | Recover a compressible object from few random measurements | Candès–Romberg–Tao, Donoho (2004); Recht–Fazel–Parrilo; Candès–Recht (matrix completion) |

**Key difference from compressive sensing (§2.1.6):** CS assumes you have *limited access* to the data and must *reconstruct* it from few measurements (typically via convex optimization). Matrix approximation assumes you have the *whole matrix* and want a cheaper representation, so it needs fewer samples and less computation.

**Intellectual ancestors (§2.2):**

1. **Random embeddings**: Johnson–Lindenstrauss (1984), Bourgain (1985); fast JL transform (Ailon–Chazelle 2006), whose speed idea (FFT + randomization) is the ancestor of the SRFT.
2. **Data streams / sketching**: Alon–Matias–Szegedy; Muthukrishnan's argument that modern data is characterized by *how it arrives*.
3. **Numerical linear algebra's own random tricks**: random starting vectors for the power method (ensures a nonzero component along the dominant eigenvector), Dixon's norm estimator (the ancestor of §4.3), von Neumann–Goldstine's random error models.
4. **Scientific computing**: Monte Carlo (von Neumann–Ulam); gave "randomized" a bad name due to its slow $m^{-1/2}$ convergence.
5. **Geometric functional analysis**: Dvoretsky's theorem and Milman's probabilistic proof (1971) that *random subspaces* have nice geometry, the birth of the **concentration of measure** phenomenon; Kashin's widths of the $\ell_1$ ball. This is where the *proof techniques* (Gaussian concentration, Gordon's inequality) come from.

---

## 10. A critical look

Good seminars include a bit of critique. Here are fair points, all grounded in the paper's own statements.

**Strengths**

- Clean modular design: one simple randomized subroutine + classical post-processing.
- Rigorous, *computable* error bounds and failure probabilities. This distinguishes it from heuristic randomized methods.
- Embraces modern hardware reality (passes, blocking, parallelism).
- Practical guidance: $p=5$–$10$, $q=1$–$2$, orthonormalize between powers, estimate error cheaply.

**Limitations / honest caveats**

1. **Flat spectra:** the basic scheme's worst-case guarantee degrades with $\sqrt{\min\{m,n\}}$. You need power iteration, i.e. more passes.
2. **Krylov can be more accurate per flop** in the sparse-matvec setting (the authors admit it, §1.4.2, 6.2).
3. **Theory vs. practice gaps:** the SRFT theory requires $\ell\sim k\log k$ and fails with probability $\sim 1/k$, while in practice $\ell=k+20$ works. Truncation's worst-case $+\sigma_{k+1}$ term "often does not appear in practice" but this isn't theoretically explained (Remark 9.1).
4. **Single-pass methods** can be noticeably worse (ill-conditioned $Q^*\Omega$).
5. **No Frobenius-norm power-scheme theory**, and **it's unknown whether power iteration can use fast transforms** (Remark 4.7).
6. **Analysis is in exact arithmetic** and for Gaussians mainly in the real case (the complex case is claimed to be better).
7. Accuracy only down to the level of the tolerance: they say "any tolerance above machine precision".
8. The experiments are from 2010-era hardware and small-ish by today's standards (though they reference larger-scale results elsewhere).

**Legacy (not in the paper, but useful context):** the randomized SVD is now a standard tool: for instance, scikit-learn's `PCA(svd_solver="randomized")`, `TruncatedSVD`, and `randomized_svd` are based on this line of work. (Verify the exact solver defaults for the version you demo.)

---

## 11. Seminar toolkit

### 11.1 Code you can demo (NumPy, ~20 lines)

```python
import numpy as np

def randomized_range_finder(A, ell, q=0, seed=0):
    """Stage A: orthonormal Q with A ≈ Q Q^T A (Algorithm 4.1 / 4.4)."""
    rng = np.random.default_rng(seed)
    m, n = A.shape
    Omega = rng.standard_normal((n, ell))        # Step 1: random test matrix
    Q, _ = np.linalg.qr(A @ Omega)               # Steps 2-3: sample + orthonormalize
    for _ in range(q):                           # subspace iteration (Alg. 4.4)
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)
    return Q

def randomized_svd(A, k, p=10, q=2, seed=0):
    """Stage A + Stage B (direct SVD, Algorithm 5.1), truncated to rank k."""
    Q = randomized_range_finder(A, k + p, q, seed)
    B = Q.T @ A                                   # small (k+p) x n matrix
    Ut, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ut)[:, :k], s[:k], Vt[:k]

# --- demo: matrix with exponentially decaying singular values ---
rng = np.random.default_rng(1)
m, n, k = 2000, 1000, 20
U0, _ = np.linalg.qr(rng.standard_normal((m, n)))
V0, _ = np.linalg.qr(rng.standard_normal((n, n)))
s_true = 2.0 ** (-np.arange(n) / 5)              # slow-ish decay
A = (U0 * s_true) @ V0.T

for q in (0, 1, 2):
    U, s, Vt = randomized_svd(A, k, p=10, q=q)
    err = np.linalg.norm(A - (U * s) @ Vt, 2)
    print(f"q={q}: error = {err:.3e}   optimal sigma_(k+1) = {s_true[k]:.3e}")
```

You should see the ratio *error / optimal* drop as $q$ grows, a live demonstration of $C^{1/(2q+1)}$. A good extra: plot error vs. oversampling $p=0,2,5,10,20$ to show why oversampling matters.

A tiny error estimator (Lemma 4.1):

```python
def estimate_error(A, Q, r=10, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((A.shape[1], r))
    R = A @ W - Q @ (Q.T @ (A @ W))              # (I - QQ^T) A ω_i
    return 10 * np.sqrt(2/np.pi) * np.linalg.norm(R, axis=0).max()
```

### 11.2 Suggested slide plan (about 35–45 min)

| # | Slide | Time | Key message |
|---|---|---|---|
| 1 | Title + hook: "How to compress a matrix with dice" | 1 | Randomness + linear algebra |
| 2 | Why low-rank matters (PCA, images, PDEs) | 2 | $A\approx BC$ saves storage/time |
| 3 | The modern cost model (flops vs passes vs parallelism) | 2 | Why classical algorithms struggle |
| 4 | SVD in one picture (sphere → ellipsoid); Eckart–Young; $\sigma_{k+1}$ as yardstick | 4 | The target to compare against |
| 5 | The two-stage framework | 3 | Randomness only in Stage A |
| 6 | Intuition: toy rank-1 example + noisy case + oversampling | 3 | Random samples span the range |
| 7 | Proto-algorithm + Algorithm 4.1 | 2 | 3 lines |
| 8 | Stage B: Direct SVD (3 steps) | 2 | Error doesn't degrade |
| 9 | Adaptive version + error estimator | 3 | Don't need to know $k$ |
| 10 | Power iteration (and why re-orthonormalize) | 4 | $C\to C^{1/(2q+1)}$ |
| 11 | SRFT: $D$, $F$, $R$ and $O(mn\log\ell)$ | 3 | Scramble–spread–sample |
| 12 | Cost comparison across 3 environments | 3 | When randomized wins |
| 13 | Experiments (decay, power effect, 6× speedup) | 3 | Does it work? Yes. |
| 14 | Theory part 1: $\Omega_1,\Omega_2$ split, Theorem 9.1 | 4 | Signal vs. contamination |
| 15 | Theory part 2: Gaussian ⇒ $k/(p-1)$; Theorems 1.1/1.2 | 3 | Why oversampling works |
| 16 | Limits & open questions | 1 | Honest critique |
| 17 | Take-aways | 1 | Summary |

**If you're short on time:** drop SRFT details, single-pass, Nyström, and compress theory to "split, Ω₁ must be well-conditioned, Gaussian makes it so, oversampling helps".

### 11.3 Likely audience questions + answers

**Q: Doesn't randomness mean it might just fail?**
A: Yes, with a user-controlled probability that decays superexponentially in the oversampling $p$: at most $6p^{-p}$ (e.g. $\sim10^{-10}$ for $p=10$). And you can certify the result afterwards with the a posteriori estimator (4.3), whose failure probability is $10^{-r}$.

**Q: Why Gaussian? Why not any random numbers?**
A: Gaussians are rotation-invariant, so $\Omega_1$ and $\Omega_2$ are independent Gaussians, which makes the exact analysis possible. In practice, other distributions (even random signs) behave similarly, and the quality of the random number generator barely matters.

**Q: Why oversample? We know $k$.**
A: With exactly $k$ samples, $\Omega_1$ is a *square* Gaussian matrix, which is nearly singular, so $\|\Omega_1^\dagger\|$ is huge and the error bound blows up. With $p$ extra columns, $\mathbb E\|\Omega_1^\dagger\|_F^2=k/(p-1)$.

**Q: Why not just pick $k$ random columns of $A$?**
A: That's the *column-selection* family. It works if you sample with the right probabilities (norms/leverage scores), but those need a pre-pass, and the guarantees are additive-type. Random projections mix all columns, need no precomputation, and the paper notes they have advantages in accuracy.

**Q: How is this different from PCA?**
A: PCA *is* a truncated SVD of the centered data matrix. This paper gives a fast approximate way to compute it.

**Q: Why use QR instead of just using $Y$?**
A: $Y$'s columns are nearly parallel (dominated by the top singular directions), so $Y$ is ill-conditioned. QR gives a well-conditioned orthonormal basis. In the power scheme, re-orthonormalization also prevents small singular directions from being lost to round-off.

**Q: What is a "pass" and why do we care?**
A: One sequential scan over all entries. For matrices stored on disk/streamed, reading data dominates cost. Classical algorithms need $\gtrsim k$ passes. Randomized needs a constant number ($2(q+1)$), or one with the single-pass variants.

**Q: How do you choose $q$?**
A: $q=1$ or $2$ typically; empirically almost no gain beyond 3–5 passes.

**Q: Is it always faster than Krylov / Lanczos?**
A: No. Krylov can be more accurate per flop. Randomized is more robust, simpler, and much more parallel/blockable.

**Q: Is the error bound tight?**
A: No. The authors say constants and forms aren't optimal. The experiments show errors are very close to $\sigma_{k+1}$.

**Q: What is the SRFT's benefit if it needs more oversampling?**
A: Asymptotic speedup: $O(mn\log\ell)$ vs $O(mn\ell)$ for the sketch. Empirically $\ell=k+20$ suffices, giving up to $\sim6\times$ speedups at $n=4096$ in their tests.

### 11.4 Common mistakes to avoid when presenting

- ❌ "The algorithm is random so the answer is approximately random." → The *error* has a very tight distribution. Randomness affects *which* good basis you get.
- ❌ "It gives the exact top-$k$ singular vectors." → It gives a subspace with error near $\sigma_{k+1}$; singular vectors are only accurate for well-separated singular values.
- ❌ Mixing up spectral and Frobenius norm results: the power-iteration theory is spectral-only.
- ❌ Forgetting that Stage B (Algorithm 5.1) needs a second pass over $A$ (or use the single-pass variant).
- ❌ Saying the bound in Theorem 1.1 is "small": for flat spectra it's large (it's a worst-case statement).

### 11.5 Three-sentence summary (for the last slide)

1. Many big matrices are nearly low rank, and the key step is finding a small orthonormal basis $Q$ that captures their range.
2. Multiplying $A$ by a random matrix with a few extra columns produces such a basis with overwhelming probability, and a classical SVD of the small matrix $Q^*A$ finishes the job.
3. With power iterations, structured random matrices, and single-pass variants, this approach is accurate, fast, robust and well suited to modern hardware, backed by rigorous bounds.

---

## 12. Notation cheat sheet & glossary

### Notation

| Symbol | Meaning |
|---|---|
| $A\in\mathbb{R}^{m\times n}$ | input matrix |
| $A^*$ | conjugate transpose (= transpose for real matrices) |
| $k$ | target rank; $p$ oversampling; $\ell=k+p$ number of samples |
| $\Omega$ | $n\times\ell$ random test matrix |
| $Y=A\Omega$ | sample matrix |
| $Q$ | orthonormal basis, $QQ^*=P_Y$ |
| $P_Y$ | orthogonal projector onto $\mathrm{range}(Y)$ |
| $\sigma_j$ | $j$-th largest singular value of $A$ |
| $A_{(k)}$ | best rank-$k$ approximation (truncated SVD) |
| $\|\cdot\|$, $\|\cdot\|_F$ | spectral and Frobenius norm |
| $M^\dagger$ | pseudoinverse |
| $\Sigma_1,\Sigma_2$ | top-$k$ / tail singular values |
| $\Omega_1=V_1^*\Omega$, $\Omega_2=V_2^*\Omega$ | test matrix in singular-vector coordinates |
| $q$ | power iteration exponent |
| $T_{\text{mult}}$ | cost of one matvec with $A$ |
| $\succeq0$, $\preceq$ | PSD ordering |

### Glossary

- **Range / column space:** all vectors of the form $Ax$.
- **Numerical rank:** number of singular values above a tolerance.
- **Orthonormal matrix:** columns orthogonal, unit length; $Q^*Q=I$.
- **QR factorization:** $Y=QR$, $Q$ orthonormal, $R$ upper triangular.
- **Orthogonal projector:** $P=P^*=P^2$; "drop onto a subspace".
- **SVD:** $A=U\Sigma V^*$; ellipsoid picture.
- **Eckart–Young–Mirsky:** truncated SVD is the best low-rank approximation.
- **Oversampling:** drawing $p$ more samples than the target rank.
- **Power iteration:** applying $(AA^*)$ repeatedly to sharpen the spectrum.
- **Krylov subspace:** $\mathrm{span}\{\omega,A\omega,\dots,A^{q-1}\omega\}$.
- **ID (interpolative decomposition):** $A=A_{(:,J)}X$ with bounded $X$.
- **Nyström method:** improved PSD approximation $A\approx(AQ)(Q^*AQ)^{-1}(AQ)^*$.
- **SRFT:** subsampled random Fourier transform $\sqrt{n/\ell}\,DFR$.
- **JL lemma:** random projection to $O(\log N)$ dims preserves pairwise distances.
- **Concentration of measure:** smooth functions of many independent random variables are nearly constant.
- **Pass:** one sequential scan of the data.
- **BLAS-3:** matrix–matrix kernels; high arithmetic intensity, fast.
- **Coupon collector:** drawing from $k$ types needs $\sim k\log k$ draws to see them all.

---

### 📚 If you read only three things from the paper itself

1. **§1.2–1.3 and the Proto-Algorithm box** (pages 3–6): the whole idea.
2. **§1.4–1.6** (pages 6–10): comparison, Theorems 1.1 and 1.2, randomized SVD.
3. **§9.2 (Theorem 9.1) with its proof** (pages 50–53): the heart of the theory.

Good luck with the seminar. May your singular values decay quickly and your oversampling be generous. 🎲
