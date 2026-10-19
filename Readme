# Column generation for the time-sampled Lyapunov LMI

This repository computes the transient-growth bound $\gamma$ from the **original** time-sampled LMI, Eq. (24) of Wei, Zhao & Liu (ACC 2026), at grid sizes where a direct SDP solve runs out of memory.

The constraints are not changed or approximated:
- the same $A_j$ as the baseline `transcient_growth_new.m`;
- the same forward-Euler step;
- the same uniform box $I \preceq P_j \preceq \gamma I$.

The method only changes **how** the SDP is solved. Every result comes with:

1. **An upper bound** $\gamma_{\rm UB}$, backed by matrices $P_j$ that are checked against every original constraint at full size.
2. **A certified lower bound** $\gamma_{\rm LB}$ from weak duality.

The true optimum always satisfies $\gamma_{\rm LB} \le \gamma^\star \le \gamma_{\rm UB}$. The run stops when the two agree to the requested accuracy (default relative gap $10^{-5}$, safely inside 4 significant figures).

---

## 1. The problem

The baseline builds $K = (T - t_0)/\Delta t$ matrices $A_1,\dots,A_K \in \mathbb{C}^{n\times n}$, with $n = 2(N-2)$, from the linearized Orr–Sommerfeld/Squire operator in energy coordinates at the times $t_j = t_0 + j\,\Delta t$. Eq. (24) is

```math
\begin{aligned}
\min_{\gamma,\;P_1,\dots,P_K}\quad & \gamma \\
\text{s.t.}\quad & I \preceq P_j \preceq \gamma I, && j = 1,\dots,K,\\
& M_j(P) := \frac{P_{j+1}-P_j}{\Delta t} + A_j^{\mathsf H} P_j + P_j A_j \preceq 0, && j = 1,\dots,K-1,
\end{aligned}
```

with every $P_j$ Hermitian. A direct solve has $K$ dense $n\times n$ matrix variables. Its memory grows like $K n^2$, and the interior-point cost per iteration grows much faster than that. That is what limits $N$.

Throughout, $H_j := A_j + A_j^{\mathsf H}$ is the energy-production matrix at time $t_j$. It is also exactly $M_j$ evaluated at $P = I$.

---

## 2. Low-rank parameterization

Each $P_j$ is written as the identity plus a correction confined to a small subspace:

```math
P_j = I + U_j S_j U_j^{\mathsf H}, \qquad U_j \in \mathbb{C}^{n\times r_j},\; U_j^{\mathsf H}U_j = I_{r_j},\; S_j = S_j^{\mathsf H}\in\mathbb{C}^{r_j\times r_j}.
```

The columns of $U_j$ are the **directions** of $P_j$, and $r_j \ll n$ is the hope.

Because $U_j$ is orthonormal and $P_j = I$ on the orthogonal complement of $\operatorname{range}(U_j)$, the box constraint reduces to a small one:

```math
I \preceq P_j \preceq \gamma I \iff 0 \preceq S_j \preceq (\gamma - 1) I_{r_j}.
```

Substituting into $M_j$ gives

```math
M_j = H_j + L_j(S),\qquad
L_j(S) = \frac{U_{j+1}S_{j+1}U_{j+1}^{\mathsf H} - U_jS_jU_j^{\mathsf H}}{\Delta t} + A_j^{\mathsf H}U_jS_jU_j^{\mathsf H} + U_jS_jU_j^{\mathsf H}A_j .
```

---

## 3. Exact reduction of each dynamics LMI (Schur complement)

$M_j$ is still $n\times n$, but $L_j(S)$ only acts on a small subspace. Every term of $L_j$ has its column space, and by symmetry its row space, inside

```math
\mathcal W_j = \operatorname{span}\big[\,U_j,\; U_{j+1},\; A_j^{\mathsf H}U_j,\; V_j^{+}\,\big],
```

where $V_j^{+}$ holds the eigenvectors of $H_j$ with positive eigenvalues. Let $W_j$ be an orthonormal basis of $\mathcal W_j$ (size $n\times m_j$) and $Q_j$ an orthonormal basis of its complement. Then $L_j Q_j = 0$, and in the basis $[W_j\;Q_j]$:

```math
[W_j\;Q_j]^{\mathsf H} M_j\, [W_j\;Q_j] =
\begin{bmatrix}
W_j^{\mathsf H}H_jW_j + W_j^{\mathsf H}L_jW_j & W_j^{\mathsf H}H_jQ_j\\
Q_j^{\mathsf H}H_jW_j & Q_j^{\mathsf H}H_jQ_j
\end{bmatrix}.
```

Since $V_j^{+} \subset \mathcal W_j$, the block $Q_j^{\mathsf H}H_jQ_j$ is **negative definite**. The code checks this every round and stops with an error if it fails. The Schur complement lemma then gives an **exact** equivalence:

```math
M_j \preceq 0 \iff
\underbrace{W_j^{\mathsf H}H_jW_j - W_j^{\mathsf H}H_jQ_j\,(Q_j^{\mathsf H}H_jQ_j)^{-1}\,Q_j^{\mathsf H}H_jW_j}_{R_j} + W_j^{\mathsf H}L_j(S)W_j \preceq 0 .
```

This is cleanest to write with the **lift map**

```math
T_j := W_j - Q_j F_j,\qquad F_j := (Q_j^{\mathsf H}H_jQ_j)^{-1}Q_j^{\mathsf H}H_jW_j ,
```

for which $R_j = T_j^{\mathsf H}H_jT_j$ and $W_j^{\mathsf H}L_jW_j = T_j^{\mathsf H}L_jT_j$ (because $L_jQ_j = 0$). So the reduced constraint is simply

```math
T_j^{\mathsf H}\,M_j(P)\,T_j \preceq 0 \qquad (\text{size } m_j\times m_j,\ \text{equivalent to } M_j\preceq 0).
```

`reduce_step` builds $W_j$, $T_j$ and $R_j$.

---

## 4. The restricted SDP (one round)

For fixed directions $U_j$, solve

```math
\begin{aligned}
\min_{\gamma,\,S,\,s}\quad & \gamma + \rho \sum_j s_j\\
\text{s.t.}\quad & S_j \succeq 0,\qquad (\gamma-1)I - S_j \succeq 0, && j=1,\dots,K,\\
& s_j I - T_j^{\mathsf H}M_j(P)\,T_j \succeq 0,\qquad s_j \ge 0, && j=1,\dots,K-1 .
\end{aligned}
```

The slacks $s_j$ (penalty $\rho = 10^4$) keep the problem feasible in the first rounds, before the directions are good enough. The printout says `NOT yet feasible` while some $s_j>0$. Once every $s_j = 0$, the $P_j$ are feasible for Eq. (24), so **$\gamma$ is a valid upper bound**. The printout then says `valid UB`.

Two implementation details:

- **Real form.** MOSEK works with real matrices, and YALMIP does not return MOSEK's duals for complex constraints. Each Hermitian constraint $X \succeq 0$ is therefore passed in the equivalent real form

  ```math
  \begin{bmatrix}\operatorname{Re}X & -\operatorname{Im}X\\ \operatorname{Im}X & \operatorname{Re}X\end{bmatrix}\succeq 0 .
  ```

  If $Z = \begin{bmatrix}Z_{11}&Z_{12}\\Z_{21}&Z_{22}\end{bmatrix}$ is the dual of that real constraint, the complex dual is $Y = (Z_{11}+Z_{22}) + i\,(Z_{21}-Z_{12})$. It satisfies $\langle Z,\operatorname{emb}(M)\rangle = \operatorname{Re}\operatorname{tr}(YM)$ for every Hermitian $M$.
- **Numeric assembly.** Each $S_j$ is expanded in a basis of Hermitian matrices, $S_j=\sum_k x_k E_k$. Every constraint is precomputed as a constant plus a numeric matrix times $x$, using $\operatorname{vec}(BEC^{\mathsf H}) = (\overline{C}\otimes B)\operatorname{vec}(E)$. This avoids YALMIP's slow symbolic matrix products.

---

## 5. Where the directions come from: the dual

### 5.1 Dual of the full problem

Attach multipliers $\Omega_j \succeq 0$ to $P_j \preceq \gamma I$, $\Lambda_j\succeq 0$ to $P_j\succeq I$, and $Y_j \succeq 0$ to $M_j \preceq 0$. The Lagrangian is

```math
\mathcal L = \gamma + \sum_{j=1}^{K}\langle \Omega_j,\,P_j-\gamma I\rangle + \sum_{j=1}^{K}\langle\Lambda_j,\,I-P_j\rangle + \sum_{j=1}^{K-1}\langle Y_j,\,M_j(P)\rangle,
\qquad \langle X,Y\rangle = \operatorname{tr}(XY).
```

Setting the derivative with respect to $\gamma$ to zero gives $\sum_j \operatorname{tr}\Omega_j = 1$.

Collecting the coefficient of each $P_j$ (with $\langle Y_j, A_j^{\mathsf H}P_j + P_jA_j\rangle = \langle A_jY_j + Y_jA_j^{\mathsf H},\,P_j\rangle$) gives the stationarity condition

```math
\boxed{\;\Lambda_j = \Omega_j + \frac{Y_{j-1} - Y_j}{\Delta t} + A_jY_j + Y_jA_j^{\mathsf H}\;}
\qquad (Y_0 = 0,\; Y_K = 0).
```

When both conditions hold, every $P_j$ and $\gamma$ term cancels, and the dual objective is $\sum_j \operatorname{tr}\Lambda_j$. **Weak duality:** for any multipliers that satisfy these conditions with $\Omega_j, Y_j, \Lambda_j \succeq 0$,

```math
\sum_{j=1}^{K}\operatorname{tr}\Lambda_j \;\le\; \gamma^\star .
```

### 5.2 Lifting the restricted duals

The solver returns small duals: $y_j$ ($m_j\times m_j$) for each reduced dynamics constraint and $\omega_j$ ($r_j\times r_j$) for each $(\gamma-1)I - S_j\succeq 0$. Because the reduced constraint is $T_j^{\mathsf H}M_jT_j\preceq 0$,

```math
\langle y_j,\,T_j^{\mathsf H}M_jT_j\rangle = \langle T_jy_jT_j^{\mathsf H},\,M_j\rangle ,
```

so the full-size multipliers are

```math
Y_j = T_j\,y_j\,T_j^{\mathsf H},\qquad \Omega_j = U_j\,\omega_j\,U_j^{\mathsf H}.
```

$\Lambda_j$ is then **defined** by the boxed stationarity equation, so stationarity in $P$ holds exactly by construction.

### 5.3 Pricing: choosing new directions

On $\operatorname{range}(U_j)$, $\Lambda_j$ matches the multiplier of $S_j\succeq 0$, which the solver keeps PSD. Negative eigenvalues of $\Lambda_j$ can therefore only appear **outside** the current directions.

The restricted problem implicitly fixes $P_j = I$ on that complement. A negative eigenvalue $\lambda<0$ of $\Lambda_j$ with eigenvector $v$ means this hidden constraint is pushing the wrong way. Letting $P_j$ grow along $v v^{\mathsf H}$ changes the Lagrangian at the first-order rate $\lambda < 0$, so adding $v$ **could lower** $\gamma$. This is the SDP version of a negative reduced cost in linear-programming column generation.

Each round, for every $j$:

1. compute the eigen-decomposition of $\Lambda_j$;
2. take the eigenvectors of the most negative eigenvalues, up to `addmax` (default 2) per time step, ignoring eigenvalues above $-10^{-7}\max|\lambda|$;
3. set $U_j \leftarrow \operatorname{orth}\,[\,U_j,\;v_1,\dots\,]$ and solve the restricted SDP again.

**Why stopping certifies the answer.** If no $\Lambda_j$ has a negative eigenvalue, the lifted multipliers are feasible for the full dual. Then $\sum_j\operatorname{tr}\Lambda_j = \gamma_{\rm LB}$ is a valid lower bound that equals the restricted optimum $\gamma_{\rm UB}$ (by strong duality of the restricted SDP). No direction anywhere in $\mathbb{C}^n$ can lower $\gamma$, so $\gamma_{\rm UB}$ is the optimum of Eq. (24).

### 5.4 A lower bound that stays rigorous with an inexact solver

Solvers satisfy the dual conditions only approximately. Each round, the code turns the approximate duals into an exactly feasible dual point:

1. Project $y_j$ and $\omega_j$ onto the PSD cone (drop round-off negative eigenvalues), so $Y_j,\Omega_j\succeq 0$.
2. Define $\Lambda_j$ by the stationarity equation, so stationarity in $P$ holds exactly.
3. Split $\Lambda_j = \Lambda_j^{+} - \Lambda_j^{-}$ into positive and negative parts. Move $\Lambda_j^{-}$ into the upper-box multiplier: $\Omega_j' = \Omega_j + \Lambda_j^{-}$, $\Lambda_j' = \Lambda_j^{+}\succeq 0$. Stationarity still holds.
4. Rescale every multiplier by $c = \sum_j \operatorname{tr}\Omega_j'$ so that $\sum_j\operatorname{tr}\Omega_j = 1$ holds exactly.

This gives the bound printed as `certified LB`:

```math
\gamma_{\rm LB} \;=\; \frac{\sum_j \operatorname{tr}\Lambda_j^{+}}{\sum_j \operatorname{tr}\Omega_j + \sum_j\operatorname{tr}\Lambda_j^{-}} \;\le\; \gamma^\star .
```

It is valid in every round, not only at convergence. While directions are still missing it is weak, because $\Lambda^{-}$ is large. It rises to meet $\gamma_{\rm UB}$ as pricing removes the negative eigenvalues.

---

## 6. Starting directions (warm start)

The initial $U_j$ come from a backward Lyapunov recursion that always gives a feasible $P$:

```math
\Big(\tfrac{1}{2\Delta t}I - A_j^{\mathsf H}\Big)X_j + X_j\Big(\tfrac{1}{2\Delta t}I - A_j\Big) = \frac{X_{j+1}}{\Delta t} + H_j^{+},\qquad X_K = 0,
```

where $H_j^{+}$ is the positive part of $H_j$. With $P_j = I + X_j$, this rearranges to $M_j = H_j - H_j^{+} = -H_j^{-} \preceq 0$, so the warm start is feasible for any $\Delta t$. It is loose ($\gamma = 206$ at $N=12$, versus the optimum $85.66$). Each $U_j$ starts as the top `r0` (default 3) eigenvectors of $X_j$.

---

## 7. Final independent check

After the last round, the code rebuilds every full $n\times n$ matrix $P_j = I + U_jS_jU_j^{\mathsf H}$ and checks the original constraints directly:

- $\lambda_{\min}(P_j) \ge 1$ and $\lambda_{\max}(P_j)\le\gamma$ for all $j$;
- $\lambda_{\max}(M_j) \le 0$ for all $j$ (printed as `max eig M_j`).

This check does not rely on the Schur reduction or on the duals.

---

## 8. Usage

Requirements: MATLAB (R2016b or newer), [YALMIP](https://yalmip.github.io), and MOSEK.

```matlab
addpath(genpath('path\to\YALMIP-master'));
addpath(genpath('path\to\Mosek'));

out = colgen_lmi(12);                              % t0 = 0, T = 30, dt = 1 (defaults)
out = colgen_lmi(20, struct('maxit', 20));         % cap the number of rounds
out = colgen_lmi(24, struct('t0', 20, 'T', 200));  % the paper's time window
```

Options (fields of the second argument): `kx, kz, Re, kappa, t0, T, dt, NN` (physics, same meaning as the baseline); `r0` (starting directions per step, 3); `addmax` (directions added per step per round, 2); `tol` (relative gap to stop, 1e-5); `maxit` (40); `solver` ('mosek'); `verbose` (0).

`out` contains `gamma`, `LB`, `gap`, `maxeigM`, the full matrices `P`, the directions `U`, and a per-round history `hist`.

### Reading the output

The first line is from the $N=12$ run. The timing line shows the format only; its numbers are illustrative.

```
it  6: gamma = 85.662450 [valid UB, max slack -5.7e-10] | certified LB = 85.662013 (min eig Lambda -1.9e-07) | gap 5.10e-06 | ranks 3-10 | +0 dirs | 14s
       time this round: build 0.4s | YALMIP 0.2s | solver 0.5s | pricing 0.0s | largest reduced LMI 18x18 (n = 20)
```

| Field | Meaning |
|---|---|
| `gamma` | Optimal $\gamma$ of the restricted SDP. A valid upper bound once tagged `valid UB`. |
| `max slack` | Largest $s_j$. About 0 (to solver tolerance) means every original constraint is met. |
| `certified LB` | The rigorous lower bound of §5.4. |
| `min eig Lambda` | Most negative eigenvalue over all $\Lambda_j$. It reaches about 0 at the optimum. |
| `gap` | $(\gamma_{\rm UB}-\gamma_{\rm LB})/\gamma_{\rm UB}$. |
| `ranks a-b` | Smallest and largest number of directions $r_j$. |
| `+k dirs` | Directions added by pricing this round. |
| `time this round` | Where the time went. |

---

## 9. Results so far

Settings: $k_x = 1.2$, $k_z=0$, $Re = 500$, $\kappa = 0.1$, decelerating flow, $t_0 = 0$, $T=30$, $\Delta t=1$ ($K=30$), unless noted.

| $N$ | $n$ | Direct full-LMI solve | Column generation $\gamma_{\rm UB}$ | Certified $\gamma_{\rm LB}$ | Rounds | Max $r_j$ | Time |
|---|---|---|---|---|---|---|---|
| 6 ($T=8$) | 8 | 30.557725 (Clarabel) | 30.557742 | 30.557689 | 2 | 4 | 12 s (SeDuMi) |
| 12 | 20 | 85.6635 (MOSEK) | **85.662450** | **85.662013** | 6 | 10 | 13.8 s (MOSEK) |
| 20 | 36 | — | 32.17 (by round 5–8) | not yet converged | 8+ | 17+ | minutes (MOSEK) |

At $N=12$, the true optimum is pinned to $[85.66201,\,85.66245]$, so $\gamma^\star = 85.66$ to 4 significant figures. The direct MOSEK value sits about $10^{-5}$ above this interval, which is within MOSEK's stopping tolerance.

### Known limitation (open problem)

At $N = 20$, the upper bound settles within a few rounds, but the certificate needs many more directions. $r_j$ grew to 17–18 out of $n = 36$, and once that happens the reduced LMIs are close to full size. Earlier experiments at $N=10,12$ (with a small tie-breaker) found optimal $P_j$ that need only about 4 significant directions per step. That suggests a good $P$ is cheap to find, while the **dual certificate** is the expensive part. Next steps:

- benchmark accuracy and time against a direct solve over $N = 10, 12, 16, 20, 24, 32, 48, \dots$;
- use the per-round timing to separate solver time from YALMIP overhead, and call MOSEK directly if YALMIP dominates;
- measure how the number of directions needed grows with $N$;
- run the paper's window ($t_0=20$, $T=200$) at $M = 24, 32, 48$ against the published values 121,847.19 / 140,027.54 / 160,484.37.

---

## 10. Files

| File | Purpose |
|---|---|
| `colgen_lmi.m` | The solver: builds $A_j$ exactly as the baseline, runs the warm start, the column-generation rounds and the final full check. |

## Reference

C. Wei, Y. Zhao, C. Liu, time-sampled Lyapunov LMI for transient growth in unsteady flows, ACC 2026 (Eq. 24).
