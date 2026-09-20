# 2D Linear Elasticity FEM and SIMP Topology Optimization

A complete continuum-mechanics-to-design study, in two parts. **Part I** builds and independently
verifies a 2D linear-elasticity finite-element solver (plane stress/strain, manufactured-solution
convergence, physical cantilever baseline with exact scaling-law checks). **Part II** builds SIMP
(Solid Isotropic Material with Penalization) topology optimization on top of that verified solver: a
derived and Taylor-tested compliance sensitivity, a density filter used inside the optimization loop
(not as cosmetic post-processing), an optimality-criteria optimizer, and studies of mesh resolution,
volume fraction, penalization exponent, and filter radius.

This is a classical computational-mechanics/optimization project: no machine learning, no
physics-informed neural networks, no neural operators, no reduced-order models, and no Bayesian
methods are used anywhere in this repository (a deliberate scope boundary for this project).

## Overview

The methodological arc is **continuum mechanics -> variational (weak) formulation -> finite-element
discretization -> verified forward solver -> sensitivity analysis -> PDE-constrained optimization ->
topology optimization**, with every stage checked before being trusted for the next: the elasticity
solver is verified against a manufactured solution and physical scaling laws *before* it is used inside
an optimization loop, and the compliance sensitivity is verified against a finite-difference
(Taylor-remainder) test *before* it is trusted to drive an optimizer. This mirrors the verification
discipline used in the companion `fem-cylinder-flow` and `darcy-inverse-problem` projects.

# Part I -- Linear Elasticity FEM

## Mathematical formulation

**Plane stress vs. plane strain.** Both are 2D idealizations of 3D linear elasticity. Plane stress
assumes the out-of-plane stress $\sigma_{zz}=0$ (a thin structure, loaded in its own plane, free to
contract/expand through its thickness); plane strain assumes the out-of-plane strain
$\varepsilon_{zz}=0$ (a structure very long/constrained out of plane, e.g. a dam cross-section).
**This project uses plane stress throughout**, the standard idealization for topology optimization of
thin structural components -- the classical SIMP cantilever/MBB-beam benchmarks (Sigmund's 99-line and
Andreassen et al.'s 88-line MATLAB codes) model a thin plate loaded in-plane.

With Young's modulus $E$ and Poisson ratio $\nu$, the effective Lame parameters are

$$\text{plane stress: } \lambda^* = \frac{E\nu}{1-\nu^2}, \qquad \text{plane strain: } \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}, \qquad \mu = \frac{E}{2(1+\nu)}\ \text{(both cases)}.$$

The constitutive law and small-strain kinematics are

$$\varepsilon(u) = \tfrac12(\nabla u + \nabla u^T), \qquad \sigma(u) = 2\mu\,\varepsilon(u) + \lambda\,\mathrm{tr}(\varepsilon(u))\,I.$$

## Weak formulation

For equilibrium $-\nabla\cdot\sigma(u) = f$ in $\Omega$, $u=0$ on the clamped boundary $\Gamma_D$, and
$\sigma(u)\cdot n = t$ on $\Gamma_N$: multiply by a vector test function $v \in [H^1_0(\Omega)]^2$ and
integrate by parts,

$$\int_\Omega \sigma(u):\varepsilon(v)\,dx = \int_\Omega f\cdot v\,dx + \int_{\Gamma_N} t\cdot v\,ds.$$

The bilinear form $a(u,v) = \int_\Omega \sigma(u):\varepsilon(v)\,dx$ satisfies $a(u,u) = 2 U_{\mathrm{strain}}(u)$,
twice the physical elastic strain energy -- used below as an independent (Clapeyron's-theorem) cross-check
on the point-load solves, and, in Part II, as the exact compliance functional. Discretized with
continuous piecewise-linear (P1) VECTOR Lagrange elements on a structured quadrilateral mesh (the
classic SIMP "pixel grid" element convention). A separate piecewise-constant (DG0) scalar space holds
the density field used in Part II.

**Point loads.** A concentrated force is a Dirac distribution, not an $L^2$ function, so it cannot be
written as a UFL surface/volume integral. Following the standard practical treatment used in the
classical SIMP reference codes (Sigmund, 2001), point loads are applied by adding the force directly to
the assembled right-hand-side vector at the nearest mesh node's degrees of freedom
(`src/elasticity.py:nearest_node_dofs`), via low-level PETSc assembly rather than `dolfinx.fem.petsc.LinearProblem`
(the same technique used for the point-sensor observation operator in the companion
`darcy-inverse-problem` project).

## Forward-solver verification: manufactured solution

Before any physical or optimization use, the solver is checked against a manufactured solution on
$[0,1]^2$ (clamped on the entire boundary, both displacement components vanishing on $\partial\Omega$):

$$u_1 = \sin(\pi x_0)\sin(\pi x_1), \qquad u_2 = \sin(2\pi x_0)\sin(\pi x_1), \qquad f = -\nabla\cdot\sigma(u_{\text{exact}}),$$

with $\sigma(u_{\text{exact}})$ and the forcing $f$ built symbolically in UFL (no external computer-algebra
system, no hand-derived algebra). Mesh-refinement study, $E=1$, $\nu=0.3$ (`scripts/verify_elasticity.py`):

| N | h | L2 error | order | H1 error | order |
|---|---|---|---|---|---|
| 4 | 0.2500 | 9.7314e-02 | -- | 1.5297e+00 | -- |
| 8 | 0.1250 | 2.4264e-02 | 2.004 | 7.7330e-01 | 0.984 |
| 16 | 0.0625 | 6.0682e-03 | 1.999 | 3.8773e-01 | 0.996 |
| 32 | 0.0312 | 1.5173e-03 | 2.000 | 1.9400e-01 | 0.999 |
| 64 | 0.0156 | 3.7935e-04 | 2.000 | 9.7018e-02 | 1.000 |

Clean $O(h^2)$ (L2) and $O(h^1)$ (H1) convergence, exactly matching the standard a priori error
estimates for P1 elements on an elliptic system (the L2 rate is one order higher than H1 via the
Aubin-Nitsche duality argument -- e.g. Braess, *Finite Elements*, Ch. II). See
`figures/elasticity_convergence.png`.

## Physical cantilever baseline

A second, independent check, on the actual boundary-value problem used throughout the rest of this
project: a $L=2$, $H=1$ rectangle, clamped on the left edge, a downward point load $(0,-1)$ applied at
the mid-height of the free (right) edge, $E=1$, $\nu=0.3$ (`examples/cantilever_baseline.py`):

- max $|u| = 39.742026$, max von Mises stress $=49.840772$, total strain energy $=19.871013$.
- **Linear-elasticity scaling laws**, checked exactly (not approximately): tip deflection at $E=2$
  (same load) is exactly half the baseline deflection (ratio $=0.500000$); tip deflection at load $=2\times$
  baseline (same $E$) is exactly double (ratio $=2.000000$).
- **Independent energy cross-check**: total strain energy ($19.871013$) equals exactly
  $\tfrac12\times F\times|\text{tip deflection}| = \tfrac12\times1\times39.742026=19.871013$
  (Clapeyron's theorem) -- a cross-check that does not reuse any part of the FEM assembly being
  validated.

See `figures/cantilever_baseline.png` for the displacement-magnitude, von Mises stress, and
strain-energy-density fields: displacement grows smoothly from zero at the clamp to its maximum at the
load point; both stress and strain energy concentrate near the clamped support and the point load, the
physically expected pattern for a cantilever.

# Part II -- SIMP Topology Optimization

## Problem statement

Minimize structural compliance under a volume constraint on a fixed design domain:

$$\min_{\rho}\ c(\rho) = f^T u(\rho) \quad\text{s.t.}\quad K(\rho)\,u(\rho) = f,\quad \frac{1}{|\Omega|}\int_\Omega \rho\,dx \le V_f,\quad 0<\rho_{\min}\le\rho\le1.$$

## SIMP material interpolation

$$E(\rho) = E_{\min} + \rho^p\,(E_0 - E_{\min}),\qquad 0\le\rho\le1$$

(Bendsoe, 1989; Bendsoe & Sigmund, 1999, "Material interpolation schemes in topology optimization",
*Arch. Appl. Mech.* 69). $E_{\min}>0$ (rather than $0$) avoids stiffness-matrix singularity for void
elements while remaining physically negligible; $E_{\min}=10^{-9}E_0$ is the standard default (Sigmund,
2001; Andreassen et al., 2011). $p>1$ penalizes intermediate ("grey") densities relative to the linear
($p=1$) rule of mixtures, pushing the optimizer toward a near-0/1 design; $p=3$ is the standard
default. Since $\lambda(E)$ and $\mu(E)$ are both linear in $E$ at fixed $\nu$,
$\sigma(u;E)=E\,\sigma_{\text{unit}}(u)$, so the bilinear form factors as
$a(u,v;\rho) = \int_\Omega E(\rho)\,\sigma_{\text{unit}}(u):\varepsilon(v)\,dx$ -- the verified Part I
solver (`solve_elasticity`) is reused **unchanged**, with $E(\rho)$ simply passed in as a UFL
expression built from the DG0 density field, rather than a constant.

## Compliance sensitivity (self-adjoint)

Implicit differentiation of the equilibrium constraint $K(\rho)u(\rho)=f$ ($f$ independent of $\rho$)
gives the standard compliance-sensitivity result (Bendsoe & Sigmund, 2003, Sec. 1.3): the "adjoint"
state for compliance is simply $\lambda=-u$, so no extra linear solve is required,

$$\frac{dc}{d\rho_e} = -u^T\frac{dK}{d\rho_e}u = -p\,\rho_e^{p-1}(E_0-E_{\min})\,w_e,\qquad w_e := u_e^T k^0_e u_e = \int_{\Omega_e}\sigma_{\text{unit}}(u):\varepsilon(u)\,dx,$$

with $k^0_e$ the elemental stiffness built from the UNIT-$E$ constitutive law. $w_e$ is computed by
assembling a DG0-test-function vector $\int_\Omega v_0\,\sigma_{\text{unit}}(u):\varepsilon(u)\,dx$ -- since
the DG0 basis function is the indicator of its own cell, this integral picks out exactly the per-element
value (`src/simp.py:elemental_unit_energy`).

## Sensitivity verification: Taylor-remainder test

Before being trusted for optimization, the analytic gradient is checked against a directional
finite-difference (Taylor-remainder) test, exactly as the adjoint gradient is verified in the companion
`darcy-inverse-problem` project: for a random density field $\rho_0$ and random direction $d$,
$c(\rho_0+td) = c(\rho_0) + t\,(\nabla c\cdot d) + O(t^2)$, so the remainder
$r(t)=|c(\rho_0+td)-c(\rho_0)-t\,\nabla c\cdot d|$ should shrink as $O(t^2)$
(`scripts/verify_sensitivity.py`, mesh $8\times4$, $p=3$, $V_f=0.4$, $r_{\min}=0.1$):

| $t$ | remainder | observed order |
|---|---|---|
| $10^{-1}$ | $1.187941\times10^{1}$ | -- |
| $10^{-2}$ | $1.345566\times10^{-1}$ | 1.946 |
| $10^{-3}$ | $1.363426\times10^{-3}$ | 1.994 |
| $10^{-4}$ | $1.365235\times10^{-5}$ | 1.999 |
| $10^{-5}$ | $1.364828\times10^{-7}$ | 2.000 |

Clean convergence to order 2, confirming the analytic sensitivity before it is used to drive the
optimizer. See `figures/sensitivity_taylor_test_local.png` and the analogous unit test
`tests/test_simp.py::test_compliance_sensitivity_taylor_remainder`.

## Density filtering

Unfiltered SIMP optimization is well known to produce mesh-dependent, checkerboard-patterned designs
(an artefact of the low-order finite-element discretization, not a physical feature -- Sigmund &
Petersson, 1998, "Numerical instabilities in topology optimization", *Struct. Optim.* 16). This project
implements the **density filter** (Bourdin, 2001, "Filters in topology optimization", *Int. J. Numer.
Methods Eng.* 50): the optimization variable $\rho$ is mapped to a "physical" density used in the state
equation and objective via a linear, cone-weighted local average,

$$\rho^{\text{phys}}_e = \frac{\sum_i H_{ei}\rho_i}{\sum_i H_{ei}}, \qquad H_{ei} = \max(0,\,r_{\min}-\lVert x_e-x_i\rVert),$$

with sensitivities propagated through this map by the **exact chain rule**,
$d(\cdot)/d\rho = H^T\big(d(\cdot)/d\rho^{\text{phys}} \oslash \text{rowsum}(H)\big)$. The filter is
applied *inside* the optimization loop on every iteration (it constrains what designs are reachable, and
what the optimizer differentiates through), not as a cosmetic smoothing pass on the final image.
`tests/test_simp.py::test_filter_preserves_uniform_density` and
`test_filter_smooths_checkerboard` check the filter's two defining properties directly.

## Optimizer

A standard optimality-criteria (OC) update with bisection on the volume-constraint Lagrange multiplier
$\lambda$ (Sigmund, 2001; Andreassen et al., 2011):

$$\rho_e^{\text{new}} = \mathrm{clip}\left(\rho_e\sqrt{\max\!\left(0,\,\frac{-dc/d\rho_e}{(dv/d\rho_e)\,\lambda}\right)},\ \rho_e-m,\ \rho_e+m\right),\ \text{further clipped to } [\rho_{\min},1],$$

with move limit $m=0.2$ and $\lambda$ found by bisection so the volume constraint is (near-)active
(`src/simp.py:oc_update`).

## Canonical cantilever SIMP benchmark

Same domain, boundary conditions and load case as the Part I physical baseline (clamped left edge,
point load $(0,-1)$ at $(L,H/2)$), so the optimized structure can be read directly against the solid
baseline. Mesh $60\times30$ (1800 elements), $E_0=1$, $E_{\min}=10^{-9}$, $\nu=0.3$, $p=3$, $V_f=0.4$,
$r_{\min}=0.08$ (`scripts/run_topopt.py`, `configs/local.yaml`):

- Final compliance $=101.391807$, down from $617.855262$ for the uniform-density ($\rho\equiv V_f$)
  starting point.
- Final volume fraction $=0.400263$ (target $0.4$).
- Grayness (discreteness) metric $M_{nd}=\text{mean}(4\rho(1-\rho))=0.317090$ (0 = fully 0/1 design; see
  Discreteness note below).
- Ran the full 150 iterations without the max-design-change criterion dropping below the $10^{-3}$
  tolerance (see Convergence-criterion note below); the compliance history plateaus by roughly
  iteration 20.

See `figures/topopt_canonical_density_local.png` (initial vs. final density) and
`figures/topopt_canonical_history_local.png` (compliance/volume-fraction/max-change vs. iteration). The
optimized topology is a two-bay diagonal-braced truss connecting the clamped edge to the load point --
qualitatively the textbook SIMP cantilever result reported in Sigmund (2001) and Bendsoe & Sigmund
(2003) for this aspect ratio and volume fraction. This visual resemblance is a qualitative sanity check,
not a validation in the same sense as the manufactured-solution/Taylor-remainder checks above: no
independent ground truth is available for the optimum of a non-convex topology-optimization problem.

**Convergence-criterion note.** The compliance objective converges quickly (essentially flat after
~20 iterations) and the volume constraint is satisfied to within $3\times10^{-4}$ throughout, but the
max-elementwise-design-change criterion does not decrease monotonically: it falls to $\approx0.01$
around iteration 90-100 and then rises again to $\approx0.015$-$0.03$ before the run ends at
`max_iter`. This is a documented, known behavior of OC-based SIMP optimization -- a small number of
elements near the 0/1 bounds and filter-transition zones can oscillate ("limit-cycle") between
iterations even once the objective has effectively converged. It is reported honestly here rather than
disguised by loosening the tolerance after the fact; the design and compliance are stable well before
the run terminates, as the history figure shows directly.

## Mesh-resolution study

The physical filter radius $r_{\min}$ is held fixed in *absolute length units* ($r_{\min}=3h$, $h=L/n_x$)
across resolutions -- refining the mesh while holding $r_{\min}$ fixed in element-count units would
silently change the physical minimum-feature length and confound the resolution effect with a filter
effect (Sigmund & Petersson, 1998; Bourdin, 2001). $V_f=0.4$, $p=3$ (`scripts/mesh_study.py`):

| $n_x\times n_y$ | $h$ | $r_{\min}$ | elements | iterations | time (s) | compliance | grayness |
|---|---|---|---|---|---|---|---|
| $30\times15$ | 0.0667 | 0.2000 | 450 | 150 | 3.7 | 175.4809 | 0.6349 |
| $60\times30$ | 0.0333 | 0.1000 | 1800 | 150 | 5.0 | 109.9431 | 0.3816 |
| $90\times45$ | 0.0222 | 0.0667 | 4050 | 150 | 7.5 | 94.3880 | 0.2601 |

**Honest mesh-dependence finding**: the $30\times15$ mesh is under-resolved relative to its own (large,
$r_{\min}=0.2\approx H/5$) filter length and produces a qualitatively different, diffuse blob-like
design (`figures/mesh_study_local.png`, top panel) rather than a resolved truss. The $60\times30$ and
$90\times45$ meshes produce the *same qualitative* two-bay diagonal-truss topology, but compliance is
still decreasing (109.94 -> 94.39, about 14%) and grayness still falling between them -- the design is
**qualitatively mesh-independent but not quantitatively mesh-converged** at these resolutions. This is
reported as a limitation, not glossed over: a rigorous mesh-convergence claim would require pushing to
substantially finer meshes than were run here (see Limitations).

## Volume-fraction study

Mesh $60\times30$, $p=3$, $r_{\min}=0.08$ (`scripts/volume_fraction_study.py`):

| $V_f$ | iterations | compliance | grayness |
|---|---|---|---|
| 0.30 | 150 | 164.7728 | 0.3513 |
| 0.40 | 150 | 101.3918 | 0.3171 |
| 0.50 | 150 | 74.6319 | 0.2821 |

Compliance decreases monotonically with more available material, as expected. See
`figures/volume_fraction_study_local.png`: lower $V_f$ forces thinner, more sparsely braced members.

## Penalization study

Mesh $60\times30$, $V_f=0.4$, $r_{\min}=0.08$ (`scripts/penalization_study.py`):

| $p$ | iterations | compliance | grayness |
|---|---|---|---|
| 1.0 | 150 | 65.3023 | 0.5999 |
| 2.0 | 150 | 88.0593 | 0.3207 |
| 3.0 | 150 | 101.3918 | 0.3171 |

$p=1$ (no penalization, a linear rule-of-mixtures interpolation) achieves the lowest *reported*
compliance but at grayness $\approx0.60$ -- `figures/penalization_study_local.png` shows this is because the
optimizer exploits intermediate ("grey", unpenalized) density to cheaply reduce the linear-interpolation
compliance rather than committing to a discrete structural topology; the $p=1$ design is a diffuse blob,
not a recognizable truss. $p=2$ and $p=3$ both converge to a near-binary, structurally interpretable
truss (grayness $\approx0.32$), at higher (but physically meaningful) compliance. This is exactly the
qualitative effect SIMP penalization is designed to produce (Bendsoe & Sigmund, 2003), demonstrated
quantitatively here rather than only asserted.

## Filter-radius study

Mesh $60\times30$, $V_f=0.4$, $p=3$ (`scripts/filter_radius_study.py`):

| $r_{\min}$ | iterations | compliance | grayness |
|---|---|---|---|
| 0.04 | 150 | 84.5195 | 0.1377 |
| 0.08 | 150 | 101.3918 | 0.3171 |
| 0.16 | 150 | 150.1396 | 0.5603 |

Larger $r_{\min}$ enforces a larger effective minimum member width, at the cost of both higher
compliance (less design freedom) and higher grayness (more blending across the wider filter kernel);
smaller $r_{\min}$ allows thinner, more efficient members but with less enforced manufacturability. See
`figures/filter_radius_study_local.png`.

## Post-hoc von Mises stress evaluation: explicitly NOT stress-constrained optimization

**This section exists to prevent a specific, easy-to-make overclaim.** The canonical benchmark above
minimizes *compliance* under a volume constraint; it was never given any information about stress, and
nothing about its optimization drove stress down or bounded it in any way.
`scripts/postprocess_stress.py` evaluates the von Mises field on the *already-optimized* compliance
design purely as a post-hoc diagnostic:

$$\sigma_{vm} = \sqrt{\sigma_{xx}^2 - \sigma_{xx}\sigma_{yy} + \sigma_{yy}^2 + 3\sigma_{xy}^2}\quad\text{(plane-stress form)}.$$

- Max von Mises stress on solid material ($\rho^{\text{phys}}\ge0.5$): $43.808072$.
- Max von Mises stress everywhere, including near-void elements, reported only for completeness and
  **not physically meaningful** (the $E_{\min}$-scaled "material" in near-void regions is a numerical
  regularization device, not a real material, so stress evaluated there is not interpretable):
  $129.119153$.

`figures/topopt_stress_postprocess_local.png` shows the expected pattern: stress concentrates at the clamped
corners and the point-load node -- a well-known feature of unconstrained-compliance topology
optimization, and precisely the motivation for a genuinely different problem class,
**stress-constrained topology optimization** (Duysinx & Bendsoe, 1998, "Topology optimization of
continuum structures with local stress constraints", *Int. J. Numer. Methods Eng.* 43), which requires
a stress measure (typically a $p$-norm aggregation of the pointwise field, since the true max-stress
constraint is non-differentiable, and the local constraints are famously "singular" at vanishing
density) inside the objective/constraints, with its own sensitivity analysis and materially different
optimizer behavior. **That problem is out of scope for this project.** Nothing in this repository's
figures, results files, or this README describes the compliance-optimized structure as "stress
optimized," "stress minimized," or "stress constrained."

## Discreteness (grayness) metric

$$M_{nd}(\rho^{\text{phys}}) = \frac{1}{n}\sum_e 4\,\rho^{\text{phys}}_e\,(1-\rho^{\text{phys}}_e) \in [0,1],$$

0 for a fully 0/1 ("black and white") design, 1 if every element sits at $\rho=0.5$ (as used e.g. in
Andreassen et al., 2011). Reported alongside compliance throughout the studies above, since compliance
alone does not indicate whether a design is structurally interpretable (the $p=1$ penalization result
is the clearest illustration: low compliance, but a design that is not close to 0/1 and is not a
recognizable structural form).

## Computational performance

All studies above were run on a single core with a direct (MUMPS) LU solve per elasticity state solve
(no warm-starting between SIMP iterations, no multigrid or matrix-free acceleration). On the
$60\times30$ mesh (1800 elements, 3782 displacement DOFs), one iteration (state solve + sensitivity
assembly + filter + OC update) takes $\approx0.033$ s; a 150-iteration run takes $\approx5$ s. On the
finest mesh tested ($150\times75$, 11250 elements, timed separately, not part of the saved study
outputs), one iteration takes $\approx0.25$ s. Runtime scales roughly linearly with element count over
this range, consistent with the direct-solver cost being dominated by matrix assembly rather than the
$O(n^{1.5})$-ish factorization cost at these small-to-moderate problem sizes.

## Limitations

- **Mesh convergence is not established, only qualitative mesh-independence.** The mesh study above
  shows the same qualitative topology at $60\times30$ and $90\times45$ but a still-decreasing
  compliance (14% between the two); no resolution tested here is demonstrated to be asymptotically
  mesh-converged.
- **The design-change convergence criterion does not monotonically decrease** in the canonical
  benchmark (see the Convergence-criterion note above) -- a known OC-method limit-cycle behavior, not
  independently resolved by e.g. a Method-of-Moving-Asymptotes optimizer in this project.
- **No stress constraint.** As emphasized above, the optimized structures are compliance-only designs;
  the post-hoc stress field is reported for context, not as evidence of an implicitly acceptable stress
  state.
- **No manufacturing/connectivity constraints** (minimum member width beyond what the filter radius
  induces, overhang/additive-manufacturing constraints, symmetry enforcement) are imposed.
- **2D, single load case, single material, linear elasticity only.** No multi-load-case robustness, no
  nonlinear (large-deformation or nonlinear-material) analysis, no 3D.
- **The OC optimizer's move limit and bisection tolerance are fixed** (`move=0.2`,
  `bisection_tol=1e-6`) rather than studied; different values were not swept.
- **Near-void ($\rho^{\text{phys}}<0.5$) stress values are reported only for completeness** and are
  explicitly flagged as not physically meaningful (see stress section above) -- this is a known
  limitation of naive SIMP stress post-processing (Le et al., 2010, "Stress-based topology optimization
  for continua", *Struct. Multidiscip. Optim.* 41).

## Reproducibility

Every quantitative figure in this README is written to a plain-text or `.npy` file under `results/` by
the script named alongside it, and every figure under `figures/` is generated by that same script --
nothing here was hand-transcribed or estimated. All optimization runs use a fixed random seed where
randomness is involved (the Taylor-remainder test's random density/direction pair); `run_simp` always
starts from the deterministic uniform-density field $\rho\equiv V_f$. Deterministic reproducibility of
both the state solver and the optimization loop is directly tested
(`tests/test_elasticity.py::test_deterministic_reproducibility`,
`tests/test_simp.py::test_optimization_reproducibility`).

```
conda env create -f environment.yml    # once
conda activate fenicsx

python -m pytest tests/ -v                              # 18 tests, seconds

python scripts/verify_elasticity.py                      # Part I: MMS convergence
python examples/cantilever_baseline.py                    # Part I: physical baseline
python scripts/verify_sensitivity.py --config configs/local.yaml   # Part II: sensitivity check
python scripts/run_topopt.py --config configs/local.yaml           # canonical benchmark
python scripts/mesh_study.py --config configs/local.yaml
python scripts/volume_fraction_study.py --config configs/local.yaml
python scripts/penalization_study.py --config configs/local.yaml
python scripts/filter_radius_study.py --config configs/local.yaml
python scripts/postprocess_stress.py --config configs/local.yaml    # after run_topopt.py
```

`configs/smoke.yaml` runs every script end-to-end in a few seconds each (coarse meshes, few
iterations) for a fast sanity check; `configs/full.yaml` runs at higher resolution (up to
$150\times75$) for more expensive, more resolved results than are embedded in this README.

## Repository structure

```
src/elasticity.py       verified linear-elasticity FEM solver (Part I)
src/manufactured.py     UFL-symbolic manufactured solution
src/postprocess.py      von Mises stress, strain-energy density
src/simp.py             SIMP interpolation, sensitivity, filter, OC optimizer (Part II)
src/viz.py              shared field-plotting helpers
scripts/verify_elasticity.py       Part I manufactured-solution convergence study
scripts/verify_sensitivity.py      Part II sensitivity Taylor-remainder test
scripts/run_topopt.py              canonical cantilever SIMP benchmark
scripts/mesh_study.py              mesh-resolution study
scripts/volume_fraction_study.py   volume-fraction study
scripts/penalization_study.py      penalization-exponent study
scripts/filter_radius_study.py     filter-radius study
scripts/postprocess_stress.py      post-hoc von Mises evaluation on the optimized design
examples/cantilever_baseline.py    Part I physical baseline with scaling-law checks
tests/                             18 unit tests (elasticity, postprocess, SIMP)
configs/{smoke,local,full}.yaml    resolution/iteration presets
results/, figures/                 saved raw results and figures, all script-generated
```

## Installation

```
conda env create -f environment.yml
conda activate fenicsx
```

Requires `dolfinx` 0.10 / PETSc (conda-forge; no pip wheels for macOS/arm64). Uses only
`dolfinx.mesh.create_rectangle` (no external mesh generator needed).

## References

- Bendsoe, M. P. (1989). "Optimal shape design as a material distribution problem." *Structural
  Optimization*, 1, 193-202.
- Bendsoe, M. P., & Sigmund, O. (1999). "Material interpolation schemes in topology optimization."
  *Archive of Applied Mechanics*, 69, 635-654.
- Bendsoe, M. P., & Sigmund, O. (2003). *Topology Optimization: Theory, Methods and Applications*.
  Springer.
- Sigmund, O. (2001). "A 99 line topology optimization code written in Matlab." *Structural and
  Multidisciplinary Optimization*, 21, 120-127.
- Andreassen, E., Clausen, A., Schevenels, M., Lazarov, B. S., & Sigmund, O. (2011). "Efficient
  topology optimization in MATLAB using 88 lines of code." *Structural and Multidisciplinary
  Optimization*, 43, 1-16.
- Sigmund, O., & Petersson, J. (1998). "Numerical instabilities in topology optimization: A survey on
  procedures dealing with checkerboards, mesh-dependencies and local minima." *Structural
  Optimization*, 16, 68-75.
- Bourdin, B. (2001). "Filters in topology optimization." *International Journal for Numerical Methods
  in Engineering*, 50, 2143-2158.
- Duysinx, P., & Bendsoe, M. P. (1998). "Topology optimization of continuum structures with local
  stress constraints." *International Journal for Numerical Methods in Engineering*, 43, 1453-1478.
- Le, C., Norato, J., Bruns, T., Ha, C., & Tortorelli, D. (2010). "Stress-based topology optimization
  for continua." *Structural and Multidisciplinary Optimization*, 41, 605-620.
- Braess, D. (2007). *Finite Elements: Theory, Fast Solvers, and Applications in Solid Mechanics* (3rd
  ed.). Cambridge University Press.
