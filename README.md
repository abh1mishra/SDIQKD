# SDIQKD
Code for SDI-QKD project on restricted information assumption

---

# `RI_bff_adv.jl`: entropy bounds under the restricted-information assumption

This section documents `two_meas/RI_bff_adv.jl` and `four_meas/RI_bff_adv.jl`. It assumes familiarity with PCPOP.jl (`@pcmonoid`, `@comms`, `Projector`, `basis_gen`, `npa_dual`, `model_new_obj`). It explains how each physical object and constraint of the paper's RI optimization maps to a Julia variable and a PCPOP call.

All six public functions in the two files run **the same pipeline with the same variable names**. They differ only in a few protocol constants (number of states, measurements and key bases) and in which physical parameter is swept. The pipeline is therefore described once, followed by a table of the per-function constants.

**Notation.** ρₓ = Alice's state for input x, M_{b|y} = Bob's projector for outcome b of measurement y, P_{a,k} = Eve's projector for key value a at grid node k, σ = auxiliary operator, Tr = trace, 𝟙 = identity, ⊗ = tensor product.

## Contents

1. [Physical problem](#1-physical-problem)
2. [Functions and protocol constants](#2-functions-and-protocol-constants)
3. [Pipeline (flow chart)](#3-pipeline)
4. [Stage-by-stage walkthrough](#4-stage-by-stage-walkthrough)
5. [Variable glossary](#5-variable-glossary)
6. [Constraint → PCPOP mapping](#6-constraint--pcpop-mapping)
7. [State and measurement conventions](#7-state-and-measurement-conventions)
8. [Output and usage](#8-output-and-usage)

---

## 1. Physical problem

Alice prepares one of `nstates` states ρₓ with uniform prior pₓ = 1/nstates. Bob performs one of `nmeas` binary measurements {M_{b|y}} with detection efficiency η. The honest states have visibility v.

The device is characterized only by the **restricted-information (RI)** assumption: an upper bound G on the probability of guessing x from the prepared state,

```text
P_g^A = max over POVMs {Nₓ} of  Σₓ pₓ Tr(ρₓ Nₓ)  ≤  G
```

The key is extracted from Alice's bit a in one or more *key bases*. Each key basis is a pair of states (ρ_{x1}, ρ_{x2}) encoding a = 1, 2, measured by Bob with a fixed setting y. The functions return a lower bound on H(A|E) computed with the BFF-type relaxation

```text
H(A|E)  ≥  (1/ln 2) · [ Tr(𝟙_A ⊗ ρ_E − ρ_AE)  +  Σ_k  inf Tr( P_k (α_k ρ_AE + β_k 𝟙_A ⊗ ρ_E) ) ]
```

In the tracial formulation, each term k is the non-commutative polynomial optimization

```text
minimize over ρₓ, M_{b|y}, P_{a,k}, σ :   Tr F_k

subject to
    ρₓ − ρₓ² ≥ 0,   Tr ρₓ = 1                      (valid, possibly mixed, states)
    σ − pₓ ρₓ ≥ 0,  Tr σ ≤ G                        (RI:  P_g^A ≤ Tr σ ≤ G)
    M_{b|y}² = M_{b|y},  Σ_b M_{b|y} = 𝟙            (projective measurements)
    Tr(ρₓ M_{b|y}) = p(b|x,y)                       (observed statistics)
    P_{a,k}² = P_{a,k},  [P_{a,k}, M_{b|y}] = 0     (Eve's BFF projectors)
```

This is relaxed to an SDP by `PCPOP.npa_dual(...; tracial=true)`.

The key rate reported in the plots is r = H(A|E) − H(A|B). The functions compute and store only H(A|E). They compute H(A|B) only to decide when to stop the sweep.

---

## 2. Functions and protocol constants

| File | Function | Protocol (states, meas.) | Swept | Fixed |
|---|---|---|---|---|
| `two_meas/RI_bff_adv.jl` | `cond_entropy_four_local_eta` | (4,2) | η | `G`, `v` |
| `two_meas/RI_bff_adv.jl` | `cond_entropy_four_local_visibility` | (4,2) | v | `G`, `η` |
| `four_meas/RI_bff_adv.jl` | `cond_entropy_four_local_eta` | (4,4) | η | `G`, `v` |
| `four_meas/RI_bff_adv.jl` | `cond_entropy_four_local_visibility` | (4,4) | v | `G`, `η` |
| `four_meas/RI_bff_adv.jl` | `cond_entropy_six_local_eta` | (6,4) | η | `G`, `v` |
| `four_meas/RI_bff_adv.jl` | `cond_entropy_six_local_visibility` | (6,4) | v | `G`, `η` |

The only places where the functions differ:

| Constant | (4,2) `two_meas` | (4,4) `four_meas`, `four_*` | (6,4) `four_meas`, `six_*` |
|---|---|---|---|
| `nstates` | 4 | 4 | 6 |
| `nmeas` | 2 | 4 | 4 |
| key bases n_b (`nbasis`) | 1 (implicit) | 2 | 3 |
| Eve's projectors `E` | `E[2,0]` | `E[2*nbasis,0]` = 4 | `E[2*nbasis,0]` = 6 |
| Key-basis pairs → Eve projectors | (ρ₁,ρ₂)→`E[1:2]` | (ρ₁,ρ₂)→`E[1:2]`, (ρ₃,ρ₄)→`E[3:4]` | (ρ₁,ρ₄)→`E[1:2]`, (ρ₂,ρ₅)→`E[3:4]`, (ρ₃,ρ₆)→`E[5:6]` |
| Honest statistics | `prob_four` | `prob_four` | `prob_six` |
| H(A\|B) | `conditional_entropy_four_A` | `conditional_entropy_four_A` (avg. over 2 bases) | `conditional_entropy_six_A` (avg. over 3 bases) |
| Entropy formula | `(1+obj_val)/log(2)` | `(2+obj_val)/(2*log(2))` | `(3+obj_val)/(3*log(2))` |
| Stop tolerance | 1e-6 | 1e-5 (η), 1e-6 (v) | 1e-5 (η), 1e-3 (v) |
| `pure` keyword | not available (always mixed) | available, default `false` | available, default `false` |

Shared helpers (defined at the top of each file):

- `obj_i_term_A(ρ, Pi, αi, βi)`: the BFF objective for **one** key basis and **one** grid node (see [Stage 8](#stage-8-objective-polynomials)).
- `obj_i_term_ua_A(...)` (`four_meas` only): the non-adaptive objective. It is not called by the six functions above.

External dependencies: `grid_points` and `grid_to_coeffs` in `utils.jl`, and the honest-model functions in `two_meas_corr.jl` / `four_meas_corr.jl`.

---

## 3. Pipeline

```mermaid
flowchart TD
    S0["Inputs: G, (v or η), level, step_size, grid options, pure, fname"] --> S1

    subgraph SETUP["Built once per call"]
      S1["1 · Protocol constants<br/><code>nstates, nmeas, nbasis</code>"]
      S2["2 · BFF grid<br/><code>points = grid_points(step_size; …)</code><br/><code>α, β = grid_to_coeffs(points)</code>"]
      S3["3 · Operator algebra<br/><code>@pcmonoid M B[nmeas,0] E[·,0] BE[nstates+1,0]</code><br/><code>@comms B E</code> · <code>Projector.([B;E])</code> · <code>build(M)</code>"]
      S4["4 · Physical operators<br/><code>PB, ρ = BE[1:nstates], σ = BE[end]</code>"]
      S5["5 · Fixed constraints<br/><code>op_ge</code>: σ − ρₓ/nstates ≥ 0, ρₓ − ρₓ² ≥ 0<br/><code>tr_ge</code>: Tr σ ≤ G"]
      S1 --> S2 --> S3 --> S4 --> S5
    end

    S5 --> S6

    subgraph SWEEP["Per sweep point i  (plx = 1.00, 0.99, …)"]
      S6["6 · Statistics constraints<br/><code>tr_eq</code>: Tr(ρₓ M₁|y) = p(1|x,y),  Tr ρₓ = 1"]
      S7["7 · Reference entropy<br/><code>cond_entropy</code> = H(A|B)"]
      S8["8 · Objective list<br/><code>objs[k]</code>, k = 1…tot_points"]
      S9["9 · Relaxation<br/><code>basis_gen</code> → <code>ops, ops_principal</code><br/><code>npa_dual(…; tracial=true, change_objective=true)</code>"]
      S10["10 · Solve per node k<br/><code>S ← S + old_obj − objs[k]</code><br/><code>model_new_obj</code> → <code>optimize!</code><br/><code>obj_val += objective_value</code>"]
      S11["11 · H(A|E) bound<br/><code>ply[i] = (n_b + obj_val)/(n_b·ln 2)</code>"]
      S12["12 · Append to file"]
      S6 --> S7 --> S8 --> S9 --> S10 --> S11 --> S12
    end

    S12 --> D{"ply[i] − H(A|B) &lt; tol ?"}
    D -- "no: next sweep point" --> S6
    D -- "yes: key rate ≤ 0" --> R["return plx, ply"]
```

---

## 4. Stage-by-stage walkthrough

The code excerpts are taken from the `*_visibility` functions of `four_meas`. The other functions are identical up to the constants in [§2](#2-functions-and-protocol-constants).

### Stage 1: Protocol constants

```julia
nstates = 4      # number of Alice's preparations x
nmeas   = 4      # number of Bob's binary measurements y
nbasis  = 2      # number of key bases (four_meas only; two_meas has one)
```

### Stage 2: BFF grid

```julia
points = grid_points(step_size; start=start_grid, stop=stop_grid, uniform=uniform)
α, β   = grid_to_coeffs(points)
tot_points = length(points)
```

- `points` are the quadrature nodes t_k ∈ [0, 1] of the entropy relaxation. `uniform=true` gives an equally spaced grid. `uniform=false` gives nodes spaced by √(ε·t), with `step_size` = ε.
- `α[k]`, `β[k]` are the corresponding weights α_k, β_k. `grid_to_coeffs` assumes `start_grid = 0` and `stop_grid = 1`.
- `tot_points` is the number of SDPs solved per sweep point. It controls both the tightness of the bound and the run time.

### Stage 3: Operator algebra

```julia
@pcmonoid M B[nmeas,0] E[2*nbasis,0] BE[nstates+1,0]
@comms B E
Projector.([B;E])
if pure
    Projector.(BE[1:nstates])
end
build(M)
```

| PCPOP call | Physical meaning |
|---|---|
| `B[nmeas,0]` | `nmeas` Hermitian letters. `B[y]` is Bob's outcome-1 projector M_{1\|y}. |
| `E[2*nbasis,0]` | Eve's BFF projectors P⁽ʳ⁾_{a,k}: two per key basis r (one per key value a). The same letters are reused for every grid node k, which is valid because each k is a separate SDP. |
| `BE[nstates+1,0]` | Letters that commute with neither `B` nor `E`: the states ρₓ and the auxiliary operator σ. The name `BE` is only a group label. |
| `@comms B E` | [P_{a,k}, M_{b\|y}] = 0. Eve's and Bob's operators act on different subsystems. |
| `Projector.([B;E])` | M_{b\|y}² = M_{b\|y}, P_{a,k}² = P_{a,k} (encoded as `:Projector` multiplication type in the graph-product monoid). |
| `Projector.(BE[1:nstates])` | Optional (`pure=true`, four_meas only): pure states ρₓ² = ρₓ. |
| `build(M)` | Freezes the monoid. No further commutation or projector rules can be added after this. |

No relation is imposed on σ. It is a generic Hermitian letter, constrained only through `op_ge` and `tr_ge`.

### Stage 4: Physical operators

```julia
PB = [B[1] B[2] B[3] B[4]; 1-B[1] 1-B[2] 1-B[3] 1-B[4]]   # PB[b,y] = M_{b|y}
ρ  = BE[1:nstates]                                       # ρ[x] = ρ_x
σ  = BE[end]                                             # σ
```

- Each binary measurement is parameterized by its outcome-1 projector. Outcome 2 is `1-B[y]`, so completeness Σ_b M_{b|y} = 𝟙 and orthogonality M_{1|y} M_{2|y} = 0 hold by construction.
- The states are unconstrained operators at this point. They become density operators through the constraints of stages 5 and 6.

### Stage 5: Fixed constraints (independent of the sweep)

```julia
op_ge = [σ - (1/nstates)*ρ[x] for x in 1:nstates]             # σ ≥ p_x ρ_x
tr_ge = [[-σ, -G]]                                            # Tr(−σ) ≥ −G  ⇔  Tr σ ≤ G
if !pure
    op_ge = vcat(op_ge, [ρ[x] - ρ[x]*ρ[x] for x in 1:nstates]) # 0 ≤ ρ_x ≤ 1
end
```

- **`op_ge`** holds operator inequalities q ≥ 0. For each q, `npa_dual` builds one localizing matrix Υ_q with entries Tr(u† q v) for u, v in `ops`, and requires Υ_q ⪰ 0.
  - σ − pₓρₓ ≥ 0 together with Tr σ ≤ G gives, for any POVM {Nₓ}, P_g^A ≤ Σₓ Tr(σ Nₓ) = Tr σ ≤ G. This is how the RI assumption is encoded.
  - ρₓ − ρₓ² ≥ 0 bounds the spectrum of ρₓ to [0, 1]. Together with Tr ρₓ = 1, this is the tracial encoding of a (mixed) state of unspecified dimension.
- **`tr_ge`** holds linear trace constraints `[poly, c]` meaning Tr(poly) ≥ c.

### Stage 6: Sweep and statistics constraints

```julia
plx = [v_start - i/100 for i in 0:100]        # (eta_start in *_eta functions)
for i in 1:length(plx)
    v = plx[i]                                # (η = plx[i] in *_eta functions)
    tr_eq = [[ρ[x]*PB[1,y], prob_four(1, x, y, η; v=v, bin=true)] for x in 1:nstates for y in 1:nmeas]
    tr_eq = vcat(tr_eq, [[ρ[x], 1] for x in 1:nstates])
```

- **`tr_eq`** holds linear trace equalities `[poly, c]` meaning Tr(poly) = c:
  - Tr(ρₓ M_{1|y}) = p(1|x,y) for all x, y. Outcome 2 is implied by normalization.
  - Tr ρₓ = 1.
- `prob_four` / `prob_six` (from `*_corr.jl`) compute the honest statistics p(b|x,y) = Tr(ρₓ⁽ᵛ⁾ M^η_{b|y}), using
  - noisy states ρₓ⁽ᵛ⁾ = v |ψₓ⟩⟨ψₓ| + (1 − v) 𝟙/2, and
  - lossy POVMs with efficiency η.
- **Binning.** `bin=true` assigns Bob's no-click event to outcome 1: M^η_{1|y} = η M_{1|y} + (1 − η) 𝟙. The SDP therefore sees only binary statistics. Coarse-graining can only lower the bound, so it stays valid.
- `npa_dual` is called with `normalize=false`, so no Tr 𝟙 = 1 constraint is added. States are normalized by Tr ρₓ = 1 instead.

### Stage 7: Reference entropy

```julia
cond_entropy = conditional_entropy_four_A(η; v=v, bin=false)
```

H(A|B) of the honest model restricted to the key basis/bases, with **unbinned** (three-outcome) statistics because Bob does observe the no-click event. For multiple key bases it is the average over bases. It is used only for the stopping rule in Stage 12.

### Stage 8: Objective polynomials

```julia
function obj_i_term_A(ρ, Pi, αi, βi)
    res_αi = Pi[1]*ρ[1] + Pi[2]*ρ[2]
    res_βi = (Pi[1] + Pi[2])*(ρ[1] + ρ[2])
    return -0.5*(αi*res_αi + βi*res_βi)
end

objs1 = [obj_i_term_A(ρ[1:2], E[1:2], α[k], β[k]) for k in 1:tot_points]   # key basis 1
objs2 = [obj_i_term_A(ρ[3:4], E[3:4], α[k], β[k]) for k in 1:tot_points]   # key basis 2
objs  = objs1 + objs2                                                      # element-wise
```

**Derivation.** Take a key basis with states (ρ_{x1}, ρ_{x2}) encoding a = 1, 2 with p_a = ½. Then

```text
ρ_AE = ½ Σ_a |a⟩⟨a| ⊗ ρ_{x_a}          ρ_E = ½ (ρ_{x1} + ρ_{x2})          P_k = Σ_a |a⟩⟨a| ⊗ P_{a,k}
```

and the BFF term becomes

```text
Tr( P_k (α_k ρ_AE + β_k 𝟙_A ⊗ ρ_E) )
    = ½ [ α_k · Σ_a Tr(P_{a,k} ρ_{x_a})  +  β_k · Tr( (P_{1,k} + P_{2,k}) (ρ_{x1} + ρ_{x2}) ) ]
```

`res_αi` and `res_βi` are exactly the two traced polynomials: `Pi` = (P_{1,k}, P_{2,k}) and `ρ` = (ρ_{x1}, ρ_{x2}). The prefactor `-0.5` is p_a = ½ with the sign convention of the weights returned by `grid_to_coeffs`.

**Several key bases.** `objs1 + objs2 (+ objs3)` is element-wise vector addition. `objs[k]` is therefore the node-k objective **summed over key bases**. It shares ρₓ, M_{b|y} and σ across bases, but each basis has its own disjoint Eve projectors. One SDP per node thus bounds all bases jointly.

### Stage 9: Relaxation

```julia
ops, ops_principal = basis_gen(objs[1], level, [], op_ge, tr_eq, tr_ge, M.vertices, -1)
model, S, V, mons, LMI = npa_dual(0, ops, ops_principal;
        op_ge=op_ge, tr_eq=tr_eq, tr_ge=tr_ge,
        min=true, tracial=true, normalize=false, change_objective=true, progress=true)
```

| Argument / return | Meaning here |
|---|---|
| `level` (default `"1+B*E"`) | Monomial set of the relaxation: the identity, all single letters (`B`, `E`, `ρ`, `σ`) and all products `B[y]*E[j]`. |
| `[]` (`op_eq`) | No operator equalities. Projectivity and commutation are enforced by the monoid itself. |
| `M.vertices` | All letters of the monoid, used to generate the monomials of `level`. |
| `-1` (`lvl_lm`) | Automatic localizing level: `ops` is the monomial set at `level`, and `ops_principal` is `ops` extended by products with the letters appearing in `op_ge` (degree ⌈deg q / 2⌉ = 1). This makes each localizing matrix a principal submatrix of the moment matrix. |
| `ops_principal` | Index set of the moment matrix Γ, with entries Tr(u† v). |
| `ops` | Index set of each localizing matrix Υ_q, one per element of `op_ge`. |
| `obj = 0` | The model is built without an objective. The objective is inserted in Stage 10. |
| `min=true` | Minimization. Internally `s = -1`, and the dual SDP is a maximization whose value lower-bounds the infimum. |
| `tracial=true` | Cyclic moments: Tr(uv) = Tr(vu) identifies moment entries. |
| `change_objective=true` | Also returns `S`, `V`, `mons` and `LMI`, so the objective can be swapped without rebuilding. |
| `S` | Dual polynomial identity, `−obj + Σ Z_eq·tr_eq − s·Σ Z_ge·tr_ge`, matched against the LMI blocks. |
| `V` | Constraint references generated from `S` (the objective-dependent part of the model). |
| `LMI`, `mons` | Moment/localizing blocks and the set of reduced trace monomials. |

The relaxation is rebuilt at every sweep point because `tr_eq` changes with η or v. It is **not** rebuilt per grid node.

### Stage 10: Per-node solves

```julia
obj_val = 0
old_obj = 0
for obj in objs
    S = S + old_obj - obj                    # replace −old_obj by −obj in the dual identity
    model, V = model_new_obj(model, S, V, mons, LMI, -1)
    set_optimizer(model, optimizer)
    optimize!(model)
    obj_val += objective_value(model)        # ≈ inf Tr F_k
    old_obj = obj
end
```

- `model_new_obj` deletes the constraints `V` and regenerates them from the updated `S`. Only the objective changes between nodes.
- Each node is solved independently, and the values are summed: Σ_k inf Tr F_k ≤ inf Σ_k Tr F_k. This is the "localized objective" referred to in the docstrings. It is still a valid lower bound and keeps each SDP small.

### Stage 11: Entropy bound

```julia
ply[i] = (2+obj_val)/(2*log(2))   # written with the literal n_b: (1+·)/ln2, (2+·)/(2ln2), (3+·)/(3ln2)
```

For each key basis, Tr(𝟙_A ⊗ ρ_E − ρ_AE) = 2·1 − 1 = 1, so the constant term summed over bases is n_b. Dividing by n_b averages over the key bases. `ply[i]` is the bound on H(A|E) in bits.

### Stage 12: Output and stopping rule

```julia
fpath = fname * "v=$(round(v;digits=2))/4Alice_$(level)_G=$(round(G;digits=2)).txt"   # "eta=…" in *_visibility
mkpath(dirname(fpath))
open(fpath, "a") do file
    write(file, "$(plx[i]) $(ply[i])\n")
end
if ply[i] - cond_entropy < tol
    break
end
```

Each sweep point appends one line `x  H(A|E)` to `fpath`. The sweep stops once H(A|E) − H(A|B) falls below `tol`, i.e. once the key rate is no longer positive. The functions return `plx, ply`. Entries of `ply` after the break keep their initial value `1.0` and should be ignored.

---

## 5. Variable glossary

Every function uses these names with the same meaning.

**Inputs**

| Variable | Symbol | Meaning |
|---|---|---|
| `G` | G | Upper bound on Alice's guessing probability P_g^A (RI parameter). G = 1/nstates is the fully restricted case. |
| `v` | v | Visibility of the honest states: ρₓ⁽ᵛ⁾ = v \|ψₓ⟩⟨ψₓ\| + (1 − v) 𝟙/2. |
| `η` | η | Bob's detection efficiency. |
| `level` | — | Relaxation level string passed to `basis_gen`. |
| `step_size`, `start_grid`, `stop_grid`, `uniform` | {t_k} | BFF grid parameters. |
| `eta_start` / `v_start` | — | First value of the sweep (decreasing in steps of 0.01). |
| `pure` | — | `true`: states are projectors. `false`: mixed states via ρₓ − ρₓ² ≥ 0. |
| `fname` | — | Output directory prefix. |
| `optimizer` | — | JuMP optimizer (Mosek by default). |

**Internal**

| Variable | Symbol | Meaning |
|---|---|---|
| `nstates`, `nmeas`, `nbasis` | \|X\|, \|Y\|, n_b | Protocol constants. |
| `points` | t_k | Grid nodes. |
| `α`, `β` | α_k, β_k | BFF weights. |
| `tot_points` | — | Number of grid nodes, i.e. SDPs per sweep point. |
| `M` | — | PCPOP graph-product monoid holding all letters. |
| `B[y]`, `PB[1,y]` | M_{1\|y} | Bob's outcome-1 projector. |
| `1-B[y]`, `PB[2,y]` | M_{2\|y} | Bob's outcome-2 projector. |
| `E[2r-1]`, `E[2r]` | P⁽ʳ⁾_{1,k}, P⁽ʳ⁾_{2,k} | Eve's BFF projectors for key basis r. |
| `ρ[x]` = `BE[x]` | ρₓ | Alice's prepared state (dimension-free). |
| `σ` = `BE[end]` | σ | Auxiliary operator certifying P_g^A ≤ Tr σ. |
| `op_ge` | q ≥ 0 | Operator inequalities → localizing matrices. |
| `tr_ge` | Tr(·) ≥ c | Linear trace inequalities. |
| `tr_eq` | Tr(·) = c | Linear trace equalities (statistics and normalization). |
| `plx` | η or v | Sweep values. |
| `cond_entropy` | H(A\|B) | Honest error-correction cost (unbinned). |
| `objs`, `objs1..3` | F_k | Objective polynomials per node (per key basis before summation). |
| `ops`, `ops_principal` | — | Monomial index sets of the localizing and moment matrices. |
| `model`, `S`, `V`, `mons`, `LMI` | Γ, Υ_q | Dual SDP and its objective-swap handles. |
| `obj_val` | Σ_k inf Tr F_k | Accumulated optimal values. |
| `ply` | H(A\|E) | Lower bound in bits for each sweep point. |
| `fpath` | — | Output file. |

---

## 6. Constraint → PCPOP mapping

| Physical constraint | Where it lives | Code |
|---|---|---|
| M_{b\|y}² = M_{b\|y} | monoid rule | `Projector.(B)` |
| Σ_b M_{b\|y} = 𝟙, M_{1\|y} M_{2\|y} = 0 | parameterization | `PB[2,y] = 1-B[y]` |
| P_{a,k}² = P_{a,k} | monoid rule | `Projector.(E)` |
| [P_{a,k}, M_{b\|y}] = 0 | monoid rule | `@comms B E` |
| ρₓ² = ρₓ (pure only) | monoid rule | `Projector.(BE[1:nstates])` |
| ρₓ − ρₓ² ≥ 0 (mixed) | localizing matrix | `op_ge` |
| σ − pₓ ρₓ ≥ 0 | localizing matrix | `op_ge` |
| Tr σ ≤ G | linear constraint | `tr_ge = [[-σ,-G]]` |
| Tr ρₓ = 1 | linear constraint | `tr_eq` |
| Tr(ρₓ M_{1\|y}) = p(1\|x,y) | linear constraint | `tr_eq` |
| Cyclicity Tr(uv) = Tr(vu) | moment identification | `npa_dual(…; tracial=true)` |
| Moment matrix Γ ⪰ 0 | PSD block | `ops_principal` |

---

## 7. State and measurement conventions

Defined in `two_meas/two_meas_corr.jl` and `four_meas/four_meas_corr.jl`.

| Protocol | States `x` | Measurements `y` | Key (states; setting) |
|---|---|---|---|
| (4,2) | 1 → \|0⟩, 2 → \|1⟩, 3 → \|+⟩, 4 → \|−⟩ | 1 → Z, 2 → π/8 | (1,2; y=1) |
| (4,4) | as (4,2) | 1 → Z, 2 → +π/8, 3 → −π/8, 4 → X | (1,2; y=1), (3,4; y=4) |
| (6,4) | 1, 2, 3 → \|0⟩, \|+⟩, \|π/8⟩; 4, 5, 6 → orthogonal partners | 1 → Z, 2 → X, 3 → +π/8, 4 → −π/8 | (1,4; y=1), (2,5; y=2), (3,6; y=3) |

The index of the X measurement differs between `povm_four` (y = 4) and `povm_six` (y = 2).

---

## 8. Output and usage

Output files contain two columns, the swept parameter and the lower bound on H(A|E) in bits:

```text
1.0  0.9999264850777991
0.99 0.9216597639180786
...
```

Default locations (relative to the script directory):

| File | `*_eta` | `*_visibility` |
|---|---|---|
| `two_meas` | `plots/Guess_prob/von_neumann/eta/v=<v>/4Alice_<level>_G=<G>.txt` | `plots/Guess_prob/von_neumann/visibility/eta=<η>/4Alice_<level>_G=<G>.txt` |
| `four_meas` (4,4) | `plots/Guess_prob/von_neumann/eta/adapt/v=<v>/4Alice_…` | `plots/Guess_prob/von_neumann/visibility/adapt/eta=<η>/4Alice_…` |
| `four_meas` (6,4) | `plots/Guess_prob/von_neumann/eta/adapt/v=<v>/6Alice_…` | `plots/Guess_prob/von_neumann/visibility/adapt/eta=<η>/6Alice_…` |

Files are opened in append mode, so re-running a function adds lines to an existing file.

Example:

```julia
cd("four_meas")
include("RI_bff_adv.jl")

# H(A|E) versus visibility for the (4,4) protocol at G = 0.5, η = 1
plx, ply = cond_entropy_four_local_visibility(0.5, 1.0)

# H(A|E) versus η for the (6,4) protocol at G = 0.34, v = 1, finer BFF grid
plx, ply = cond_entropy_six_local_eta(0.34, 1.0; step_size=0.05)
```
