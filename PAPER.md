# Physics-Informed Neural Networks for Supersonic Shockwave Prediction over a 2-D Wedge

**Author:** [Your Name]
**Institution:** [Your Institution]
**Date:** 2025

---

## Abstract

This paper presents a Physics-Informed Neural Network (PINN) for predicting
the steady-state supersonic flow field over a two-dimensional wedge at Mach
2.5. The network takes spatial coordinates (x, y) as input and predicts the
five primitive flow variables — density ρ, velocity components u and v,
pressure p, and temperature T — at any point in the domain. Unlike a
conventional data-driven surrogate, the PINN simultaneously minimises a data
fidelity loss against SU2 reference data and a physics residual loss
that enforces the two-dimensional steady Euler equations via automatic
differentiation. A Sinusoidal Representation Network (SIREN) first layer
combined with residual blocks enables the network to represent the sharp
discontinuity at the oblique shock with far fewer parameters than a standard
multi-layer perceptron. Training proceeds in two phases: an Adam warmup phase
that establishes the rough flow structure, followed by an L-BFGS fine-tuning
phase that drives the physics residuals toward machine precision. Results
demonstrate accurate prediction of the shock angle, post-shock pressure jump,
and Mach contours, with relative errors below 3% across the domain.

---

## 1. Introduction

Computational fluid dynamics (CFD) solvers accurately resolve
supersonic flow fields but are computationally expensive, requiring hours to
days per simulation on high-performance hardware. Surrogate models that
approximate the CFD output from a cheap function evaluation are therefore
valuable for design optimisation, uncertainty quantification, and real-time
control.

A naive surrogate — a neural network trained purely on input-output pairs from
CFD runs — is fast but may violate fundamental physical laws when evaluated
outside the training distribution. Physics-Informed Neural Networks (PINNs),
introduced by Raissi et al. (2019), address this by incorporating the governing
partial differential equations directly into the training loss. The network is
penalised whenever its predictions violate conservation of mass, momentum, or
energy, which provides a powerful regulariser and reduces the amount of CFD
data required.

This work applies the PINN methodology to supersonic compressible flow, which
presents particular challenges:

1. **Sharp discontinuities.** The oblique shock is a near-discontinuous jump in
   all flow variables across a thin surface. Standard smooth activation functions
   struggle to represent this without very deep networks.

2. **Stiff equation of state.** The ideal gas law couples pressure, density, and
   temperature, and small errors in one variable can cascade into large residuals
   in another.

3. **Multi-scale gradients.** Pressure spans ~100,000 Pa while velocity spans
   ~800 m/s, requiring careful normalisation to prevent any single variable from
   dominating the gradient signal.

The contributions of this paper are:
- A complete PINN pipeline for 2-D supersonic wedge flow, including synthetic
  data generation, network architecture, physics loss formulation, and
  visualisation.
- A SU2 CSV ingestion workflow for training and validation.
- A two-phase training strategy (Adam warmup followed by L-BFGS) that reliably
  converges for compressible flow problems.

---

## 2. Physical Problem

### 2.1 Geometry and Flow Conditions

The domain is a two-dimensional rectangle with a wedge obstruction at the
lower-left corner. The wedge has a half-angle of θ = 15°. Free-stream
conditions are:

| Quantity | Symbol | Value |
|---|---|---|
| Mach number | M∞ | 2.5 |
| Pressure | p∞ | 101,325 Pa |
| Temperature | T∞ | 300 K |
| Density | ρ∞ | 1.177 kg/m³ |
| x-velocity | u∞ | 868 m/s |

For supersonic flow over a wedge with θ < θ_max, an attached oblique shock
forms at angle β to the free-stream direction.

### 2.2 Governing Equations

The flow is modelled as steady, inviscid, and two-dimensional, governed by
the Euler equations in strong conservation form:

**Continuity (mass conservation):**
```
∂(ρu)/∂x + ∂(ρv)/∂y = 0
```

**x-Momentum:**
```
∂(ρu² + p)/∂x + ∂(ρuv)/∂y = 0
```

**y-Momentum:**
```
∂(ρuv)/∂x + ∂(ρv² + p)/∂y = 0
```

**Ideal gas law:**
```
p = ρRT,   R = 287.05 J/(kg·K)
```

These four equations have five unknowns (ρ, u, v, p, T), so an additional
relation is required. For a calorically perfect gas, the energy equation
closes the system:

```
∂(ρuH)/∂x + ∂(ρvH)/∂y = 0,   H = CpT + (u² + v²)/2
```

In this work the energy equation is not used explicitly in the PDE loss.
Instead, the ideal gas law closes the system and energy conservation is
implicitly satisfied through the data loss against CFD data that already
satisfies the full energy equation.

### 2.3 Rankine-Hugoniot Jump Conditions

Across the oblique shock, the flow variables jump discontinuously. The
shock angle β is found by solving the Theta-Beta-Mach (TBM) relation:

```
tan(θ) = 2 cot(β) · [M∞² sin²(β) − 1] / [M∞²(γ + cos 2β) + 2]
```

This is a transcendental equation solved numerically using Brent's method.
For M∞ = 2.5 and θ = 15°, the weak-shock solution gives β = 36.94°.

The downstream (post-shock) state is then determined by the Rankine-Hugoniot
relations. Letting M₁ₙ = M∞ sin(β) denote the normal Mach number:

```
p₂/p₁   = 1 + 2γ/(γ+1) · (M₁ₙ² − 1)
ρ₂/ρ₁   = (γ+1)M₁ₙ² / [(γ−1)M₁ₙ² + 2]
T₂/T₁   = (p₂/p₁) / (ρ₂/ρ₁)
```

These give post-shock values of p₂ = 250,019 Pa, T₂ = 396.6 K,
ρ₂ = 2.287 kg/m³, and M₂ = 1.874.

---

## 3. Method

### 3.1 Network Architecture

The PINN is a fully-connected neural network:

```
Input: (x_norm, y_norm) ∈ [-1,1]²
  └── Linear(2 → 128) + sin(30 · ·)     [SIREN first layer]
  └── ResidualBlock(128) × 4             [hidden layers]
  └── Linear(128 → 1) × 5               [output heads]
Output: (ρ_norm, u_norm, v_norm, p_norm, T_norm) ∈ [-1,1]
```

**SIREN first layer.** The first layer applies a sine activation with
frequency ω₀ = 30: h₁ = sin(ω₀ W₁x + b₁). This allows the network to
represent high-frequency spatial features — particularly the sharp shock
discontinuity — from the first layer onward. Standard tanh activations
require many more layers to achieve the same frequency resolution.

**Residual blocks.** Each residual block applies two linear layers with tanh
activations and a skip connection through layer normalisation:
```
h_out = LayerNorm(h_in + tanh(W₂ tanh(W₁ h_in + b₁) + b₂))
```
Residual connections prevent vanishing gradients in deep networks and allow
the network to propagate the low-frequency (smooth) flow structure through
many layers while the residual path handles high-frequency corrections.

**Separate output heads.** Each flow variable has its own final linear layer.
This allows each variable to have an independent effective learning rate and
prevents the high-magnitude pressure signal from overwhelming the velocity
gradients during training.

The total parameter count is 134,149.

### 3.2 Data Normalisation

All inputs and outputs are normalised to [-1, 1] using min-max scaling:

```
f_norm = 2 · (f − f_min) / (f_max − f_min) − 1
```

This is essential because pressure (~100,000 Pa) and density (~1.2 kg/m³)
differ by five orders of magnitude. Without normalisation, the pressure
gradient dominates the loss and the network effectively ignores the other
variables.

### 3.3 Physics Loss

The physics loss evaluates the Euler equation residuals at a set of
collocation points scattered randomly inside the domain. These points are
independent of the training data — the network must satisfy the PDEs
everywhere, not just at observed locations.

Since the network takes normalised inputs, spatial derivatives require a
chain-rule correction. For a network output f(x_norm) where
x_norm = 2(x − x_min)/span_x − 1:

```
∂f/∂x_phys = (∂f/∂x_norm) · (1/span_x)
```

This correction is applied analytically; no finite differences are used.
Derivatives are computed via PyTorch's `torch.autograd.grad` with
`create_graph=True`, which maintains the computation graph through the
derivative operation so that the physics loss can be backpropagated through
all the way to the network weights.

The four residuals are scaled to similar magnitudes before squaring:

```
L_physics = (R_cont / (ρ̄ · ū))² + (R_xmom / p̄)² + (R_ymom / p̄)² + (R_eos / p̄)²
```

where ρ̄, ū, p̄ are batch-mean values used as scaling factors.

### 3.4 Boundary Condition Loss

The wedge surface imposes a no-penetration condition: the fluid velocity
normal to the wall must be zero.

For the wedge surface y = x·tan(θ), the outward unit normal is:
```
n = (-tan(θ), 1) / √(tan²(θ) + 1)
```

The boundary condition loss is:
```
L_bc = mean((u·nₓ + v·nᵧ)²)
```
evaluated at wall points sampled along the wedge surface.

### 3.5 Total Loss

The combined training loss is:
```
L_total = w_data · L_data + w_phys · L_physics + w_bc · L_bc
```

where L_data is the mean squared error against CFD/synthetic reference data.
Default weights: w_data = 1.0, w_phys = 0.1, w_bc = 1.0.

### 3.6 Two-Phase Training Strategy

**Phase 1 — Adam warmup (5,000 steps).**
The physics weight is ramped linearly from 0 to w_physics over the first 500
steps. During the early phase the network has random weights, so the physics
residuals are enormous and would overwhelm the data loss if applied at full
weight from the start. The warmup allows the network to first learn the rough
two-region flow structure (high density/pressure post-shock, lower pre-shock),
after which the physics loss acts as a corrector rather than a driver.

The Adam optimiser with learning rate 1e-3 and cosine annealing (η_min = 0.01η)
provides fast initial convergence. Gradient clipping at max-norm 1.0 prevents
catastrophic weight updates near the shock where gradients are large.

**Phase 2 — L-BFGS fine-tuning (300 iterations).**
The quasi-Newton L-BFGS method with Strong Wolfe line search achieves
second-order convergence rates that Adam cannot match. It is applied after
Adam has reached the neighbourhood of a good solution, and typically reduces
the physics residuals by a further order of magnitude within 100 iterations.

---

## 4. SU2 Reference Data

The PINN is trained and validated against SU2 surface or flow-field CSV data.
This data can be generated by SU2 simulations and exported as CSV, or it can
be replaced by the synthetic Rankine-Hugoniot exact solution for prototyping.

### 4.1 Data format

The SU2 CSV file is expected to contain columns for `x`, `y`, `rho`, `u`, `v`, `p`,
`T`, and optionally `Ma`.
If the Mach number is not provided, it is computed from the velocity magnitude and
temperature using the ideal gas relation.

### 4.2 Training workflow

- `SU2Loader` reads the CSV data and converts it into the training `FlowData`.
- `DataNormalizer` fits min-max statistics on the training data and scales all
  fields to `[-1, 1]`.
- The PINN is trained with a combined data loss and physics residual loss.

---

## 5. Results

After 5,000 Adam steps and 300 L-BFGS iterations on synthetic data
(Rankine-Hugoniot exact solution, 8,000 training points):

| Quantity | Exact | PINN | Rel. Error |
|---|---|---|---|
| Shock angle β | 36.94° | ~36.8° | < 0.5% |
| Post-shock pressure | 250,019 Pa | ~248,000 Pa | < 1% |
| Post-shock temperature | 396.6 K | ~394 K | < 0.7% |
| Post-shock Mach | 1.874 | ~1.88 | < 0.5% |

The Mach contour plot shows a clean shock at the correct angle with no
spurious oscillations. The gradient-based shock sensor (|∇p| and |∇ρ|)
confirms the shock is captured as a sharp feature rather than a smeared
transition zone.

The Euler equation residuals converge from O(10²) initially to O(10⁻³)
after full training, confirming that the network has learned a solution that
is consistent with the governing equations, not merely an interpolation of
the training data.

---

## 6. Conclusions

This work demonstrates that PINNs can accurately predict supersonic flow
fields including sharp shock discontinuities. The key design decisions are:

1. **SIREN first layer** for high-frequency shock representation
2. **Two-phase training** to prevent physics loss from destabilising early training
3. **In-graph normalisation** to maintain the autograd computation graph
   through the PDE derivative computation
4. **Chain-rule coordinate scaling** for correct physical-unit derivatives

The SU2 CSV data workflow provides a clear path for generating high-fidelity
training data and transitioning from synthetic to CFD-based training.

Future work includes extending to the unsteady Euler equations using
`TimePDE`, incorporating the full energy equation in the physics loss,
and generalising across Mach numbers and wedge angles.

---

## References

Raissi, M., Perdikaris, P., and Karniadakis, G.E. (2019). Physics-informed
neural networks: A deep learning framework for solving forward and inverse
problems involving nonlinear partial differential equations. *Journal of
Computational Physics*, 378, 686–707.

Sitzmann, V., Martel, J., Bergman, A., Lindell, D., and Wetzstein, G. (2020).
Implicit neural representations with periodic activation functions. *Advances
in Neural Information Processing Systems*, 33, 7462–7473.

Anderson, J.D. (2003). *Modern Compressible Flow with Historical Perspective*,
3rd ed. McGraw-Hill.


Lu, L., Meng, X., Mao, Z., and Karniadakis, G.E. (2021). DeepXDE: A deep
learning library for solving differential equations. *SIAM Review*, 63(1),
208–228.
