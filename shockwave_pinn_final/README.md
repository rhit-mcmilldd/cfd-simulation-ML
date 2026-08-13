# Shockwave PINN

Physics-Informed Neural Network for predicting supersonic oblique shockwaves
over a 2-D wedge. Trained on SU2 reference data or synthetic oblique shock flow.

---

## Project structure

```
shockwave_pinn_final/
├── main.py                          # Entry point — runs the full pipeline
├── requirements.txt
├── data/
│   └── loader.py                    # SU2 CSV reader + synthetic generator
├── models/
│   └── pinn.py                      # PINN architecture (SIREN + residual blocks)
├── physics/
│   └── physics_loss.py              # Euler equation residuals via autograd
├── training/
│   └── trainer.py                   # Adam + L-BFGS two-phase training loop
├── visualization/
│   └── visualizer.py                # Contour plots, error maps, loss curves
├── README.md
├── PAPER.md
├── streamlit_app.py
├── visualize.py
└── __init__.py
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training only

Generate training data and train the PINN without visualization.

```bash
python main.py --task train --mode synthetic
```

### 3. Run training + visualization

Train the model and then produce plots from the trained checkpoint.

```bash
python main.py --task train_visualize --mode synthetic
```

### 4. Run on SU2 CSV data

```bash
python main.py --task train --mode su2 --data_dir path/to/su2/output.csv
```

### 5. Streamlit UI

Run the browser-based interface with:

```bash
streamlit run streamlit_app.py
```

Use the sidebar to select:
- `mode` (`synthetic` or `su2`)
- `task` (`train`, `train_visualize`, `visualize`)
- `mach_inf`, `wedge_angle_deg`, `p_inf`, `T_inf`, `rho_inf`
- `output_dir` and SU2 CSV data settings when needed

The UI also displays generated output images after the pipeline finishes.

### 6. Visualize from a saved checkpoint

```bash
python main.py --task visualize --mode synthetic --checkpoint outputs/model_final.pt
```

For SU2 visualization, use `--mode su2` and the same CSV file or directory used during training.

---

## Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `train_visualize` | `train`, `visualize`, or `train_visualize` |
| `--mode` | `synthetic` | `synthetic` or `su2` |
| `--data_dir` | `path/to/su2/output.csv` | SU2 CSV file or directory for mode=su2 |
| `--mach_inf` | `2.5` | Free-stream Mach number |
| `--wedge_angle_deg` | `15.0` | Wedge half-angle [degrees]. |
| `--aoa_deg` | `None` | Alias for `--wedge_angle_deg` when using the wedge as a flow incidence angle. |
| `--n_points` | `8000` | Training points |
| `--hidden_layers` | `8` | PINN depth |
| `--hidden_width` | `128` | PINN width |
| `--n_adam` | `5000` | Adam training steps |
| `--n_lbfgs` | `300` | L-BFGS iterations |
| `--lr` | `1e-3` | Adam learning rate |
| `--w_physics` | `0.1` | Physics loss weight |
| `--w_bc` | `1.0` | Wall BC loss weight |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

---

## Outputs

All saved to `outputs/` by default:

| File | Description |
|---|---|
| `mach_contour.png` | Mach number field with shock line overlay |
| `flow_field.png` | Side-by-side CFD vs PINN vs Error for ρ, u, p, Ma |
| `error_field.png` | Relative error maps |
| `shock_detection.png` | Gradient-based shock sensor |
| `training_loss.png` | Loss curves and LR schedule |
| `model_final.pt` | Saved model weights |
| `normalizer.npz` | Normalisation statistics |

---

## Physics

The PINN enforces the 2-D steady Euler equations at random interior collocation
points via automatic differentiation (no grid, no finite differences):

```
∂(ρu)/∂x + ∂(ρv)/∂y = 0          continuity
∂(ρu²+p)/∂x + ∂(ρuv)/∂y = 0      x-momentum
∂(ρuv)/∂x + ∂(ρv²+p)/∂y = 0      y-momentum
p = ρRT                             ideal gas law
```

The wedge surface no-penetration condition (V·n = 0) is enforced as an
additional boundary condition loss term.

---

## References

- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural
  networks: A deep learning framework for solving forward and inverse problems
  involving nonlinear partial differential equations. *Journal of Computational
  Physics*, 378, 686-707.

- Anderson, J.D. (2003). *Modern Compressible Flow*, 3rd ed. McGraw-Hill.

- Sitzmann, V. et al. (2020). Implicit Neural Representations with Periodic
  Activation Functions (SIREN). *NeurIPS 2020*.
