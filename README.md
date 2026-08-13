# Shockwave PINN

This repository implements a physics-informed neural network (PINN) for 2-D supersonic wedge/shock flow. The pipeline can:

- generate synthetic exact oblique-shock data,
- read SU2 surface sample data,
- train a neural network to satisfy both data and Euler equations,
- save model checkpoints,
- visualize flow fields and error plots.

## Repository Structure

`shockwave_pinn_final/` is the main package. Important submodules:

- `main.py` — main CLI pipeline driver
- `visualize.py` — standalone visualization entrypoint
- `streamlit_app.py` — browser UI for pipeline control
- `models/pinn.py` — PINN network architecture
- `physics/physics_loss.py` — physics residual and loss definitions
- `data/loader.py` — SU2 CSV data reader, synthetic data generator, normalizer
- `training/trainer.py` — training loop and checkpointing
- `visualization/visualizer.py` — plotting utilities

Utility inspection scripts were previously included at the repository root, but those compare/inspect utilities have been removed because `outputs_test` and `outputs_test2` are no longer relevant.

---

## Core Components

### `shockwave_pinn_final/main.py`

**Purpose**
Main CLI for training, visualizing, or doing both in one run.

**Behavior**
- Fixes imports to work regardless of current working directory.
- Parses command-line arguments for task, mode, physical conditions, architecture, training, device, and output directory.
- Decides device: `cuda` if available else `cpu`.
- Loads raw data from either synthetic generation or SU2 CSV.
- Builds the PINN model and loss function.
- For training:
  - fits a `DataNormalizer` on raw training data,
  - normalizes the data,
  - saves `normalizer.npz`,
  - creates `PINNTrainer`,
  - trains with Adam then optional L-BFGS,
  - saves loss plot and final checkpoint.
- For visualization:
  - loads checkpoint and normalizer,
  - runs inference on a dense grid,
  - interpolates SU2 reference data,
  - saves contour plots and error plots.

**Key functions**
- `load_raw_data(args)` — loads synthetic or SU2 CSV data.
- `build_model(args)` — constructs `ShockwavePINN`.
- `build_loss(args)` — constructs `PhysicsLoss`.
- `build_trainer(...)` — builds trainer with checkpoint directory.
- `inference_and_plot(...)` — performs grid inference, denormalizes predictions, computes Mach number, interpolates CFD fields, and calls plotting routines.
- `--seed` — optional synthetic data seed. Use an integer for repeatable synthetic sample generation, or omit it for non-deterministic sampling.

---


### `shockwave_pinn_final/visualize.py`

**Purpose**
Standalone script to load a checkpoint and normalizer, then generate plots without training.

**Behavior**
- Parses CLI arguments for mode, checkpoint, normalizer path, data directory, output directory, device, and flow parameters.
- Loads raw data and checkpoint.
- Loads normalization stats from `.npz`.
- Calls `inference_and_plot()` from `main.py`.

---

### `shockwave_pinn_final/streamlit_app.py`

**Purpose**
Streamlit UI for controlling the pipeline via a browser.

**Behavior**
- Displays sidebar controls for pipeline mode, task, flow conditions, SU2 data path, and checkpoint path.
- Builds a `main.py` command for synthetic or SU2 workflow.
- Runs the command as a subprocess and streams output to the UI.
- Displays generated output images if they exist.

---

### `shockwave_pinn_final/models/pinn.py`

**Purpose**
Defines the PINN neural network architecture.

**Architecture**
- Input layer: `Linear(2, width)` followed by Sine activation.
- Hidden stack: either residual blocks with LayerNorm and Tanh or standard linear+Tanh layers.
- Output heads: separate linear layers for `rho`, `u`, `v`, `p`, and `T`.

**Key classes**
- `SineActivation` — applies `sin(omega_0 * x)`.
- `ResidualBlock` — two linear layers + Tanh + residual connection + LayerNorm.
- `ShockwavePINN` — full model.
- `ShockwavePINNLight` — smaller test variant.

**Methods**
- `forward(x, y)` — returns per-field output dict.
- `predict(x, y)` — NumPy inference wrapper.
- `_init_weights(...)` — custom SIREN-friendly initialization.

---

### `shockwave_pinn_final/physics/physics_loss.py`

**Purpose**
Implements physics-informed loss using Euler equation residuals and boundary conditions.

**Contents**
- `_grad()` — gradient helper using autograd.
- `euler_residuals(...)` — computes continuity, momentum, and ideal gas residuals.
- `PhysicsLoss` — combines data loss, physics residual loss, and BC loss.

**Important details**
- Model inputs/outputs are normalized; outputs are denormalized in-graph for residual evaluation.
- Chain rule corrects derivatives from normalized coordinates to physical coordinates.
- `bc_loss(...)` enforces no-penetration on the wedge surface.

---

### `shockwave_pinn_final/data/loader.py`

**Purpose**
Handles SU2 CSV ingestion, synthetic flow generation, and normalization.

**Main classes**
- `FlowData` — container for `x`, `y`, `rho`, `u`, `v`, `p`, `T`, and `Ma`.
- `SU2Loader` — reads SU2 CSV output and computes Mach if needed.
- `WedgeSyntheticData` — generates exact oblique-shock flow from theory.
- `DataNormalizer` — fits per-field min/max stats and normalizes to `[-1, 1]`.

**Usage**
- `main.py` fits the normalizer and transforms training data.
- `PhysicsLoss` uses in-graph denormalization to compute physical residuals.
- `visualize.py` and `main.py` use inverse transform for plotting.

---

### `shockwave_pinn_final/training/trainer.py`

**Purpose**
Training loop implementation for the PINN.

**Behavior**
- Converts NumPy arrays to PyTorch tensors on device.
- Splits points into training and collocation sets.
- Samples wall boundary points on the wedge surface.
- Trains in two phases:
  - Adam with physics weight warmup.
  - L-BFGS fine-tuning.
- Saves checkpoints and supports loading them.

---

### `shockwave_pinn_final/visualization/visualizer.py`

**Purpose**
Plotting utilities used by training and visualization scripts.

**Functions**
- `plot_mach_contour(...)`
- `plot_flow_field(...)`
- `plot_training_loss(...)`
- `plot_error_field(...)`
- `plot_shock_detection(...)`

**Implementation**
- Interpolates data onto a grid with `scipy.interpolate.griddata`.
- Uses Matplotlib to save PNG plots.

---

## Root utility scripts

The repository root no longer includes the old compare/inspect scripts. The main workflow is now:

- `shockwave_pinn_final/main.py` for training and visualization
- `shockwave_pinn_final/visualize.py` for inference-only plotting
- `shockwave_pinn_final/streamlit_app.py` for browser-based execution

---

## Package initialization note

`shockwave_pinn_final/__init__.py` is intentionally minimal and does not load saved model artifacts on import.

---

## Detailed function reference

### `shockwave_pinn_final/main.py`
- `build_parser()` — defines CLI arguments for task, mode, data directory, checkpoint, normalizer, output directory, device, flow conditions, architecture, and training hyperparameters.
- `parse_device(device_arg)` — returns `cuda` if available and `auto` was requested, otherwise returns the requested device string.
- `load_raw_data(args)` — loads either synthetic `WedgeSyntheticData` or SU2 CSV data via `SU2Loader` depending on `args.mode`; returns raw data plus computed shock and wedge angles.
- `build_model(args)` — constructs a `ShockwavePINN` instance using the requested hidden layer count and width.
- `build_loss(args)` — creates a `PhysicsLoss` instance with data, physics, and BC weights.
- `build_trainer(model, loss_fn, normalizer, device, output_dir, log_interval)` — builds a `PINNTrainer` and prepares checkpoint directory.
- `load_checkpoint(model, checkpoint_path, device)` — loads saved model weights from checkpoint into the model.
- `inference_and_plot(args, model, normalizer, raw_data, beta_rad, theta_rad, outdir, device)` — runs PINN prediction on a dense grid above the wedge, converts normalized outputs to physical values, interpolates reference SU2 fields, computes Mach number, and saves plots.
- `main()` — orchestrates training and visualization logic based on CLI task, handles normalizer saving/loading, and triggers the trainer and plotting functions.

### `shockwave_pinn_final/visualize.py`
- `build_parser()` — defines CLI arguments for visualization-only runs.
- `parse_device(device_arg)` — resolves device string the same way as `main.py`.
- `main()` — loads raw data, checkpoint, and normalizer, then calls `inference_and_plot()` to save plots.

### `shockwave_pinn_final/streamlit_app.py`
- `format_command(cmd)` — formats a subprocess command string safely.
- `run_subprocess(command, cwd)` — executes the command and streams stdout/stderr back to Streamlit.
- `show_output_images(output_dir)` — displays generated plot images in the UI.
- `build_main_command(...)` — builds a `main.py` command based on UI inputs.
- `main()` — renders the UI, collects inputs, runs the selected pipeline, and shows results.

### `shockwave_pinn_final/models/pinn.py`
- `SineActivation.forward(x)` — applies `sin(omega_0 * x)` for SIREN-style first-layer features.
- `ResidualBlock.forward(x)` — applies two linear layers with Tanh, adds a residual connection, and normalizes the result.
- `ShockwavePINN.__init__(hidden_layers, hidden_width, use_residual, omega_0)` — builds the network architecture and initializes weights.
- `ShockwavePINN._init_weights(omega_0)` — applies SIREN-inspired initialization to stabilize training.
- `ShockwavePINN.forward(x, y)` — stacks coordinates, applies the network, and returns a dict of field outputs.
- `ShockwavePINN.predict(x, y, device)` — wrapper for inference that returns NumPy arrays.
- `ShockwavePINN.n_parameters()` — counts model parameters.
- `ShockwavePINNLight.__init__()` — constructs a smaller test model.

### `shockwave_pinn_final/physics/physics_loss.py`
- `_grad(output, coord)` — computes autograd derivatives of a scalar output with respect to coordinates.
- `euler_residuals(x_norm, y_norm, rho, u, v, p, T, span_x, span_y)` — computes residuals for continuity, momentum, and ideal-gas relations using normalized coordinate gradients.
- `PhysicsLoss.__init__(w_data, w_physics, w_bc, gamma, R)` — stores loss weights and sets up MSE loss.
- `PhysicsLoss.data_loss(pred, target)` — computes MSE over predicted and target fields.
- `PhysicsLoss.physics_loss(model, x_col_norm, y_col_norm, normalizer)` — evaluates PDE residuals at collocation points after in-graph denormalization and scales them for numerical stability.
- `PhysicsLoss.bc_loss(pred_wall, theta_rad)` — enforces the wedge wall no-penetration condition by penalizing wall-normal velocity.
- `PhysicsLoss.total_loss(...)` — combines data loss, physics loss, and BC loss into a total objective, returning both the scalar loss and a breakdown dictionary.

### `shockwave_pinn_final/data/loader.py`
- `FlowData.n_points` — returns number of points.
- `FlowData.field_names()` — returns field names.
- `FlowData.to_tensors(device)` — converts fields into PyTorch tensors.
- `FlowData.summary()` — prints min/max/mean statistics for each field.
- `SU2Loader.__init__(data_path)` — initialises the SU2 CSV loader and resolves the input file.
- `SU2Loader.load()` — reads SU2 CSV data and returns a `FlowData` object.
- `WedgeSyntheticData.__init__(...)` — sets free-stream and wedge parameters, solves shock angle, computes post-shock state.
- `WedgeSyntheticData._tbm_residual(beta)` — theta-beta-Mach residual used to solve shock angle.
- `WedgeSyntheticData._solve_shock_angle()` — finds the weak shock angle using root finding.
- `WedgeSyntheticData._rankine_hugoniot()` — computes post-shock pressure, density, temperature, and Mach from jump conditions.
- `WedgeSyntheticData._velocity(M, T, flow_angle)` — computes velocity components from Mach and temperature.
- `WedgeSyntheticData.generate()` — generates random points above the wedge, assigns pre-shock or post-shock exact values, and returns `FlowData`.
- `WedgeSyntheticData.shock_angle_deg` / `wedge_angle_deg` — convenience property accessors.
- `DataNormalizer.fit(data)` — computes min/max statistics for each field.
- `DataNormalizer.transform(data)` — scales each field to `[-1, 1]`.
- `DataNormalizer.inverse_transform_field(field, arr)` — maps normalized arrays back to physical units.
- `DataNormalizer.inverse_transform_tensor(field, t)` — in-graph denormalization for autograd.
- `DataNormalizer.physical_coord_scale()` — returns physical span of `x` and `y` for gradient scaling.
- `DataNormalizer.save(path)` — writes stats to `.npz`.
- `DataNormalizer.load(path)` — loads stats from `.npz`.
- `WedgeSyntheticData.__init__(...)` — sets free-stream and wedge parameters, solves shock angle, computes post-shock state.
- `WedgeSyntheticData._tbm_residual(beta)` — theta-beta-Mach residual used to solve shock angle.
- `WedgeSyntheticData._solve_shock_angle()` — finds the weak shock angle using root finding.
- `WedgeSyntheticData._rankine_hugoniot()` — computes post-shock pressure, density, temperature, and Mach from jump conditions.
- `WedgeSyntheticData._velocity(M, T, flow_angle)` — computes velocity components from Mach and temperature.
- `WedgeSyntheticData.generate()` — generates random points above the wedge, assigns pre-shock or post-shock exact values, and returns `FlowData`.
- `WedgeSyntheticData.shock_angle_deg` / `wedge_angle_deg` — convenience property accessors.
- `DataNormalizer.fit(data)` — computes min/max statistics for each field.
- `DataNormalizer.transform(data)` — scales each field to `[-1, 1]`.
- `DataNormalizer.inverse_transform_field(field, arr)` — maps normalized arrays back to physical units.
- `DataNormalizer.inverse_transform_tensor(field, t)` — in-graph denormalization for autograd.
- `DataNormalizer.physical_coord_scale()` — returns physical span of `x` and `y` for gradient scaling.
- `DataNormalizer.save(path)` — writes stats to `.npz`.
- `DataNormalizer.load(path)` — loads stats from `.npz`.

### `shockwave_pinn_final/training/trainer.py`
- `PINNTrainer.__init__(...)` — stores model, loss, normalizer, device, and checkpoint directory.
- `PINNTrainer._t(arr, grad)` — helper to convert NumPy arrays to torch tensors.
- `PINNTrainer._make_wall_points(x_range, theta_rad, n)` — samples normalized points along the wedge surface for BC enforcement.
- `PINNTrainer._build_tensors(data, col_fraction)` — splits the dataset into data points and collocation points.
- `PINNTrainer.train_adam(...)` — runs Adam optimization with physics loss warmup and periodic checkpoint saving.
- `PINNTrainer.fine_tune_lbfgs(...)` — runs L-BFGS optimization for final model convergence.
- `PINNTrainer.train(...)` — executes Adam then L-BFGS sequentially.
- `PINNTrainer._save_checkpoint(tag)` — saves a checkpoint with model state and history.
- `PINNTrainer.save(path)` — saves final checkpoint to a given path.
- `PINNTrainer.load(path)` — loads checkpoint weights and history.

### `shockwave_pinn_final/visualization/visualizer.py`
- `_wedge(ax, theta, xmax, color)` — draws the wedge geometry patch.
- `_shock(ax, beta, xmax, **kw)` — draws the theoretical shock line.
- `_grid(x, y, f, res)` — interpolates scattered data onto a regular grid.
- `plot_mach_contour(...)` — plots Mach contours with the wedge and shock overlay.
- `plot_flow_field(...)` — plots CFD reference, PINN prediction, and pointwise error for selected fields.
- `plot_training_loss(history, save_path)` — plots loss terms and learning rate history.
- `plot_error_field(...)` — plots relative error contours for each field.
- `plot_shock_detection(...)` — plots gradient magnitude of pressure/density to reveal shock location.

### Root utility scripts
The old compare/inspect utility scripts are no longer part of the main repository workflow. If needed for historical reference, they are backed up in `backup_removed_inspect/`.

---

## Usage summary

### Train on synthetic data

```bash
python shockwave_pinn_final/main.py --mode synthetic --task train_visualize
```

### Run visualization only

```bash
python shockwave_pinn_final/visualize.py --mode synthetic --checkpoint outputs/model_final.pt
```

### SU2 CSV training

```bash
python shockwave_pinn_final/main.py --mode su2 --data_dir path/to/su2/output.csv
```

### Browser UI

```bash
streamlit run shockwave_pinn_final/streamlit_app.py
```

---

## Summary

This repository is built around a PINN for shockwave flow over a wedge, combining CFD or synthetic reference data with physics-based loss terms. The main pipeline supports training, inference, visualization, and SU2 CSV data ingestion.
