import numpy as np
import torch
from data.loader import WedgeSyntheticData, DataNormalizer
from models.pinn import ShockwavePINN
from physics.physics_loss import PhysicsLoss
from training.trainer import PINNTrainer

np.random.seed(0)

synth = WedgeSyntheticData(
    mach_inf=2.5,
    wedge_half_angle_deg=15.0,
    n_points=400,
    seed=1,
)
raw = synth.generate()
normalizer = DataNormalizer().fit(raw)
norm_data = normalizer.transform(raw)
model = ShockwavePINN(hidden_layers=4, hidden_width=64)
loss_fn = PhysicsLoss(w_data=1.0, w_physics=0.1, w_bc=1.0)
trainer = PINNTrainer(
    model,
    loss_fn,
    normalizer,
    device='cpu',
    checkpoint_dir='checkpoints',
    log_interval=1,
)

x_d, y_d, target, x_col, y_col = trainer._build_tensors(norm_data, col_fraction=0.4)
print('data points', x_d.shape[0], 'collocation points', x_col.shape[0])

x_range_phys = (normalizer.stats['x'][0], normalizer.stats['x'][1])
x_wall, y_wall = trainer._make_wall_points(x_range_phys, synth.theta)

loss, breakdown = loss_fn.total_loss(
    model,
    x_d, y_d, target,
    x_col, y_col,
    normalizer,
    x_wall_norm=x_wall,
    y_wall_norm=y_wall,
    theta_rad=synth.theta,
    physics_weight_override=0.01,
)
print('loss', loss.item())
print('breakdown keys', list(breakdown.keys()))
print('data', breakdown['data'].item())
print('physics', breakdown['physics'].item())
print('bc', breakdown.get('bc', 'MISSING'))
print('residuals', breakdown['residuals'])

l_phys, residuals = loss_fn.physics_loss(model, x_col, y_col, normalizer)
print('physics loss', l_phys.item())
print('physics residuals', residuals)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
opt.zero_grad()
loss.backward()
opt.step()
print('one step OK')
