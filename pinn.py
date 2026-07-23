"""
models/pinn.py
--------------
Physics-Informed Neural Network architecture for supersonic 2-D wedge flow.

Network: (x, y) -> (rho, u, v, p, T)

Design choices:
  - SIREN sine first layer: captures sharp shock discontinuity
  - Residual blocks with LayerNorm: stable deep training
  - Separate output heads per variable: independent learning rates
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List


class SineActivation(nn.Module):
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1  = nn.Linear(width, width)
        self.fc2  = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.act  = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.act(self.fc2(self.act(self.fc1(x)))))


class ShockwavePINN(nn.Module):
    """
    Physics-Informed Neural Network for supersonic 2-D compressible flow.

    Parameters
    ----------
    hidden_layers : number of hidden layers (must be even for residual mode)
    hidden_width  : neurons per layer
    use_residual  : use residual connections (recommended for depth >= 6)
    omega_0       : SIREN frequency for first layer
    """

    def __init__(
        self,
        hidden_layers: int   = 8,
        hidden_width:  int   = 128,
        use_residual:  bool  = True,
        omega_0:       float = 30.0,
    ):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(2, hidden_width),
            SineActivation(omega_0),
        )

        layers: List[nn.Module] = []
        if use_residual:
            for _ in range(max(1, hidden_layers // 2)):
                layers.append(ResidualBlock(hidden_width))
        else:
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]

        self.hidden = nn.Sequential(*layers)

        self._rho_head = nn.Linear(hidden_width, 1)
        self._u_head   = nn.Linear(hidden_width, 1)
        self._v_head   = nn.Linear(hidden_width, 1)
        self._p_head   = nn.Linear(hidden_width, 1)
        self._T_head   = nn.Linear(hidden_width, 1)

        self._init_weights(omega_0)

    def _init_weights(self, omega_0: float):
        with torch.no_grad():
            n = self.input_layer[0].weight.shape[1]
            self.input_layer[0].weight.uniform_(-1.0 / n, 1.0 / n)
            nn.init.zeros_(self.input_layer[0].bias)
            for module in self.modules():
                if isinstance(module, nn.Linear) and module is not self.input_layer[0]:
                    fan_in = module.weight.shape[1]
                    bound  = np.sqrt(6.0 / fan_in) / omega_0
                    module.weight.uniform_(-bound, bound)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        x, y : (N,) normalised coordinate tensors

        Returns
        -------
        dict of (N, 1) tensors: rho, u, v, p, T
        """
        xy = torch.stack([x, y], dim=-1)
        h  = self.input_layer(xy)
        h  = self.hidden(h)
        return {
            "rho": self._rho_head(h),
            "u":   self._u_head(h),
            "v":   self._v_head(h),
            "p":   self._p_head(h),
            "T":   self._T_head(h),
        }

    def predict(self, x: np.ndarray, y: np.ndarray, device: str = "cpu") -> dict:
        """Inference wrapper returning numpy arrays."""
        self.eval()
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).to(device)
            yt = torch.tensor(y, dtype=torch.float32).to(device)
            out = self.forward(xt, yt)
        return {k: v.squeeze(-1).cpu().numpy() for k, v in out.items()}

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ShockwavePINNLight(ShockwavePINN):
    """Small variant for quick tests — 4 layers, 64 neurons."""
    def __init__(self):
        super().__init__(hidden_layers=4, hidden_width=64, use_residual=False)
