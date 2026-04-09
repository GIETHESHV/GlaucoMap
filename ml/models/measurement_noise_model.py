"""
Measurement Noise Model.

Models test-retest variability as a function of sensitivity level
using the empirical Henson curve:

    SD(sensitivity) = a * sensitivity^b

where a=0.081, b=0.473 (Henson et al. 2000).

Used to distinguish real progression from measurement noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MeasurementNoiseModel(nn.Module):
    """
    Henson curve: SD = a * sensitivity^b

    Parameters can be fit from a normative dataset or kept at defaults.

    Args:
        a: Henson curve parameter a (default 0.081)
        b: Henson curve parameter b (default 0.473)
        learnable: If True, a and b are learnable parameters.
    """

    def __init__(
        self,
        a: float = 0.081,
        b: float = 0.473,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        if learnable:
            self.log_a = nn.Parameter(torch.tensor(a).log())
            self.log_b = nn.Parameter(torch.tensor(b).log())
        else:
            self.register_buffer("log_a", torch.tensor(a).log())
            self.register_buffer("log_b", torch.tensor(b).log())

    @property
    def a(self) -> torch.Tensor:
        return self.log_a.exp()

    @property
    def b(self) -> torch.Tensor:
        return self.log_b.exp()

    def forward(self, sensitivity: torch.Tensor) -> torch.Tensor:
        """
        Compute expected measurement noise SD for given sensitivity values.

        Args:
            sensitivity : Tensor of any shape, values in dB [0, 40]

        Returns:
            noise_sd : same shape as input, in dB
        """
        sensitivity = sensitivity.clamp(min=0.1)  # avoid log(0)
        return self.a * sensitivity.pow(self.b)

    def noise_variance(self, sensitivity: torch.Tensor) -> torch.Tensor:
        """Return noise variance (SD^2)."""
        return self.forward(sensitivity) ** 2

    def fit_from_data(
        self,
        sensitivity_values: torch.Tensor,
        retest_sd_values: torch.Tensor,
        lr: float = 1e-3,
        epochs: int = 500,
    ) -> None:
        """
        Fit a and b from paired (sensitivity, retest_sd) observations
        using gradient descent.

        Args:
            sensitivity_values : [N] observed sensitivity values
            retest_sd_values   : [N] corresponding test-retest SDs
        """
        assert hasattr(self, "log_a") and isinstance(self.log_a, nn.Parameter), \
            "Model must be initialized with learnable=True to fit parameters"

        optimizer = torch.optim.Adam([self.log_a, self.log_b], lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            pred_sd = self.forward(sensitivity_values)
            loss = nn.functional.mse_loss(pred_sd, retest_sd_values)
            loss.backward()
            optimizer.step()
