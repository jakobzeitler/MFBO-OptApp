
from __future__ import annotations
from typing import Dict, Optional
import torch
from botorch.models.deterministic import DeterministicModel
from torch import Tensor

class AffineFidelityCostModel(DeterministicModel):

    def __init__(
        self,
        fidelity_weights: Optional[Dict[int, float]] = None,
        fixed_cost: float = 0.01,
    ) -> None:

        if fidelity_weights is None:
            fidelity_weights = {-1: 1.0}
        super().__init__()
        self.fidelity_dims = sorted(fidelity_weights)
        self.fixed_cost = fixed_cost
        weights = torch.tensor([fidelity_weights[i] for i in self.fidelity_dims])
        self.register_buffer("weights", weights)
        self._num_outputs = 1

    def forward(self, X: Tensor) -> Tensor:
        output_tensor = torch.where(X[..., self.fidelity_dims] == 1, torch.tensor(1, dtype=torch.float64),
                                    torch.tensor(self.fixed_cost, dtype=torch.float64))

        lin_cost = torch.einsum(
            "...f,f", X[..., self.fidelity_dims], self.weights.to(X)
        )
        return output_tensor
