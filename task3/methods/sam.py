# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Implement standard non-adaptive Sharpness-Aware Minimization."""

import torch
from torch.optim import AdamW


class SAMOptimizer:
    """Wrap AdamW with normalized ascent and descent parameter steps."""

    def __init__(self, parameters, rho, learning_rate, weight_decay):
        if rho <= 0.0:
            raise ValueError("SAM radius rho must be positive.")
        self.rho = float(rho)
        self.base_optimizer = AdamW(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.param_groups = self.base_optimizer.param_groups
        self.perturbations = {}

    def calculate_gradient_norm(self):
        """Calculate the L2 norm over every available parameter gradient."""
        gradient_norms = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    gradient_norms.append(parameter.grad.norm(p=2))
        if not gradient_norms:
            raise ValueError("SAM cannot perturb parameters without gradients.")
        device = gradient_norms[0].device
        return torch.norm(torch.stack([value.to(device) for value in gradient_norms]))

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        """Move parameters to the normalized local ascent point."""
        gradient_norm = self.calculate_gradient_norm()
        if not torch.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise FloatingPointError("SAM requires a positive finite gradient norm.")
        scale = self.rho / (gradient_norm + 1e-12)
        self.perturbations.clear()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                perturbation = parameter.grad * scale.to(parameter)
                parameter.add_(perturbation)
                self.perturbations[parameter] = perturbation
        if zero_grad:
            self.zero_grad()
        return gradient_norm

    @torch.no_grad()
    def restore_parameters(self):
        """Return every perturbed parameter to its original value."""
        for parameter, perturbation in self.perturbations.items():
            parameter.sub_(perturbation)
        self.perturbations.clear()

    def zero_grad(self, set_to_none=True):
        """Clear gradients through the wrapped AdamW optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return the underlying AdamW state for checkpoint storage."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Restore the underlying AdamW state from a checkpoint."""
        return self.base_optimizer.load_state_dict(state_dict)


def build_sam_optimizer(configuration, model):
    """Build the configured non-adaptive SAM optimizer around AdamW."""
    method = configuration["method"]
    training = configuration["training"]
    if method["adaptive"]:
        raise ValueError("The assignment requires non-adaptive SAM.")
    return SAMOptimizer(
        model.parameters(),
        rho=method["rho"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
