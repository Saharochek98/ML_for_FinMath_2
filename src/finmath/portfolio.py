
"""Portfolio allocation utilities."""

import torch


def normalize_weights(
    raw_weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert raw portfolio weights into dollar-neutral allocations.

    Args:
        raw_weights: Tensor of shape (batch_size, n_assets).
        eps: Small constant for numerical stability.

    Returns:
        Portfolio weights with approximately zero net exposure
        and unit gross exposure for non-degenerate inputs.

    Raises:
        ValueError: If the input tensor is not two-dimensional.
    """
    if raw_weights.ndim != 2:
        raise ValueError(
            "Expected weights with shape (batch_size, n_assets)"
        )

    centered = raw_weights - raw_weights.mean(
        dim=1,
        keepdim=True,
    )

    gross_exposure = centered.abs().sum(
        dim=1,
        keepdim=True,
    )

    return centered / gross_exposure.clamp_min(eps)
