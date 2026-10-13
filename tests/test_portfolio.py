
import pytest
import torch

from finmath.portfolio import normalize_weights


def test_weights_are_dollar_neutral():
    raw_weights = torch.tensor([
        [0.4, -0.1, 0.7],
        [0.3, 0.5, -0.2],
    ])

    weights = normalize_weights(raw_weights)

    assert torch.allclose(
        weights.sum(dim=1),
        torch.zeros(2),
        atol=1e-6,
    )


def test_weights_have_unit_gross_exposure():
    raw_weights = torch.tensor([
        [0.4, -0.1, 0.7],
    ])

    weights = normalize_weights(raw_weights)

    assert torch.allclose(
        weights.abs().sum(dim=1),
        torch.ones(1),
        atol=1e-6,
    )


def test_identical_weights_do_not_produce_nan():
    raw_weights = torch.ones(2, 4)

    weights = normalize_weights(raw_weights)

    assert torch.isfinite(weights).all()
    assert torch.all(weights == 0)


def test_invalid_input_shape_raises_error():
    raw_weights = torch.tensor([0.2, 0.5, -0.1])

    with pytest.raises(ValueError):
        normalize_weights(raw_weights)
