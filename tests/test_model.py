import torch

from src.model import get_model


def test_model_output_shape():
    model = get_model()

    x = torch.randn(2, 3, 32, 32)

    output = model(x)

    assert output.shape == (2, 10)


def test_model_is_pytorch_model():
    model = get_model()

    assert isinstance(model, torch.nn.Module)