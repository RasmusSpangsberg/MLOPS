import re

from src.models.model import MyAwesomeModel
import torch
from torch.utils.data import DataLoader
import pytest


def test_model():
    test_data_set = torch.load("data/processed/corruptmnist/test.pth")
    batch_size = 2
    test_loader = DataLoader(test_data_set, batch_size=batch_size)

    x, _ = next(iter(test_loader))

    assert x.shape == torch.Size([batch_size, 1, 28, 28]), "Input to model has incorrect shape"

    model = MyAwesomeModel()
    output = model(x)

    assert output.shape == torch.Size([batch_size, 10]), "Output from model has incorrect shape"


def test_error_on_wrong_shape():
    model = MyAwesomeModel()

    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1, 2, 3))

    # [ and ] are special characters that we need to escape because the below function uses regex
    # to find a matching string
    with pytest.raises(ValueError,
                       match=re.escape('Expected each sample to have shape [1, 28, 28]')):
        model(torch.randn(1, 2, 3, 4))
