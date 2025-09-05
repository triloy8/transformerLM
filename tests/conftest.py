import os
import random
import pytest
import numpy as np
import torch


@pytest.fixture(scope="session", autouse=True)
def set_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def tiny_dims():
    return {
        "vocab_size": 16,
        "T": 4,
        "d_model": 8,
        "num_heads": 2,
        "num_layers": 1,
        "d_ff": 16,
    }