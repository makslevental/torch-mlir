import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

from torch_mlir.eager_mode.torch_mlir_tensor import TorchMLIRTensor

warnings.simplefilter("always", UserWarning)
warnings.simplefilter("always", RuntimeWarning)

seed = random.randint(0, int(1e6))
seed = 522386
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNet18Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.train(False)

    def forward(self, img):
        return self.resnet.forward(img)


def test_net(mod_factory):
    mod = mod_factory()
    orig_params = {
        k: v.detach().clone() for k, v in list(mod.named_parameters(recurse=True))
    }

    torch_T = torch.randn((1, 3, 32, 32), requires_grad=True)
    mlir_T = TorchMLIRTensor(torch_T.detach().clone(), requires_grad=True)

    golden_y = mod(torch_T)
    golden_loss = golden_y.sum()
    golden_loss.backward()
    golden_grads = {
        k: v.grad.detach().clone() for k, v in list(mod.named_parameters(recurse=True))
    }

    del mod

    mod = mod_factory()
    for param, val in mod.named_parameters(recurse=True):
        val.data.copy_(orig_params[param])
        assert val.grad is None
        assert val.grad_fn is None

    test_y = mod(mlir_T)
    test_loss = test_y.sum()
    test_loss.backward()
    test_grads = {
        k: v.grad.detach().clone() for k, v in list(mod.named_parameters(recurse=True))
    }
    #

    assert np.allclose(
        golden_y.detach().numpy(),
        test_y.elem.numpy(),
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    ), np.linalg.norm(golden_y.detach().numpy() - test_y.elem.numpy())

    assert np.allclose(
        golden_loss.detach().numpy(),
        test_loss.elem.numpy(),
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    ), np.linalg.norm(golden_loss.detach().numpy() - test_loss.elem.numpy())

    assert np.allclose(
        torch_T.grad.detach().numpy(),
        mlir_T.grad.elem.numpy(),
        rtol=1e-03,
        atol=1e-03,
        equal_nan=True,
    ), np.linalg.norm(torch_T.grad.detach().numpy() - mlir_T.grad.elem.numpy())

    for param, val in test_grads.items():
        assert np.allclose(
            golden_grads[param].numpy(),
            val.elem.numpy(),
            rtol=1e-03,
            atol=1e-03,
            equal_nan=True,
        ), np.linalg.norm(golden_grads[param].numpy() - val.data.elem.numpy())

    print()


if __name__ == "__main__":
    test_net(Net)
    test_net(ResNet18Module)
