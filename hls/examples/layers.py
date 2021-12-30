import torch


class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs, rhs):
        return torch.mm(lhs, rhs)


class MatmulDotOut(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lhs, rhs, out):
        return torch.mm(lhs, rhs, out=out)


class Conv2dNoPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 10, 3, bias=False)
        self.train(False)

    def forward(self, x):
        return self.conv(x)


class Conv2dNoPaddingOutModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.kernel_size = (3, 3)
        conv = torch.nn.Conv2d(2, 10, self.kernel_size, bias=False)
        self.weight = conv.weight
        self.train(False)

    def forward(self, inp, out):
        return torch._C._nn.thnn_conv2d(inp, self.weight, self.kernel_size, out=out)


class MaxPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.train(False)

    def forward(self, inp):
        return self.maxpool(inp)


class MaxPool2dOutModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.kernel_size = (3, 3)
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.ceil_mode = False
        self.return_indices = False
        self.train(False)

    def forward(self, inp, out, indices):
        out, _ = torch._C._nn.max_pool2d_with_indices(
            inp,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            out=out,
            indices=indices,
        )
        return out


class AdaptiveAvgPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.train(False)

    def forward(self, inp):
        return self.pool(inp)


class AdaptiveAvgPool2dOutModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.output_size = (1, 1)
        self.train(False)

    def forward(self, inp, out):
        return torch._C._nn.adaptive_avg_pool2d(inp, self.output_size, out=out)


class ReLUModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.train(False)

    def forward(self, x):
        y = self.relu(x)
        return y


class BatchNorm2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(4)
        self.train(False)

    def forward(self, x):
        y = self.bn(x)
        return y


class BatchNorm2dOut(torch.nn.Module):
    def __init__(self):
        super().__init__()
        bn = torch.nn.BatchNorm2d(4)
        self.weight = bn.weight
        self.bias = bn.bias
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var
        self.momentum = bn.momentum
        self.eps = bn.eps
        self.training = False
        self.train(self.training)

    def forward(self, inp, out, save_mean, save_invstd):
        out, *_ = torch.native_batch_norm(
            inp, self.weight, self.bias, self.running_mean, self.running_var, self.training, self.momentum, self.eps,
            out=out, save_mean=save_mean, save_invstd=save_invstd
        )
        return out


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(512, 1000)
        self.train(False)

    def forward(self, x):
        y = self.fc(x)
        return y


class LinearOut(torch.nn.Module):
    def __init__(self):
        super().__init__()
        fc = torch.nn.Linear(512, 1000)
        self.weight_T = torch.empty(fc.weight.T.shape)
        self.bias = fc.bias
        self.train(False)

    def forward(self, inp, out):
        y = torch._C._nn.linear(inp, self.weight_T, self.bias, out=out)
        return y
