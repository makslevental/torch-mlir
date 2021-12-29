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


# JitOperator 'aten::native_batch_norm.out : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float, Tensor, Tensor, Tensor) -> (Tensor, Tensor, Tensor)':
# MLIR op name = torch.aten.native_batch_norm.out
# MLIR td def name = Torch_AtenNativeBatchNormOutOp
# namespace = aten
# unqualified_name = native_batch_norm
# overload_name = out
# is_c10_op = True
# is_vararg = False
# is_varret = False
# is_mutable = True
# arguments:
# arg: {'name': 'input', 'type': 'Tensor', 'pytype': 'Tensor'}
# arg: {'name': 'weight', 'type': 'Tensor?', 'pytype': 'Optional[Tensor]'}
# arg: {'name': 'bias', 'type': 'Tensor?', 'pytype': 'Optional[Tensor]'}
# arg: {'name': 'running_mean', 'type': 'Tensor?', 'pytype': 'Optional[Tensor]'}
# arg: {'name': 'running_var', 'type': 'Tensor?', 'pytype': 'Optional[Tensor]'}
# arg: {'name': 'training', 'type': 'bool', 'pytype': 'bool'}
# arg: {'name': 'momentum', 'type': 'float', 'pytype': 'float'}
# arg: {'name': 'eps', 'type': 'float', 'pytype': 'float'}
# arg: {'name': 'out', 'type': 'Tensor', 'pytype': 'Tensor', 'alias_info': {'is_write': True, 'before': ['alias::a'], 'after': ['alias::a']}}
# arg: {'name': 'save_mean', 'type': 'Tensor', 'pytype': 'Tensor', 'alias_info': {'is_write': True, 'before': ['alias::b'], 'after': ['alias::b']}}
# arg: {'name': 'save_invstd', 'type': 'Tensor', 'pytype': 'Tensor', 'alias_info': {'is_write': True, 'before': ['alias::c'], 'after': ['alias::c']}}
# returns:
# ret: {'name': '', 'type': 'Tensor', 'pytype': 'Tensor', 'alias_info': {'is_write': True, 'before': ['alias::a'], 'after': ['alias::a']}}
# ret: {'name': '', 'type': 'Tensor', 'pytype': 'Tensor', 'alias_info': {'is_write': True, 'before': ['alias::b'], 'after': ['alias::b']}}
# ret: {'name': '', 'type': 'Tensor', 'pytype': 'Tensor', 'alias_info': {'is_write': True, 'before': ['alias::c'], 'after': ['alias::c']}}


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
