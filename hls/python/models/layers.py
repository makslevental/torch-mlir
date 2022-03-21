import torch
from torch import nn
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendInvoker

TORCH_MLIR_EXPORT_ATTR_NAME = "_torch_mlir_export"
TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME = "_torch_mlir_arg_annotations"


def export(fn):
    setattr(fn, TORCH_MLIR_EXPORT_ATTR_NAME, True)
    return fn


def annotate_args(annotations):
    def decorator(fn):
        setattr(fn, TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME, annotations)
        return fn

    return decorator


def annotate_forward(mod, annotations):
    a = annotate_args(annotations)
    mod.forward = a(mod.forward)
    mod.forward = export(mod.forward)
    return mod


class Conv2dNoPaddingOut(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):  # TODO
        super().__init__()
        torch.manual_seed(0)
        self.kernel_size = (kernel_size, kernel_size)
        conv = torch.nn.Conv2d(in_planes, out_planes, self.kernel_size, bias=False)
        self.weight = conv.weight
        self.train(False)

    @export
    def forward(self, inp, out):
        return torch._C._nn.thnn_conv2d(inp, self.weight, self.kernel_size, out=out)


def make_conv2d_no_bias_no_padding_no_stride_no_dilation_out(
    batch_size, in_planes, out_planes, kernel_size, height, width
) -> torch.nn.Module:
    c = annotate_forward(
        Conv2dNoPaddingOut,
        [
            None,
            ([batch_size, in_planes, height, width], torch.float32, True),
            (
                [
                    batch_size,
                    out_planes,
                    height - (kernel_size // 2 + 1) if height > 0 else -1,
                    width - (kernel_size // 2 + 1) if width > 0 else -1,
                ],
                torch.float32,
                True,
            ),
        ],
    )
    return c(in_planes, out_planes, kernel_size)


class MaxPool2dOut(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        torch.manual_seed(0)
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.ceil_mode = False
        self.train(False)

    @export
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


def make_maxpool2d_no_padding_no_stride_no_dilation_out(
    batch_size, channels, kernel_size, height, width
):
    out_dims = [
        batch_size,
        channels,
        height - (kernel_size // 2 + 1) if height > 0 else -1,
        width - (kernel_size // 2 + 1) if width > 0 else -1,
    ]
    m = annotate_forward(
        MaxPool2dOut,
        [
            None,
            ([batch_size, channels, height, width], torch.float32, True),
            (out_dims, torch.float32, True),  # out
            (out_dims, torch.float32, True),  # indices
        ],
    )
    return m(kernel_size)


class AdaptiveAvgPool2dOut(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.output_size = (1, 1)
        self.train(False)

    @export
    def forward(self, inp, out):
        return torch._C._nn.adaptive_avg_pool2d(inp, self.output_size, out=out)


def make_adaptiveavg2d_1x1(batch_size, channels, height, width):
    out_dims = [batch_size, channels, 1, 1]
    m = annotate_forward(
        AdaptiveAvgPool2dOut,
        [
            None,
            ([batch_size, channels, height, width], torch.float32, True),
            (out_dims, torch.float32, True),  # out
        ],
    )
    return m()


class ReLUInplace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.train(False)

    @export
    def forward(self, x):
        y = self.relu(x)
        return y


def make_relu(batch_size, channels, height, width):
    r = annotate_forward(
        ReLUInplace,
        [None, ([batch_size, channels, height, width], torch.float32, True)],
    )
    return r()


class BatchNorm2dOut(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        bn = torch.nn.BatchNorm2d(channels)
        self.weight = bn.weight
        self.bias = bn.bias
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var
        self.momentum = bn.momentum
        self.eps = bn.eps
        self.training = False
        self.train(self.training)

    @export
    def forward(self, inp, out, save_mean, save_invstd):
        out, *_ = torch.native_batch_norm(
            inp,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.training,
            self.momentum,
            self.eps,
            out=out,
            save_mean=save_mean,
            save_invstd=save_invstd,
        )
        return out


def make_bn(batch_size, channels, height, width):
    b = annotate_forward(
        BatchNorm2dOut,
        [
            None,
            ([batch_size, channels, height, width], torch.float32, True),  # input
            ([batch_size, channels, height, width], torch.float32, True),  # output
            ([batch_size, channels], torch.float32, True),  # save_mean
            ([batch_size, channels], torch.float32, True),  # save_invstd
        ],
    )
    return b(channels)


class LinearOut(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        fc = torch.nn.Linear(in_features, out_features)
        self.weight_T = torch.empty(fc.weight.T.shape)
        self.bias = fc.bias
        self.train(False)

    @export
    def forward(self, inp, out):
        y = torch._C._nn.linear(inp, self.weight_T, self.bias, out=out)
        return y


def make_linear_out(batch_size, in_features, out_features):
    l = annotate_forward(
        LinearOut,
        [
            None,
            ([batch_size, in_features], torch.float32, True),  # input
            ([batch_size, out_features], torch.float32, True),  # output
        ],
    )
    return l(in_features=in_features, out_features=out_features)


class TestMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = make_conv2d_no_bias_no_padding_no_stride_no_dilation_out(
            batch_size=5, in_planes=2, out_planes=10, kernel_size=3, height=20, width=20
        )
        self.relu = make_relu(batch_size=5, channels=10, height=18, width=18)
        self.train(False)

    def forward(self, inp, outp):
        x = self.conv(inp, outp)
        x = self.relu(x)
        # x = torch.flatten(x, 1)
        return x


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class MyBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = norm_layer(planes)
        # self.bn1.train(False)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.bn2 = norm_layer(planes)
        # self.bn2.train(False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_layer(test_module, annotations):
    class_annotator = ClassAnnotator()
    recursivescriptmodule = torch.jit.script(test_module)
    frozen_recursivescriptmodule = torch.jit.freeze(recursivescriptmodule)
    class_annotator.exportNone(frozen_recursivescriptmodule._c._type())
    class_annotator.exportPath(frozen_recursivescriptmodule._c._type(), ["forward"])
    class_annotator.annotateArgs(
        frozen_recursivescriptmodule._c._type(), ["forward"], annotations
    )
    # extract_annotations(test_module, frozen_recursivescriptmodule, class_annotator)
    # torch._C._jit_pass_inline(frozen_recursivescriptmodule.graph)
    return frozen_recursivescriptmodule, class_annotator


# class Matmul(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, lhs, rhs):
#         return torch.mm(lhs, rhs)
#
#
# class MatmulDotOut(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, lhs, rhs, out):
#         return torch.mm(lhs, rhs, out=out)


# class Conv2dNoPadding(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         torch.manual_seed(0)
#         self.conv = torch.nn.Conv2d(2, 10, 3, bias=False)
#         self.train(False)
#
#     def forward(self, x):
#         return self.conv(x)

# class MaxPool2d(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         torch.manual_seed(0)
#         self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.train(False)
#
#     def forward(self, inp):
#         return self.maxpool(inp)

# class AdaptiveAvgPool2d(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         torch.manual_seed(0)
#         self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
#         self.train(False)
#
#     def forward(self, inp):
#         return self.pool(inp)

# class BatchNorm2d(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bn = torch.nn.BatchNorm2d(4)
#         self.train(False)
#
#     def forward(self, x):
#         y = self.bn(x)
#         return y

# class Linear(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = torch.nn.Linear(512, 1000)
#         self.train(False)
#
#     def forward(self, x):
#         y = self.fc(x)
#         return y
