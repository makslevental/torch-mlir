import torch
from torch.nn import Sequential, Module

from layers import (
    make_conv2d_no_bias_no_padding_no_stride_no_dilation_out,
    make_bn,
    make_relu,
    make_maxpool2d_no_padding_no_stride_no_dilation_out,
    make_adaptiveavg2d_1x1,
    make_linear_out, make_layer,
)

BATCH_SIZE = 1


def conv3x3(
        in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
):
    """3x3 convolution with padding"""
    return make_conv2d_no_bias_no_padding_no_stride_no_dilation_out(
        BATCH_SIZE, in_planes, out_planes, kernel_size=3, height=-1, width=-1
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return make_conv2d_no_bias_no_padding_no_stride_no_dilation_out(
        BATCH_SIZE, in_planes, out_planes, kernel_size=1, height=-1, width=-1
    )


class BasicBlock(Module):
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
        norm_layer = lambda planes: make_bn(BATCH_SIZE, planes, -1, -1)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = make_relu(BATCH_SIZE, planes, -1, -1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, out1, out2, save_mean, save_invstd):
        identity = x

        x = self.conv1(x, out1)
        x = self.bn1(x, out1, save_mean, save_invstd)
        x = self.relu(x)

        x = self.conv2(x, out2)
        x = self.bn2(x, out2, save_mean, save_invstd)

        if self.downsample is not None:
            identity = self.downsample(x)

        x = torch.add(x, identity, alpha=1, out=out2)
        x = self.relu(x)

        return x


class Bottleneck(Module):
    expansion: int = 4

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
            norm_layer = lambda planes: make_bn(BATCH_SIZE, planes, -1, -1)
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = make_relu(BATCH_SIZE, planes * self.expansion, -1, -1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, out, save_mean, save_invstd):
        identity = x

        x = self.conv1(x, out)
        x = self.bn1(x, out, save_mean, save_invstd)
        x = self.relu(x)

        x = self.conv2(x, out)
        x = self.bn2(x, out, save_mean, save_invstd)
        x = self.relu(x)

        x = self.conv3(x, out)
        x = self.bn3(x, out, save_mean, save_invstd)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet(Module):
    def __init__(
            self,
            block,
            layers,
            num_classes: int = 10,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation=[False, False, False],
            norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = lambda planes: make_bn(BATCH_SIZE, planes, -1, -1)
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = Conv2d(
        #     3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        # )
        self.conv1 = make_conv2d_no_bias_no_padding_no_stride_no_dilation_out(
            BATCH_SIZE, 3, self.inplanes, kernel_size=7, height=-1, width=-1
        ),
        self.bn1 = make_bn(BATCH_SIZE, self.inplanes, -1, -1),
        self.relu1 = make_relu(BATCH_SIZE, self.inplanes, -1, -1),
        self.maxpool1 = make_maxpool2d_no_padding_no_stride_no_dilation_out(
            BATCH_SIZE, self.inplanes, kernel_size=3, height=-1, width=-1
        ),

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = make_adaptiveavg2d_1x1(BATCH_SIZE, -1, -1, -1)
        # self.fc = Linear(512 * block.expansion, num_classes)
        self.fc = make_linear_out(BATCH_SIZE, 512 * block.expansion, num_classes)

    def _make_layer(
            self,
            block,
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return Sequential(*layers)

    def _forward_impl(self, x, out, save_mean, save_invstd, indices):
        # See note [TorchScript super()]
        x = self.conv1(x, out)
        x = self.bn1(x, out, save_mean, save_invstd)
        x = self.relu1(x)
        x = self.maxpool1(x, out, indices)

        x = self.layer1(x, out, save_mean, save_invstd)
        x = self.layer2(x, out, save_mean, save_invstd)
        x = self.layer3(x, out, save_mean, save_invstd)
        x = self.layer4(x, out, save_mean, save_invstd)

        x = self.avgpool(x, out)
        # x = torch.flatten(x, 1)
        x = self.fc(x, out)

        return x

    def forward(self, x, out, save_mean, save_invstd, indices):
        return self._forward_impl(x, out, save_mean, save_invstd, indices)


def resnet18() -> ResNet:
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model


def make_test_basic_block():
    b = BasicBlock(3, 10)
    return make_layer(b, [
        None,
        ([BATCH_SIZE, 3, -1, -1], torch.float32, True),  # inp
        ([BATCH_SIZE, -1, -1, -1], torch.float32, True),  # out1
        ([BATCH_SIZE, -1, -1, -1], torch.float32, True),  # out2
        ([BATCH_SIZE, -1], torch.float32, True),  # save_mean
        ([BATCH_SIZE, -1], torch.float32, True),  # save_invstd

    ])
