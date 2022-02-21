import torch
from torch import nn
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export


class NLB(torch.nn.Module):
    def __init__(self, in_ch, relu_a=0.01):
        self.inter_ch = in_ch // 2
        super().__init__()
        self.theta_layer = torch.nn.Conv2d(
            in_channels=in_ch, out_channels=self.inter_ch, kernel_size=1, padding=0
        )
        self.phi_layer = torch.nn.Conv2d(
            in_channels=in_ch, out_channels=self.inter_ch, kernel_size=1, padding=0
        )
        self.g_layer = torch.nn.Conv2d(
            in_channels=in_ch, out_channels=self.inter_ch, kernel_size=1, padding=0
        )
        self.out_cnn = torch.nn.Conv2d(
            in_channels=self.inter_ch, out_channels=in_ch, kernel_size=1, padding=0
        )

        for m in self.modules():
            if hasattr(m, "weight"):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)

    def forward(self, x):
        theta = self.theta_layer(x)
        phi = self.phi_layer(x)
        g = self.g_layer(x)

        theta_phi = theta * phi
        # theta_phi = theta_phi.exp() / theta_phi.exp().sum((-2, -1), keepdim=True)
        # theta_phi = torch.ops.aten._softmax(theta_phi, -1, False)
        theta_phi_g = theta_phi * g

        _out_tmp = self.out_cnn(theta_phi_g)
        _out_tmp = torch.add(_out_tmp, x)

        return _out_tmp


class BraggNN(torch.nn.Module):
    def __init__(
        self,
        imgsz=11,
        # fcsz=(64, 32, 16, 8)
        fcsz=(16, 8, 4, 2),
    ):
        super().__init__()
        self.cnn_ops = []
        cnn_out_chs = (16, 8, 2)
        # cnn_out_chs = (64, 32, 8)
        cnn_in_chs = (1,) + cnn_out_chs[:-1]
        fsz = imgsz
        for (
            ic,
            oc,
        ) in zip(cnn_in_chs, cnn_out_chs):
            self.cnn_ops += [
                torch.nn.Conv2d(
                    in_channels=ic, out_channels=oc, kernel_size=3, stride=1, padding=0
                ),
                torch.nn.LeakyReLU(negative_slope=0.01),
            ]
            fsz -= 2
        self.nlb = NLB(in_ch=cnn_out_chs[0])
        self.dense_ops = []
        dense_in_chs = (fsz * fsz * cnn_out_chs[-1],) + fcsz[:-1]
        for ic, oc in zip(dense_in_chs, fcsz):
            self.dense_ops += [
                torch.nn.Linear(ic, oc),
                torch.nn.LeakyReLU(negative_slope=0.01),
            ]
        # output layer
        self.dense_ops += [
            torch.nn.Linear(fcsz[-1], 2),
        ]

        for m in self.cnn_ops:
            if hasattr(m, "weight"):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)

        for m in self.dense_ops:
            if hasattr(m, "weight"):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)

        self.cnn_layers_1 = self.cnn_ops[0]
        self.cnn_layers_2 = torch.nn.Sequential(*self.cnn_ops[1:])
        self.dense_layers = torch.nn.Sequential(*self.dense_ops)

        for m in self.modules():
            if hasattr(m, "weight"):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)

    def forward(self, x):
        _out = x
        _out = self.cnn_layers_1(_out)
        _out = self.nlb(_out)
        _out = self.cnn_layers_2(_out)
        _out = _out.flatten(start_dim=1)
        _out = self.dense_layers(_out)

        return _out
