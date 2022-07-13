import torch
from torch import nn

from torch_mlir_e2e_test.torchscript.annotations import export, annotate_args


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    # https://dl.acm.org/doi/pdf/10.1145/2851507
    # https://hal.inria.fr/inria-00071879/document
    def forward(self, x):
        return (
            x
            + (x * x) * 0.5
            + (x * x * x) * 0.16666666666666666
            + (x * x * x * x) * 0.041666666666666664
            + 1
        )



class Div(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0x5F3759DF - x


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.exp = Exp()
        self.div = Div()

    @export
    @annotate_args([None, ([-1, -1, -1, 1], torch.float32, True)])
    def forward(self, x):
        y = self.exp(x)
        z = y.sum()
        return y * self.div(z)


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
        self.soft = Softmax()

    def forward(self, x):
        theta = self.theta_layer(x)
        phi = self.phi_layer(x)
        g = self.g_layer(x)

        theta_phi = theta * phi
        theta_phi = self.soft(theta_phi)
        theta_phi_g = theta_phi * g

        _out_tmp = self.out_cnn(theta_phi_g)
        _out_tmp = torch.add(_out_tmp, x)

        return _out_tmp


SCALE = 4


class BraggNN(torch.nn.Module):
    def __init__(self, imgsz=11, scale=SCALE):
        super().__init__()
        fcsz = tuple(map(int, (16 * scale, 8 * scale, 4 * scale, 2 * scale)))
        self.cnn_ops = []
        cnn_out_chs = tuple(map(int, (16 * scale, 8 * scale, 2 * scale)))
        cnn_in_chs = (1,) + cnn_out_chs[:-1]
        fsz = imgsz
        for (ic, oc) in zip(cnn_in_chs, cnn_out_chs):
            self.cnn_ops += [
                torch.nn.Conv2d(
                    in_channels=ic, out_channels=oc, kernel_size=3, stride=1, padding=0
                ),
                # torch.nn.LeakyReLU(negative_slope=0.025),
                torch.nn.ReLU(),
                # Exp()
            ]
            fsz -= 2
        self.nlb = NLB(in_ch=cnn_out_chs[0])
        self.dense_ops = []
        dense_in_chs = (fsz * fsz * cnn_out_chs[-1],) + fcsz[:-1]
        for ic, oc in zip(dense_in_chs, fcsz):
            self.dense_ops += [
                torch.nn.Linear(ic, oc),
                # torch.nn.LeakyReLU(negative_slope=0.025),
                torch.nn.ReLU(),
            ]
        # output layer
        if fcsz[-1] != 2:
            self.dense_ops += [torch.nn.Linear(fcsz[-1], 2)]

        self.cnn_layers_1 = self.cnn_ops[0]
        self.cnn_layers_2 = torch.nn.Sequential(*self.cnn_ops[1:])
        self.dense_layers = torch.nn.Sequential(*self.dense_ops)

    def forward(self, x):
        _out = x
        _out = self.cnn_layers_1(_out)
        _out = self.nlb(_out)
        _out = self.cnn_layers_2(_out)
        _out = _out.flatten(start_dim=1)
        _out = self.dense_layers(_out)

        return _out


class cnn_layers_1(torch.nn.Module):
    def __init__(self, imgsz=11, scale=SCALE):
        super().__init__()
        self.cnn_ops = []
        cnn_out_chs = tuple(map(int, (16 * scale, 8 * scale, 2 * scale)))
        cnn_in_chs = (1,) + cnn_out_chs[:-1]
        fsz = imgsz
        for (ic, oc) in zip(cnn_in_chs, cnn_out_chs):
            self.cnn_ops += [
                torch.nn.Conv2d(
                    in_channels=ic, out_channels=oc, kernel_size=3, stride=1, padding=0
                ),
                # torch.nn.LeakyReLU(negative_slope=0.025),
                torch.nn.ReLU(),
            ]
            fsz -= 2

        self.cnn_layers_1 = self.cnn_ops[0]

    def forward(self, x):
        _out = x
        _out = self.cnn_layers_1(_out)

        return _out


class nlb(torch.nn.Module):
    def __init__(self, imgsz=11, scale=SCALE):
        super().__init__()
        self.cnn_ops = []
        cnn_out_chs = tuple(map(int, (16 * scale, 8 * scale, 2 * scale)))
        self.nlb = NLB(in_ch=cnn_out_chs[0])

    def forward(self, x):
        _out = x
        _out = self.nlb(_out)

        return _out


class theta_phi_g(torch.nn.Module):
    def __init__(self, imgsz=11, scale=SCALE):
        super().__init__()
        cnn_out_chs = tuple(map(int, (16 * scale, 8 * scale, 2 * scale)))
        in_ch = cnn_out_chs[0]
        self.inter_ch = in_ch // 2
        self.conv = torch.nn.Conv2d(
            in_channels=in_ch, out_channels=self.inter_ch, kernel_size=1, padding=0
        )

    def forward(self, x):
        _out = x
        _out = self.conv(_out)
        return _out


class theta_phi_g_combine(torch.nn.Module):
    def __init__(self, imgsz=11, scale=SCALE):
        super().__init__()
        cnn_out_chs = tuple(map(int, (16 * scale, 8 * scale, 2 * scale)))
        in_ch = cnn_out_chs[0]
        self.inter_ch = in_ch // 2
        self.soft = nn.Softmax(dim=-1)
        self.out_cnn = torch.nn.Conv2d(
            in_channels=self.inter_ch, out_channels=in_ch, kernel_size=1, padding=0
        )

    def forward(self, x, theta, phi, g):
        _out = theta * phi
        _out = self.soft(_out)
        _out = _out * g
        _out = self.out_cnn(_out)
        _out = _out + x
        return _out


class cnn_layers_2(torch.nn.Module):
    def __init__(self, imgsz=11, scale=SCALE):
        super().__init__()
        self.cnn_ops = []
        cnn_out_chs = tuple(map(int, (16 * scale, 8 * scale, 2 * scale)))
        cnn_in_chs = (1,) + cnn_out_chs[:-1]
        fsz = imgsz
        for (ic, oc) in zip(cnn_in_chs, cnn_out_chs):
            self.cnn_ops += [
                torch.nn.Conv2d(
                    in_channels=ic, out_channels=oc, kernel_size=3, stride=1, padding=0
                ),
                # torch.nn.LeakyReLU(negative_slope=0.025),
                torch.nn.ReLU(),
                # Exp()
            ]
            fsz -= 2

        self.cnn_layers_2 = torch.nn.Sequential(*self.cnn_ops[1:])

    def forward(self, x):
        _out = x
        _out = self.cnn_layers_2(_out)

        return _out


class dense_layers(torch.nn.Module):
    def __init__(self, imgsz=11, scale=SCALE):
        super().__init__()
        fcsz = tuple(map(int, (16 * scale, 8 * scale, 4 * scale, 2 * scale)))
        self.cnn_ops = []
        cnn_out_chs = tuple(map(int, (16 * scale, 8 * scale, 2 * scale)))
        cnn_in_chs = (1,) + cnn_out_chs[:-1]
        fsz = imgsz
        for (ic, oc) in zip(cnn_in_chs, cnn_out_chs):
            fsz -= 2
        self.dense_ops = []
        dense_in_chs = (fsz * fsz * cnn_out_chs[-1],) + fcsz[:-1]
        for ic, oc in zip(dense_in_chs, fcsz):
            self.dense_ops += [
                torch.nn.Linear(ic, oc),
                # torch.nn.LeakyReLU(negative_slope=0.025),
                torch.nn.ReLU(),
            ]
        # output layer
        self.dense_ops += [torch.nn.Linear(fcsz[-1], 2)]
        self.dense_layers = torch.nn.Sequential(*self.dense_ops)

    def forward(self, x):
        _out = x
        _out = _out.flatten(start_dim=1)
        _out = self.dense_layers(_out)

        return _out
