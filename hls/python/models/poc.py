import re
import subprocess

import torch

# noinspection PyUnresolvedReferences
from torch import nn
from torchvision.models.resnet import BasicBlock

from braggnn import BraggNN

# from eager.torch_dispatch import TorchMLIRTensor
from hls.python.models.resnet18 import ResNet18
from layers import make_layer
from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder

# noinspection PyUnresolvedReferences
from torch_mlir.passmanager import PassManager

# i have no idea why but if you don't import this then linalg passes aren't registered
# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendInvoker

# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export


from torch_mlir_e2e_test.utils import run_pipeline_with_repro_report


def set_weights(mod, typ=torch.float32, val=1, requires_grad=False):
    for m in mod.modules():
        if hasattr(m, "weight"):
            nn.init.constant_(m.weight, val)
            m.weight.requires_grad_(False)
            m.weight = torch.nn.Parameter(
                m.weight.type(typ), requires_grad=requires_grad
            )
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, val)
            m.bias.requires_grad_(False)
            m.bias = torch.nn.Parameter(
                m.bias.type(torch.float32), requires_grad=requires_grad
            )


def make_resnet():
    mod = ResNet18()
    print(mod)
    mod.eval()
    mod.apply(set_weights)
    return make_layer(mod, [None, ([1, 3, 8, 8], torch.float32, True)])


def make_mod():
    mod = BasicBlock(2, 2)
    mod.eval()
    mod.apply(set_weights)
    return make_layer(mod, [None, ([1, 2, 8, 8], torch.float32, True)])


class SmallBraggNN(nn.Module):
    def __init__(self, img_size=11):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, bias=False
        )
        self.dense = torch.nn.Linear(in_features=(img_size - 2) ** 2, out_features=2)

    def forward(self, x):
        out = self.conv(x)
        out = out.flatten(start_dim=1)
        out = self.dense(out)
        return out


def make_small_braggnn(in_channels=1, hw=11):
    with torch.no_grad():
        mod = SmallBraggNN(img_size=hw)
        mod.eval()
        mod.apply(set_weights)

        t = torch.randn((1, in_channels, hw, hw))
        y = mod(t)
        print(y.shape)

        return make_layer(mod, [None, ([1, in_channels, hw, hw], torch.float32, True)])


def make_conv(
    hw=5, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False
):
    with torch.no_grad():
        mod = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        mod.eval()
        # mod.apply(set_weights)
        return make_layer(mod, [None, ([1, in_channels, hw, hw], torch.float32, True)])


def make_mm():
    h = w = 10
    with torch.no_grad():
        mod = torch.nn.Linear(h, w, bias=False)
        mod.eval()
        # mod.apply(set_weights)
        return make_layer(mod, [None, ([1, 1, 10, 10], torch.float32, True)])


def make_braggnn(scale=4, imgsz=11):
    with torch.no_grad():
        mod = BraggNN(imgsz=imgsz, scale=scale)
        mod.eval()
        mod(torch.randn(1, 1, imgsz, imgsz))
        mod.apply(set_weights)
        return make_layer(mod, [None, ([1, 1, imgsz, imgsz], torch.float32, True)])


def make_resnet18():
    with torch.no_grad():
        mod = ResNet18()
        return make_layer(mod, [None, ([1, 3, 32, 32], torch.int32, True)])


def hack_for_calyx(out):
    out = (
        out.replace("f32", "i32")
        .replace("2.000000e+00", "2")
        .replace("1.000000e+00", "1")
    )
    open(f"../scripts/braggnn.affine.mlir", "w").write(out)
    out = subprocess.run(
        [
            "/home/mlevental/dev_projects/torch-mlir/cmake-build-debug/bin/torch-mlir-opt",
            "-pass-pipeline=reconcile-unrealized-casts",
            "/home/mlevental/dev_projects/torch-mlir/hls/scripts/braggnn.affine.mlir",
        ],
        capture_output=True,
    )

    # %0 = memref.get_global @__constant_16x1x3x3xi32 : memref<16x1x3x3xi32>
    # arith.constant dense<1.000000e+00> : vector<32x256xf32>
    # out = re.sub(r"memref.get_global .* : memref<(.*)xi32>", r"arith.constant dense<1> : vector<\1xi32>", out.stdout.decode())
    out = re.sub(
        r"memref.collapse_shape .* into", "memref.alloc() :", out.stdout.decode()
    )
    out = re.sub(r"memref.get_global .* :", "memref.alloc() :", out)
    out = re.sub(r"memref.global .*", "", out)

    open(f"../scripts/braggnn.affine.mlir", "w").write(out)
    open(f"/home/mlevental/dev_projects/circt/hls/braggnn.{dialect}.mlir", "w").write(
        out
    )


BUFFERIZATION_PIPELINE = [
    # "tensor-constant-bufferize{alignment=0}",
    "builtin.func(torch-hls-linalg-bufferize)",
    "builtin.func(linalg-bufferize)",
    "builtin.func(cse)",
    "arith-bufferize",
    "builtin.func(tensor-bufferize)",
    "func-bufferize",
    ## "builtin.func(buffer-hoisting)",
    "builtin.func(buffer-loop-hoisting)",
    ## "builtin.module(buffer-results-to-out-params)",
    "builtin.func(finalizing-bufferize)",
    "builtin.func(cse)",
]

LOWERING_PIPELINE = [
    "builtin.func(cse)",
    "torch-hls-drop-public-return",
    "builtin.func(cse)",
    "builtin.func(convert-linalg-to-loops)",
    # "builtin.func(convert-linalg-to-affine-loops)",
    "builtin.func(convert-linalg-to-parallel-loops)",
    "builtin.func(promote-buffers-to-stack{max-alloc-size-in-bytes=1000000000 max-rank-of-allocated-memref=10})",
    "cse",
]

PIPELINE = (
    [
        "torchscript-module-to-torch-backend-pipeline",
        "torch-backend-to-linalg-on-tensors-backend-pipeline",
    ]
    + BUFFERIZATION_PIPELINE
    + LOWERING_PIPELINE
)
# print('torch-mlir-opt -debug --pass-pipeline"', ",".join(PIPELINE), '"')

if __name__ == "__main__":
    mb = ModuleBuilder()
    recursivescriptmodule, class_annotator = make_small_braggnn()
    print(recursivescriptmodule.graph)
    mb.import_module(recursivescriptmodule._c, class_annotator)

    print(",".join(PIPELINE))
    run_pipeline_with_repro_report(mb.module, ",".join(PIPELINE), "")

    dialect = "scf"
    out = mb.module.operation.get_asm(
        large_elements_limit=100000, enable_debug_info=False
    )
    # hard coded here for vitis
    open(
        f"/home/mlevental/dev_projects/torch-mlir/hls/scripts/{recursivescriptmodule.original_name}.mlir",
        "w",
    ).write(out)
    # hack_for_calyx(out)

# kintex7 kintex7l artix7 artix7l aartix7 zynq azynq spartan7 aspartan7 virtexuplus virtexuplusHBM kintexuplus artixuplusb zynquplus azynquplus kintexu
# set_directive_pipeline
# set_directive_top
# set_directive_unroll
