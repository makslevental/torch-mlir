import inspect
import argparse

import os
import pathlib
from pathlib import Path
import re
import shutil
import stat
import subprocess

import torch

# noinspection PyUnresolvedReferences
from torch import nn
from torchvision.models.resnet import BasicBlock

from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder

# noinspection PyUnresolvedReferences
from torch_mlir.passmanager import PassManager

# i have no idea why but if you don't import this then linalg passes aren't registered
# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendInvoker

# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export
from torch_mlir_e2e_test.utils import run_pipeline_with_repro_report

from hls.python.models.braggnn import (
    BraggNN,
    cnn_layers_1,
    cnn_layers_2,
    dense_layers,
    theta_phi_g_combine,
)
from hls.python.models.layers import make_layer
from hls.python.models.resnet18 import ResNet18


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
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, bias=False
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=3, bias=False
        )
        self.dense = torch.nn.Linear(in_features=1568, out_features=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
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


def make_braggnn_mod(scale=4, imgsz=11):
    with torch.no_grad():
        mod = BraggNN(imgsz=imgsz, scale=scale)
        mod.eval()
        t = torch.randn((1, 1, imgsz, imgsz))
        y = mod(t)
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
            "/home/maksim/dev_projects/torch-mlir/cmake-build-debug/bin/torch-mlir-opt",
            "-pass-pipeline=reconcile-unrealized-casts",
            "/home/maksim/dev_projects/torch-mlir/hls/scripts/braggnn.affine.mlir",
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
    open(f"/home/maksim/dev_projects/circt/hls/braggnn.{dialect}.mlir", "w").write(out)


BUFFERIZATION_PIPELINE = [
    # Bufferize.
    "func.func(scf-bufferize)",
    "func.func(tm-tensor-bufferize)",
    # "func.func(torch-hls-linalg-bufferize)",
    "func.func(linalg-bufferize)",
    "func-bufferize",
    "arith-bufferize",
    "func.func(tensor-bufferize)",
    "func.func(buffer-loop-hoisting)",
    "func.func(finalizing-bufferize)",
    # "tensor-constant-bufferize{alignment=0}",
    ## "func.func(buffer-hoisting)",
    ## "builtin.module(buffer-results-to-out-params)",
    "func.func(finalizing-bufferize)",
]

LOWERING_PIPELINE = [
    "func.func(cse)",
    # TODO: use this correctly in promoteallocs
    "torch-hls-drop-public-return",
    "func.func(cse)",
    # "func.func(convert-linalg-to-loops)",
    # "func.func(convert-linalg-to-affine-loops)",
    "func.func(convert-linalg-to-parallel-loops)",
    # "func.func(lower-affine)",
    "func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=1000000000 max-rank-of-allocated-memref=10})",
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

import numpy as np


def make_wrapper(wrapper_str, in_shape, out_shape):
    array_in_shapes = "".join([f"[{s}]" for s in in_shape])
    array_out_shapes = "".join([f"[{s}]" for s in out_shape])
    XXX_EXTERN_C_XXX = f'extern "C" void forward(float (&arg_2){array_in_shapes}, float (&arg_3){array_out_shapes});'
    wrapper_str = wrapper_str.replace("XXX_EXTERN_C_XXX", XXX_EXTERN_C_XXX)
    # XXX_DTYPE_XXX = 'float'

    XXX_L_A_XXX = f"float l_A{array_in_shapes};"
    wrapper_str = wrapper_str.replace("XXX_L_A_XXX", XXX_L_A_XXX)
    XXX_L_C_XXX = f"float l_C{array_out_shapes};"
    wrapper_str = wrapper_str.replace("XXX_L_C_XXX", XXX_L_C_XXX)

    XXX_I_LIMIT_XXX = (
        f"int i_limit = ceil((float)({np.prod(in_shape)}) / (float)NUM_ITEMS);"
    )
    wrapper_str = wrapper_str.replace("XXX_I_LIMIT_XXX", XXX_I_LIMIT_XXX)
    XXX_REF_L_A_XXX = f"(&l_A{'[0]' * len(in_shape)})[index] = converter.d;"
    wrapper_str = wrapper_str.replace("XXX_REF_L_A_XXX", XXX_REF_L_A_XXX)

    XXX_K_LIMIT_XXX = (
        f"int k_limit = ceil((float)({np.prod(out_shape)}) / (float)NUM_ITEMS);"
    )
    wrapper_str = wrapper_str.replace("XXX_K_LIMIT_XXX", XXX_K_LIMIT_XXX)
    XXX_REF_L_C_XXX = f"converter.d = (&l_C{'[0]' * len(out_shape)})[index];"
    wrapper_str = wrapper_str.replace("XXX_REF_L_C_XXX", XXX_REF_L_C_XXX)

    return wrapper_str


def put_script_files(*, out_str, in_shape, out_shape, out_dir, forward_suffix=""):
    out_str = re.sub(r"cf.assert .*", "", out_str)

    os.makedirs(f"{out_dir}", exist_ok=True)
    open(f"{out_dir}/forward{forward_suffix}.mlir", "w").write(out_str)

    # wrapper_str = open("wrapper.cpp.fmt", "r").read()
    # wrapper_str = make_wrapper(wrapper_str, in_shape, out_shape)
    # open(f"{out_dir}/wrapper.cpp", "w").write(wrapper_str)

    # run_hls = open("run_vitis.tcl", "r").read()
    # run_hls = run_hls.replace("XXX_DIR_XXX", out_dir)
    # run_hls = run_hls.replace("XXX_LL_FILE_XXX", "forward.ll")
    # open(f"{out_dir}/run_vitis.tcl", "w").write(run_hls)

    # hls_hooks = open("hls_hooks.tcl", "r").read()
    # hls_hooks = hls_hooks.replace("XXX_DIR_XXX", out_dir)
    # hls_hooks = hls_hooks.replace("XXX_LL_FILE_XXX", "forward.ll")
    # open(f"{out_dir}/hls_hooks.tcl", "w").write(hls_hooks)

    # shutil.copyfile("Makefile", f"{out_dir}/Makefile")
    # shutil.copyfile("read_large_file.py", f"{out_dir}/read_large_file.py")
    # shutil.copyfile("run_vitis.sh", f"{out_dir}/run_vitis.sh")
    # st = os.stat(f"{out_dir}/run_vitis.sh")
    # os.chmod(f"{out_dir}/run_vitis.sh", st.st_mode | stat.S_IEXEC)
    # shutil.copyfile("mlir_ops.py", f"{out_dir}/mlir_ops.py")
    # shutil.copyfile("llvm_val.py", f"{out_dir}/llvm_val.py")
    # shutil.copyfile("verilog_val.py", f"{out_dir}/verilog_val.py")
    shutil.copyfile("translate.sh", f"{out_dir}/translate.sh")
    st = os.stat(f"{out_dir}/translate.sh")
    os.chmod(f"{out_dir}/translate.sh", st.st_mode | stat.S_IEXEC)

    # shutil.copyfile("make_schedule.py", f"{out_dir}/make_schedule.py")
    shutil.copyfile("run_vivado.tcl", f"{out_dir}/run_vivado.tcl")
    shutil.copyfile("run_vivado.sh", f"{out_dir}/run_vivado.sh")
    st = os.stat(f"{out_dir}/run_vivado.sh")
    os.chmod(f"{out_dir}/run_vivado.sh", st.st_mode | stat.S_IEXEC)


def make_sub_layer(mod, in_shapes, out_shape, scale, root_out_dir):
    recursivescriptmodule, class_annotator = make_layer(
        mod, [None] + [(in_shape, torch.float32, True) for in_shape in in_shapes]
    )

    mb = ModuleBuilder()
    mb.import_module(recursivescriptmodule._c, class_annotator)
    run_pipeline_with_repro_report(mb.module, ",".join(PIPELINE), "")

    out = mb.module.operation.get_asm(
        large_elements_limit=100000, enable_debug_info=False
    )

    out_dir = str(
        root_out_dir / Path(recursivescriptmodule.original_name + f".{scale}")
    )
    put_script_files(
        out_str=out, in_shape=in_shapes[0], out_shape=out_shape, out_dir=out_dir
    )


def make_module_artifacts(
    mod,
    example_input,
    example_output,
    recursivescriptmodule,
    class_annotator,
    root_out_dir,
    scale,
):
    out_dir = str(
        root_out_dir / Path(recursivescriptmodule.original_name + f".{scale}")
    )
    os.makedirs(out_dir, exist_ok=True)
    f = open(f"{out_dir}/forward.ts.mlir", "w")
    f.write(str(recursivescriptmodule.graph))
    f.close()

    mb = ModuleBuilder()
    mb.import_module(recursivescriptmodule._c, class_annotator)
    run_pipeline_with_repro_report(
        mb.module,
        ",".join(
            [
                "torchscript-module-to-torch-backend-pipeline",
                "torch-backend-to-linalg-on-tensors-backend-pipeline",
            ]
        ),
        "",
    )
    out = mb.module.operation.get_asm(
        large_elements_limit=100000, enable_debug_info=False
    )
    open(f"{out_dir}/forward.linalg.mlir", "w").write(out)

    mb = ModuleBuilder()
    mb.import_module(recursivescriptmodule._c, class_annotator)
    run_pipeline_with_repro_report(mb.module, ",".join(PIPELINE), "")
    out = mb.module.operation.get_asm(
        large_elements_limit=100000, enable_debug_info=False
    )

    put_script_files(
        out_str=out,
        in_shape=tuple(example_output.shape),
        out_shape=tuple(example_output.shape),
        out_dir=out_dir,
    )
    open(f"{out_dir}/mod.txt", "w").write(str(mod))


def make_whole_braggnn(root_out_dir, scale=4, imgsz=11, simplify_weights=False):
    with torch.no_grad():
        mod = BraggNN(imgsz=imgsz, scale=scale)
        mod.eval()
        t = torch.randn((1, 1, imgsz, imgsz))
        y = mod(t)
        if simplify_weights:
            mod.apply(set_weights)
        recursivescriptmodule, class_annotator = make_layer(
            mod, [None, ([1, 1, imgsz, imgsz], torch.float32, True)]
        )

    make_module_artifacts(
        mod, t, y, recursivescriptmodule, class_annotator, root_out_dir, scale
    )


class ConvPlusReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = torch.nn.Conv2d(out_channels, in_channels, 3)
        # self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.conv2(self.conv1(x))


def make_single_small_cnn(
    root_out_dir, in_channels=2, out_channels=4, imgsz=11, simplify_weights=False
):
    with torch.no_grad():
        mod = ConvPlusReLU(in_channels, out_channels)
        mod.eval()
        t = torch.randn((1, in_channels, imgsz, imgsz))
        y = mod(t)
        if simplify_weights:
            mod.apply(set_weights)
        recursivescriptmodule, class_annotator = make_layer(
            mod, [None, ([1, in_channels, imgsz, imgsz], torch.float32, True)]
        )

    make_module_artifacts(
        mod, t, y, recursivescriptmodule, class_annotator, root_out_dir, out_channels
    )


class DoubleCNN(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16 * scale, 3)
        self.conv2_1 = torch.nn.Conv2d(16 * scale, 8 * scale, 1)
        self.conv2_2 = torch.nn.Conv2d(16 * scale, 8 * scale, 1)
        self.conv2_3 = torch.nn.Conv2d(16 * scale, 8 * scale, 1)
        self.conv3 = torch.nn.Conv2d(8 * scale, 16 * scale, 1)
        self.conv4 = torch.nn.Conv2d(16 * scale, 8 * scale, 3)

    def forward(self, x):
        y = self.conv1(x)
        z = self.conv2_1(y)
        w = self.conv2_2(y)
        u = self.conv2_3(y)

        uu = self.conv3(z + w + u)
        ww = self.conv4(uu)
        return ww


def make_double_small_cnn(root_out_dir, scale=1, imgsz=11, simplify_weights=False):
    with torch.no_grad():
        mod = DoubleCNN(scale)
        mod.eval()
        t = torch.randn((1, 1, imgsz, imgsz))
        y = mod(t)
        if simplify_weights:
            mod.apply(set_weights)
        recursivescriptmodule, class_annotator = make_layer(
            mod, [None, ([1, 1, imgsz, imgsz], torch.float32, True)]
        )

    make_module_artifacts(
        mod, t, y, recursivescriptmodule, class_annotator, root_out_dir, scale
    )


class Linear(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 4)
        # self.linear2 = torch.nn.Linear(4, 2)

    def forward(self, x):
        y = self.linear1(x)
        # z = self.linear2(y)
        return y.sum()


def make_linear(root_out_dir, scale=1, imgsz=11, simplify_weights=False):
    with torch.no_grad():
        mod = Linear(scale)
        mod.eval()
        t = torch.randn((2, 3))
        y = mod(t)
        if simplify_weights:
            mod.apply(set_weights)
        recursivescriptmodule, class_annotator = make_layer(
            mod, [None, ([2, 3], torch.float32, True)]
        )

    make_module_artifacts(
        mod, t, y, recursivescriptmodule, class_annotator, root_out_dir, scale
    )


def main():
    parser = argparse.ArgumentParser(description="make stuff")
    parser.add_argument("--low_scale", type=int, default=1)
    parser.add_argument("--high_scale", type=int, default=5)
    parser.add_argument("--out_dir", type=Path, default=Path("../examples"))
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()

    # make_single_small_cnn(
    #     args.out_dir, in_channels=2, out_channels=4, imgsz=7, simplify_weights=False
    # )
    # make_linear(args.out_dir, imgsz=5, simplify_weights=False)
    make_double_small_cnn(args.out_dir, scale=1, imgsz=11, simplify_weights=False)
    # for i in range(args.low_scale, args.high_scale):
    #     make_whole_braggnn(args.out_dir, scale=i, imgsz=11, simplify_weights=False)


if __name__ == "__main__":
    main()
