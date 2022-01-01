# import ctypes
# sh_obj = ctypes.cdll.LoadLibrary('/home/mlevental/dev_projects/torch-mlir/build/lib/libruntimePatchLinalg.so')
import torch
from torch import nn
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.passmanager import PassManager

# i have no idea why but if you don't import this then linalg passes aren't registered
# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendInvoker
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import (
    extract_annotations,
)

from torch_mlir_e2e_test.torchscript.annotations import export
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock

from braggnn import BraggNN
from resnet import myresnet18
from layers import make_layer, TestMod, MyBasicBlock


def make_resnet():
    mod = myresnet18()
    mod.train(False)
    return make_layer(
        mod,
        [
            None,
            ([1, 3, 32, 32], torch.float32, True),
        ],
    )


def make_mod():
    mod = BasicBlock(2, 2)
    mod.train(False)
    return make_layer(
        mod,
        [
            None,
            ([1, 2, 8, 8], torch.float32, True),
        ],
    )

def make_braggnn():
    mod = BraggNN()
    t = torch.randn((1, 1, 11, 11))
    y = mod(t)
    mod.train(False)
    return make_layer(
        mod,
        [
            None,
            ([1, 1, 11, 11], torch.float32, True),
        ],
    )


PIPELINE = [
    "symbol-dce",
    "torch-prepare-for-globalize-object-graph",
    "torch-globalize-object-graph",
    "symbol-dce",
    "inline",
    "torch-adjust-calling-conventions",
    "builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})",
    "torch-inline-global-slots",
    "builtin.func(torch-reduce-op-variants)",
    "builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})",
    "symbol-dce",
    # "builtin.func(torch-hls-refine-types)",
    "builtin.func(torch-refine-types)",
    "torch-refine-public-return",
    "builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})",
    "builtin.func(torch-maximize-value-semantics)",
    "builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})",
    "builtin.func(torch-decompose-complex-ops)",
    "torch-verify-invariants-before-backend-lowering",
    "builtin.module(symbol-dce)",
    # "builtin.func(convert-torch-hls-to-linalg)",
    "builtin.func(convert-torch-to-linalg)",
    "builtin.func(convert-torch-to-std)",
    "builtin.func(convert-torch-to-scf)",
    "builtin.func(linalg-strategy-tile-and-fuse-pass)",
    "builtin.func(std-expand)",
    "builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})",
    "builtin.func(resolve-shaped-type-result-dims)",
    "builtin.func(cse)",
    "torch-func-backend-type-conversion",
    "builtin.func(torch-finalizing-backend-type-conversion)",
    "torch-verify-linalg-on-tensors-backend-contract",
    "tensor-constant-bufferize{alignment=0}",
    "builtin.func(linalg-detensorize)",
    "builtin.module(linalg-comprehensive-module-bufferize)",
    "builtin.func(linalg-bufferize)",
    "builtin.func(std-bufferize)",
    "builtin.func(tensor-bufferize)",
    "func-bufferize",
    "builtin.func(buffer-hoisting)",
    "builtin.func(buffer-loop-hoisting)",
    "builtin.module(buffer-results-to-out-params)",
    "builtin.func(finalizing-bufferize)",
    "builtin.func(cse)",
    "torch-hls-promote-allocs",
    "builtin.func(cse)",
    "torch-hls-drop-public-return",
    "builtin.func(cse)",
    "builtin.func(convert-linalg-to-loops)",
    # "parallel-loop-fusion"
    "builtin.func(lower-affine)",
    "builtin.func(convert-scf-to-std)",
    "builtin.func(refback-expand-ops-for-llvm)",
    "builtin.func(arith-expand)",
    "builtin.func(convert-math-to-llvm)",
    "convert-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false}",
    "convert-std-to-llvm{data-layout= emit-c-wrappers=false index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
    "reconcile-unrealized-casts",
]

if __name__ == "__main__":
    # mod = BraggNN()
    mb = ModuleBuilder()
    mb.import_module(*make_braggnn())
    with mb.module.context:
        mb.set_multithreading(False)
        pm = PassManager.parse(",".join(PIPELINE))
        pm.enable_ir_printing()
        pm.run(mb.module)

    open(f"braggnn.llvm.mlir", "w").write(str(mb.module))

# def make_mat_mul():
#     return make_layer(
#         Matmul(),
#         [
#             None,
#             ([4, 5], torch.float32, True),
#             ([5, 10], torch.float32, True),
#         ],
#     )
#
#
# def make_mat_mul_out():
#     return make_layer(
#         MatmulDotOut(),
#         [
#             None,
#             ([4, 5], torch.float32, True),
#             ([5, 10], torch.float32, True),
#             ([4, 10], torch.float32, True),
#         ],
#     )
#
#
# def make_conv2d():
#     return make_layer(
#         Conv2dNoPaddingModule(),
#         [
#             None,
#             ([5, 2, 10, 20], torch.float32, True),
#         ],
#     )
#
#
# def make_conv2d_out():
#     return make_layer(
#         Conv2dNoPaddingOutModule(),
#         [
#             None,
#             ([5, 2, 10, 20], torch.float32, True),
#             ([5, 10, 8, 18], torch.float32, True),
#         ],
#     )
#
#
# def make_relu():
#     return make_layer(
#         ReLUModule()[
#             None,
#             ([10, 10], torch.float32, True),
#         ],
#     )
#
#
# def make_adaptive_avg_pool2d():
#     test_module = AdaptiveAvgPool2dModule()
#     return make_layer(
#         test_module,
#         [
#             None,
#             ([1, 1, 20, 20], torch.float32, True),
#         ],
#     )
#
#
# def make_adaptive_avg_pool2d_out():
#     test_module = AdaptiveAvgPool2dOutModule()
#     return make_layer(
#         test_module,
#         [
#             None,
#             ([7, 2, 20, 20], torch.float32, True),
#             ([7, 2, 1, 1], torch.float32, True),
#         ],
#     )
#
#
# def make_max_pool2d():
#     test_module = MaxPool2dModule()
#     return make_layer(
#         test_module,
#         [
#             None,
#             ([1, 1, 10, 10], torch.float32, True),
#         ],
#     )
#
#
# def make_max_pool2d_out():
#     test_module = MaxPool2dOutModule()
#     return make_layer(
#         test_module,
#         [
#             None,
#             ([1, 1, 12, 12], torch.float32, True),
#             ([1, 1, 10, 10], torch.float32, True),
#             ([1, 1, 10, 10], torch.float32, True),
#         ],
#     )
#
#
# def make_bn():
#     test_module = BatchNorm2d()
#     return make_layer(
#         test_module,
#         [
#             None,
#             ([1, 4, 10, 10], torch.float32, True),  # input
#         ],
#     )
#
# def make_bn_out():
#     test_module = BatchNorm2dOut()
#     return make_layer(
#         test_module,
#         [
#             None,
#             ([1, 4, 10, 10], torch.float32, True),  # input
#             ([1, 4, 10, 10], torch.float32, True),  # output
#             ([1, 4], torch.float32, True),  # save_mean
#             ([1, 4], torch.float32, True),  # save_invstd
#         ],
#     )
#
# def make_linear():
#     test_module = Linear()
#     return make_layer(
#         test_module,
#         [
#             None,
#             ([8, 512], torch.float32, True),  # input
#         ],
#     )
#
# def make_linear_out_():
#     test_module = make_linear_out(8, 512, 1000)
#     return make_layer(
#         test_module,
#         [
#             None,
#             ([8, 512], torch.float32, True),  # input
#             ([8, 1000], torch.float32, True),  # output
#         ],
#     )
