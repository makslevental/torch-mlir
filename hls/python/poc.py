# import ctypes
# sh_obj = ctypes.cdll.LoadLibrary('/home/mlevental/dev_projects/torch-mlir/build/lib/libruntimePatchLinalg.so')
import torch
# noinspection PyUnresolvedReferences
from torch import nn
from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder, ClassAnnotator
# noinspection PyUnresolvedReferences
from torch_mlir.passmanager import PassManager
# i have no idea why but if you don't import this then linalg passes aren't registered
# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendInvoker
# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export
from torchvision.models.resnet import BasicBlock

from braggnn import BraggNN
from layers import make_layer
from resnet import myresnet18


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


def make_conv():
    mod = torch.nn.Conv2d(
        in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0
    )
    for m in mod.modules():
        if hasattr(m, "weight"):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 1)
    # t = torch.randn((1, 1, 11, 11))
    # y = mod(t)
    mod.train(False)
    return make_layer(
        mod,
        [
            None,
            ([1, 1, 8, 8], torch.float32, True),
        ],
    )

def make_braggnn():
    mod = BraggNN()
    # t = torch.randn((1, 1, 11, 11))
    # y = mod(t)
    mod.train(False)
    return make_layer(
        mod,
        [
            None,
            ([1, 1, 8, 8], torch.float32, True),
        ],
    )


TORCH_PIPELINE = [
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
    ## "builtin.func(torch-hls-refine-types)",
    "builtin.func(torch-refine-types)",
    "torch-refine-public-return",
    "builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})",
    "builtin.func(torch-maximize-value-semantics)",
    "builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})",
    "builtin.func(torch-decompose-complex-ops)",
    "torch-verify-invariants-before-backend-lowering",
    "builtin.module(symbol-dce)",
    ]

TO_LINALG_PIPELINE = [
    ## "builtin.func(convert-torch-hls-to-linalg)",
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
]

BUFFERIZATION_PIPELINE = [
    "tensor-constant-bufferize{alignment=0}",
    "builtin.func(torch-hls-linalg-bufferize)",
    "builtin.func(linalg-bufferize)",
    "builtin.func(cse)",
    "builtin.func(std-bufferize)",
    "builtin.func(tensor-bufferize)",
    "func-bufferize",
    ## "builtin.func(buffer-hoisting)",
    "builtin.func(buffer-loop-hoisting)",
    ## "builtin.module(buffer-results-to-out-params)",
    "builtin.func(finalizing-bufferize)",
    "builtin.func(cse)",
    ]

LOWERING_PIPELINE = [
    "torch-hls-promote-allocs",
    "builtin.func(cse)",
    "torch-hls-drop-public-return",
    "builtin.func(cse)",
    "builtin.func(convert-linalg-to-loops)",
    # "builtin.func(convert-linalg-to-affine-loops)",
    # "builtin.func(parallel-loop-fusion)",
    # "builtin.func(affine-loop-fusion)",
    "builtin.func(lower-affine)",
    # "builtin.func(convert-scf-to-std)",
    # "builtin.func(refback-expand-ops-for-llvm)",
    # "builtin.func(arith-expand)",
    # ## "builtin.func(quant-convert-const)",
    # "builtin.func(convert-math-to-llvm)",
    # "convert-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false}",
    # ## "convert-std-to-llvm{data-layout= emit-c-wrappers=true index-bitwidth=0 use-bare-ptr-memref-call-conv=false}",
    # "convert-std-to-llvm{data-layout= emit-c-wrappers=false index-bitwidth=0 use-bare-ptr-memref-call-conv=true}",
    # "reconcile-unrealized-casts",
]

PIPELINE = TORCH_PIPELINE + TO_LINALG_PIPELINE + BUFFERIZATION_PIPELINE + LOWERING_PIPELINE

if __name__ == "__main__":
    # mod = BraggNN()
    mb = ModuleBuilder()
    mb.import_module(*make_conv())
    # mb = make_traced_mod()
    # mb = make_traced_mod()
    # Verify again with debug info present. Just checking that it makes it in there.
    with mb.module.context:
        mb.set_multithreading(False)
        pm = PassManager.parse(",".join(PIPELINE))
        # pm.enable_ir_printing()
        pm.run(mb.module)

    asm_for_error_report = mb.module.operation.get_asm(
        large_elements_limit=100000, enable_debug_info=False)
    open(f"../scripts/braggnn.affine.mlir", "w").write(asm_for_error_report)

# kintex7 kintex7l artix7 artix7l aartix7 zynq azynq spartan7 aspartan7 virtexuplus virtexuplusHBM kintexuplus artixuplusb zynquplus azynquplus kintexu
# set_directive_pipeline
# set_directive_top
# set_directive_unroll
