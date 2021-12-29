# import ctypes
# sh_obj = ctypes.cdll.LoadLibrary('/home/mlevental/dev_projects/torch-mlir/build/lib/libruntimePatchLinalg.so')

import torch
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.passmanager import PassManager

# i have no idea why but if you don't import this then linalg passes aren't registered
# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendInvoker

from layers import (
    Matmul,
    MatmulDotOut,
    Conv2dNoPaddingOutModule,
    MaxPool2dModule,
    MaxPool2dOutModule,
    AdaptiveAvgPool2dModule,
    AdaptiveAvgPool2dOutModule,
    Conv2dNoPaddingModule,
    ReLUModule, BatchNorm2d, BatchNorm2dOut,
)

mb = ModuleBuilder()


# self.bn1 = norm_layer(self.inplanes)
# self.fc = nn.Linear(512 * block.expansion, num_classes)


def make_layer(test_module, annotations):
    class_annotator = ClassAnnotator()
    recursivescriptmodule = torch.jit.script(test_module)
    class_annotator.exportNone(recursivescriptmodule._c._type())
    class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
    class_annotator.annotateArgs(
        recursivescriptmodule._c._type(), ["forward"], annotations
    )
    return recursivescriptmodule._c, class_annotator


def make_mat_mul():
    return make_layer(
        Matmul(),
        [
            None,
            ([4, 5], torch.float32, True),
            ([5, 10], torch.float32, True),
        ],
    )


def make_mat_mul_out():
    return make_layer(
        MatmulDotOut(),
        [
            None,
            ([4, 5], torch.float32, True),
            ([5, 10], torch.float32, True),
            ([4, 10], torch.float32, True),
        ],
    )


def make_conv2d():
    return make_layer(
        Conv2dNoPaddingModule(),
        [
            None,
            ([5, 2, 10, 20], torch.float32, True),
        ],
    )


def make_conv2d_out():
    return make_layer(
        Conv2dNoPaddingOutModule(),
        [
            None,
            ([5, 2, 10, 20], torch.float32, True),
            ([5, 10, 8, 18], torch.float32, True),
        ],
    )


def make_relu():
    return make_layer(
        ReLUModule()[
            None,
            ([10, 10], torch.float32, True),
        ],
    )


def make_adaptive_avg_pool2d():
    test_module = AdaptiveAvgPool2dModule()
    return make_layer(
        test_module,
        [
            None,
            ([1, 1, 20, 20], torch.float32, True),
        ],
    )


def make_adaptive_avg_pool2d_out():
    test_module = AdaptiveAvgPool2dOutModule()
    return make_layer(
        test_module,
        [
            None,
            ([7, 2, 20, 20], torch.float32, True),
            ([7, 2, 1, 1], torch.float32, True),
        ],
    )


def make_max_pool2d():
    test_module = MaxPool2dModule()
    return make_layer(
        test_module,
        [
            None,
            ([1, 1, 10, 10], torch.float32, True),
        ],
    )


def make_max_pool2d_out():
    test_module = MaxPool2dOutModule()
    return make_layer(
        test_module,
        [
            None,
            ([1, 1, 12, 12], torch.float32, True),
            ([1, 1, 10, 10], torch.float32, True),
            ([1, 1, 10, 10], torch.float32, True),
        ],
    )


def make_bn():
    test_module = BatchNorm2d()
    return make_layer(
        test_module,
        [
            None,
            ([1, 4, 10, 10], torch.float32, True),  # input
        ],
    )

def make_bn_out():
    test_module = BatchNorm2dOut()
    return make_layer(
        test_module,
        [
            None,
            ([1, 4, 10, 10], torch.float32, True),  # input
            ([1, 4, 10, 10], torch.float32, True),  # output
            ([1, 4], torch.float32, True),  # save_mean
            ([1, 4], torch.float32, True),  # save_invstd
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
    "builtin.func(torch-hls-refine-types)",
    "torch-refine-public-return",
    "builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})",
    "builtin.func(torch-maximize-value-semantics)",
    "builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})",
    "builtin.func(torch-decompose-complex-ops)",
    "builtin.func(torch-hls-decompose-complex-ops)",
    "torch-verify-invariants-before-backend-lowering",
    "builtin.func(torch-hls-convert-torch-to-linalg)",
    "builtin.func(convert-torch-to-linalg)",
    "builtin.func(convert-torch-to-std)",
    "builtin.func(convert-torch-to-scf)",
    "builtin.func(std-expand)",
    "builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})",
    "builtin.func(resolve-shaped-type-result-dims)",
    "builtin.func(cse)",
    "torch-func-backend-type-conversion",
    "builtin.func(torch-finalizing-backend-type-conversion)",
    "torch-verify-linalg-on-tensors-backend-contract",
    "tensor-constant-bufferize{alignment=0}",
    "builtin.func(torch-hls-linalg-bufferize)",
    "builtin.func(std-bufferize)",
    "builtin.func(tensor-bufferize)",
    "func-bufferize",
    "builtin.func(finalizing-bufferize)",
    "torch-hls-drop-public-return",
    "builtin.func(convert-linalg-to-loops)",
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
    mb.import_module(*make_bn_out())
    with mb.module.context:
        mb.set_multithreading(False)
        pm = PassManager.parse(",".join(PIPELINE))
        pm.enable_ir_printing()
        pm.run(mb.module)

    open(f"batchnorm2d.llvm.mlir", "w").write(str(mb.module))
