# import ctypes
# sh_obj = ctypes.cdll.LoadLibrary('/home/mlevental/dev_projects/torch-mlir/build/lib/libruntimePatchLinalg.so')

import torch

from torch_mlir_e2e_test.linalg_on_tensors_backends.abc import LinalgOnTensorsBackend
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendInvoker
from torch_mlir_e2e_test.torchscript.annotations import annotate_args
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.torchscript.annotations import export
from torch_mlir_e2e_test.utils import run_pipeline_with_repro_report

mb = ModuleBuilder()


class Matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 5], torch.float32, True),
        ([5, 10], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.mm(lhs, rhs)


class MatmulDotOut(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 5], torch.float32, True),
        ([5, 10], torch.float32, True),
        ([4, 10], torch.float32, True),
    ])
    def forward(self, lhs, rhs, out):
        return torch.mm(lhs, rhs, out=out)


class RefBackendLinalgOnTensorsBackend(LinalgOnTensorsBackend):
    def __init__(self):
        super().__init__()

    def compile(self, imported_module):
        run_pipeline_with_repro_report(
            imported_module, LOWERING_PIPELINE,
            "Lowering Linalg-on-Tensors IR to LLVM with RefBackend")
        return imported_module

    def load(self, module) -> RefBackendInvoker:
        return RefBackendInvoker(module)


def make_mat_mul():
    test_module = Matmul()
    class_annotator = ClassAnnotator()
    recursivescriptmodule = torch.jit.script(test_module)
    class_annotator.exportNone(recursivescriptmodule._c._type())
    class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
    class_annotator.annotateArgs(
        recursivescriptmodule._c._type(),
        ["forward"],
        [
            None,
            ([4, 5], torch.float32, True),
            ([5, 10], torch.float32, True),
        ],
    )
    return recursivescriptmodule._c, class_annotator


def make_mat_mul_out():
    test_module = MatmulDotOut()
    class_annotator = ClassAnnotator()
    recursivescriptmodule = torch.jit.script(test_module)
    class_annotator.exportNone(recursivescriptmodule._c._type())
    class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
    class_annotator.annotateArgs(
        recursivescriptmodule._c._type(),
        ["forward"],
        [
            None,
            ([4, 5], torch.float32, True),
            ([5, 10], torch.float32, True),
            ([4, 10], torch.float32, True),
        ],
    )
    return recursivescriptmodule._c, class_annotator


LOWERING_PIPELINE = ",".join([
    # Bufferize.

    "tensor-constant-bufferize",
    "builtin.func(scf-bufferize)",
    "builtin.func(torch-hls-linalg-bufferize)",
    "builtin.func(std-bufferize)",
    "builtin.func(tensor-bufferize)",
    "func-bufferize",
    # "buffer-results-to-out-params",
    "builtin.func(finalizing-bufferize)",
    "torch-hls-drop-public-return",

    # Lower to LLVM

    "builtin.func(convert-linalg-to-loops)",
    "builtin.func(lower-affine)",
    "builtin.func(convert-scf-to-std)",
    "builtin.func(refback-expand-ops-for-llvm)",
    "builtin.func(arith-expand)",
    "builtin.func(convert-math-to-llvm)",
    "convert-memref-to-llvm",
    "convert-std-to-llvm",
    "reconcile-unrealized-casts",
])

if __name__ == "__main__":
    mb.import_module(*make_mat_mul_out())

    backend = RefBackendLinalgOnTensorsBackend()
    with mb.module.context:
        pm = PassManager.parse(",".join([
            'torchscript-module-to-torch-hls-backend-pipeline',
            'torch-hls-backend-to-linalg-on-tensors-backend-pipeline'
        ]))
        pm.run(mb.module)

    compiled = backend.compile(mb.module)
    open(f"matmul.llvm.mlir", "w").write(str(mb.module))
