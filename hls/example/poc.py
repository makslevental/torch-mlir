# import ctypes
# sh_obj = ctypes.cdll.LoadLibrary('/home/mlevental/dev_projects/torch-mlir/build/lib/libruntimePatchLinalg.so')

import torch
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.passmanager import PassManager
# i have no idea why but if you don't import this then linalg passes aren't registered
# noinspection PyUnresolvedReferences
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendInvoker
from torch_mlir_e2e_test.torchscript.annotations import annotate_args
from torch_mlir_e2e_test.torchscript.annotations import export

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


class Conv2dNoPaddingOutModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.kernel_size = (3, 3)
        conv = torch.nn.Conv2d(2, 10, self.kernel_size, bias=False)
        self.weight = conv.weight
        self.train(False)

    @export
    @annotate_args([
        None,
        ([5, 2, 10, 20], torch.float32, True),
        ([5, 10, 8, 18], torch.float32, True),
    ])
    def forward(self, inp, out):
        return torch._C._nn.thnn_conv2d(inp, self.weight, self.kernel_size, out=out)


class Conv2dNoPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 10, 3, bias=False)
        self.train(False)

    @export
    @annotate_args([
        None,
        ([5, 2, 10, 20], torch.float32, True),
    ])
    def forward(self, x):
        return self.conv(x)


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


def make_conv2d():
    test_module = Conv2dNoPaddingModule()
    class_annotator = ClassAnnotator()
    recursivescriptmodule = torch.jit.script(test_module)
    class_annotator.exportNone(recursivescriptmodule._c._type())
    class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
    class_annotator.annotateArgs(
        recursivescriptmodule._c._type(),
        ["forward"],
        [
            None,
            ([5, 2, 10, 20], torch.float32, True),
        ],
    )
    return recursivescriptmodule._c, class_annotator

def make_conv2d_out():
    test_module = Conv2dNoPaddingOutModule()
    # conv = torch.nn.Conv2d(2, 10, (3, 3), bias=False)
    # t = torch.randn((5, 2, 10, 20))
    # y = conv(t)
    # print(y.shape)
    class_annotator = ClassAnnotator()
    recursivescriptmodule = torch.jit.script(test_module)
    class_annotator.exportNone(recursivescriptmodule._c._type())
    class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
    class_annotator.annotateArgs(
        recursivescriptmodule._c._type(),
        ["forward"],
        [
            None,
            ([5, 2, 10, 20], torch.float32, True),
            ([5, 10, 8, 18], torch.float32, True),
        ],
    )
    return recursivescriptmodule._c, class_annotator


if __name__ == "__main__":
    mb.import_module(*make_conv2d_out())
    with mb.module.context:
        pm = PassManager.parse(",".join([
            'torchscript-module-to-torch-hls-backend-pipeline',
            'torch-hls-backend-to-linalg-on-tensors-backend-pipeline',
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
        ]))
        pm.run(mb.module)

    open(f"conv2d.llvm.mlir", "w").write(str(mb.module))
