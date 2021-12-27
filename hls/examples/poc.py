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


# self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
# self.bn1 = norm_layer(self.inplanes)
# self.relu = nn.ReLU(inplace=True)
# self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


class ReLUModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.train(False)

    @export
    @annotate_args([
        None,
        ([10, 10], torch.float32, True),
    ])
    def forward(self, x):
        y = self.relu(x)
        return y


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


def make_relu():
    test_module = ReLUModule()
    class_annotator = ClassAnnotator()
    recursivescriptmodule = torch.jit.script(test_module)
    # print(recursivescriptmodule.graph)
    class_annotator.exportNone(recursivescriptmodule._c._type())
    class_annotator.exportPath(recursivescriptmodule._c._type(), ["forward"])
    class_annotator.annotateArgs(
        recursivescriptmodule._c._type(),
        ["forward"],
        [
            None,
            ([10, 10], torch.float32, True),
        ],
    )
    return recursivescriptmodule._c, class_annotator


PIPELINE = [
    'symbol-dce',
    'torch-prepare-for-globalize-object-graph',
    'torch-globalize-object-graph',
    'symbol-dce',
    'inline',
    'torch-adjust-calling-conventions',
    'builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})',
    'torch-inline-global-slots',
    'builtin.func(torch-reduce-op-variants)',
    'builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})',
    'symbol-dce',
    'builtin.func(torch-hls-refine-types)',
    'torch-refine-public-return',
    'builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})',
    'builtin.func(torch-maximize-value-semantics)',
    'builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})',
    'builtin.func(torch-decompose-complex-ops)',
    'builtin.func(torch-hls-decompose-complex-ops)',
    'torch-verify-invariants-before-backend-lowering',
    'builtin.func(torch-hls-convert-torch-to-linalg)',
    'builtin.func(convert-torch-to-linalg)',
    'builtin.func(convert-torch-to-std)',
    'builtin.func(convert-torch-to-scf)',
    'builtin.func(std-expand)',
    'builtin.func(canonicalize{  max-iterations=10 region-simplify=true top-down=true})',
    'builtin.func(resolve-shaped-type-result-dims)',
    'builtin.func(cse)',
    'torch-func-backend-type-conversion',
    'builtin.func(torch-finalizing-backend-type-conversion)',
    'torch-verify-linalg-on-tensors-backend-contract',
    'tensor-constant-bufferize{alignment=0}',
    'builtin.func(torch-hls-linalg-bufferize)',
    'builtin.func(std-bufferize)',
    'builtin.func(tensor-bufferize)',
    'func-bufferize',
    'builtin.func(finalizing-bufferize)',
    'torch-hls-drop-public-return',
    'builtin.func(convert-linalg-to-loops)',
    'builtin.func(lower-affine)',
    'builtin.func(convert-scf-to-std)',
    'builtin.func(refback-expand-ops-for-llvm)',
    'builtin.func(arith-expand)',
    'builtin.func(convert-math-to-llvm)',
    'convert-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false}',
    'convert-std-to-llvm{data-layout= emit-c-wrappers=false index-bitwidth=0 use-bare-ptr-memref-call-conv=false}',
    'reconcile-unrealized-casts'
]

if __name__ == "__main__":
    mb.import_module(*make_relu())
    with mb.module.context:
        mb.set_multithreading(False)
        pm = PassManager.parse(",".join(PIPELINE))
        pm.enable_ir_printing()
        pm.run(mb.module)

    open(f"relu.llvm.mlir", "w").write(str(mb.module))
