#ifndef HLSTORCH_OPS
#define HLSTORCH_OPS

// include "mlir/IR/OpAsmInterface.td"
// include "mlir/IR/SymbolInterfaces.td"
// include "mlir/Interfaces/CastInterfaces.td"
// include "mlir/Interfaces/ControlFlowInterfaces.td"
// include "mlir/Interfaces/InferTypeOpInterface.td"
// include "mlir/Interfaces/SideEffectInterfaces.td"
// include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"
//
//
// class Torch_Op<string mnemonic, list<OpTrait> traits = []>
//     : Op<Torch_Dialect, mnemonic, traits> {
// }

def HLSTorch_AtenMmOutOp : Torch_Op<"aten.mm.out", [
    HasValueSemantics,
    AllowsTypeRefinement,
    // IsOutOfPlaceVariant
  ]> {
  let summary = "Generated op for `aten::mm.out : (Tensor, Tensor, Tensor) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    AnyTorchTensorType:$mat2,
    AnyTorchTensorType:$out
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$self `,` $mat2 `,` $out attr-dict `:` type($self) `,` type($mat2) `,` type($out) `->` type($result)";
}

def HLSTorch_AtenMatmulOutOp : Torch_Op<"aten.matmul.out", [
    HasValueSemantics,
    AllowsTypeRefinement,
    // IsOutOfPlaceVariant
  ]> {
  let summary = "Generated op for `aten::matmul.out : (Tensor, Tensor, Tensor) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    AnyTorchTensorType:$other,
    AnyTorchTensorType:$out
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$self `,` $other `,` $out attr-dict `:` type($self) `,` type($other) `,` type($out) `->` type($result)";
}

// torch.operator "aten.thnn_conv2d.out"(%1, %5, %6, %none, %4, %3, %2) : (!torch.tensor<[5,2,10,20],f32>, !torch.tensor<[10,2,3,3],f32>, !torch.list<!torch.int>, !torch.none, !torch.list<!torch.int>, !torch.list<!torch.int>, !torch.tensor<[5,10,8,18],f32>) -> !torch.tensor

def HLSTorch_AtenTHNNConv2dOutOp : Torch_Op<"aten.thnn_conv2d.out", [
    AllowsTypeRefinement,
    HasValueSemantics
  ]> {
  let summary = "Generated op for `aten.thnn_conv2d.out : (Tensor, Tensor, int[], Tensor?, int[], int[], Tensor) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$input,
    AnyTorchTensorType:$weight,
    TorchIntListType:$kernel_size,
    AnyTorchOptionalTensorType:$bias,
    TorchIntListType:$stride,
    TorchIntListType:$padding,
    AnyTorchTensorType:$out
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$input `,` $weight `,` $kernel_size `,` $bias `,` $stride `,` $padding `,` $out attr-dict `:` type($input) `,` type($weight) `,` type($kernel_size) `,` type($bias) `,` type($stride) `,` type($padding) `,` type($out) `->` type($result)";
}

#endif // HLSTORCH_OPS