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

// JitOperator 'aten::max_pool2d_with_indices.out : (Tensor, int[], int[], int[], int[], bool, Tensor, Tensor) -> (Tensor, Tensor)':
// MLIR op name = torch.aten.max_pool2d_with_indices.out
// MLIR td def name = Torch_AtenMaxPool2dWithIndicesOutOp

def HLSTorch_AtenMaxPool2dWithIndicesOutOp : Torch_Op<"aten.max_pool2d_with_indices.out", [
    AllowsTypeRefinement,
    HasValueSemantics
  ]> {
  let summary = "aten::max_pool2d_with_indices.out : (Tensor, int[], int[], int[], int[], bool, Tensor, Tensor) -> (Tensor, Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    TorchIntListType:$kernel_size,
    TorchIntListType:$stride,
    TorchIntListType:$padding,
    TorchIntListType:$dilation,
    Torch_BoolType:$ceil_mode,
    AnyTorchTensorType:$out,
    AnyTorchTensorType:$indices
  );
  let results = (outs
    AnyTorchTensorType:$out_result,
    AnyTorchTensorType:$indices_result
  );
  let assemblyFormat = "$self `,` $kernel_size `,` $stride `,` $padding `,` $dilation `,` $ceil_mode attr-dict `,` $out `,` $indices `:` type($self) `,` type($kernel_size) `,` type($stride) `,` type($padding) `,` type($dilation) `,` type($ceil_mode) `,` type($out) `,` type($indices) `->` type($out_result) `,` type($indices_result)";
}

def HLSTorch_AtenAdaptiveAvgPool2dOutOp : Torch_Op<"aten.adaptive_avg_pool2d.out", [
    AllowsTypeRefinement,
    HasValueSemantics
  ]> {
  let summary = "Generated op for `aten::adaptive_avg_pool2d.out : (Tensor, int[], Tensor) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    TorchIntListType:$output_size,
    AnyTorchTensorType:$out
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$self `,` $output_size attr-dict `,` $out `:` type($self) `,` type($output_size) `,` type($out) `->` type($result)";
}


def HLSTorch_AtenNativeBatchNormOutOp : Torch_Op<"aten.native_batch_norm.out", [
    AllowsTypeRefinement,
    HasValueSemantics
]> {
let summary = "Generated op for `aten::native_batch_norm.out : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float, Tensor, Tensor, Tensor) -> (Tensor, Tensor, Tensor)`";
let arguments = (ins
    AnyTorchTensorType:$input,
    AnyTorchOptionalTensorType:$weight,
    AnyTorchOptionalTensorType:$bias,
    AnyTorchOptionalTensorType:$running_mean,
    AnyTorchOptionalTensorType:$running_var,
    Torch_BoolType:$training,
    Torch_FloatType:$momentum,
    Torch_FloatType:$eps,
    AnyTorchOptionalTensorType:$out,
    AnyTorchOptionalTensorType:$save_mean,
    AnyTorchOptionalTensorType:$save_invstd
  );
  let results = (outs
    AnyTorchTensorType:$result_out,
    AnyTorchTensorType:$result_save_mean,
    AnyTorchTensorType:$result_save_invstd
  );
  let assemblyFormat = "$input `,` $weight `,` $bias `,` $running_mean `,` $running_var `,` $training `,` $momentum `,` $eps `,` $out `,` $save_mean `,` $save_invstd attr-dict `:` type($input) `,` type($weight) `,` type($bias) `,` type($running_mean) `,` type($running_var) `,` type($training) `,` type($momentum) `,` type($eps) `,` type($out) `,` type($save_mean) `,` type($save_invstd) `->` type($result_out) `,` type($result_save_mean) `,` type($result_save_invstd)";
}

def HLSTorch_AtenLinearOutOp : Torch_Op<"aten.linear.out", [
  AllowsTypeRefinement,
  HasValueSemantics
]> {
  let summary = "Generated op for `aten::linear.out : (Tensor, Tensor, Tensor?, Tensor) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$input,
    AnyTorchTensorType:$weight,
    AnyTorchOptionalTensorType:$bias,
    AnyTorchTensorType:$out
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$input `,` $weight `,` $bias `,` $out attr-dict `:` type($input) `,` type($weight) `,` type($bias) `,` type($out) `->` type($result)";
}

def HLSTorch_AtenAddOutTensorOp : Torch_Op<"aten.add.out", [
    AllowsTypeRefinement,
    HasValueSemantics
  ]> {
  let summary = "Generated op for `aten::add.out : (Tensor, Tensor, Scalar, Tensor) -> (Tensor)`";
  let arguments = (ins
    AnyTorchTensorType:$self,
    AnyTorchTensorType:$other,
    AnyTorchScalarType:$alpha,
    AnyTorchTensorType:$out
  );
  let results = (outs
    AnyTorchTensorType:$result
  );
  let assemblyFormat = "$self `,` $other `,` $alpha `,` $out attr-dict `:` type($self) `,` type($other) `,` type($alpha) `,` type($out) `->` type($result)";
}



#endif // HLSTORCH_OPS