import mlir

# Read a file path, file handle (stream), or a string
ast1 = mlir.parse_path("/home/mlevental/dev_projects/torch-mlir/hls/scripts/individual_layers/BraggNN.1/forward.mlir")
ast2 = mlir.parse_file(open("/home/mlevental/dev_projects/torch-mlir/hls/scripts/individual_layers/theta_phi_g_combine.1/forward.mlir", "r"))
ast3 = mlir.parse_string(
    """
module {
  func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
    %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %t_tensor : tensor<3x2xf64>
  }
}
"""
)
