import numpy as np
from completely_unroll_to_llvm_ir import ArrayDecl, Global, Exp, Forward, FMulAdd


# fmt: off
__constant_16x1x3x3xf32 = np.array([-2.484830e-02, 1.143797e-01, -9.939806e-02, -2.908628e-01, 1.772281e-02, -1.944778e-01, 3.081904e-01, -1.707404e-01, -4.625805e-02, 9.391245e-02, -3.153162e-01, 2.696663e-01, -2.284244e-01, 8.857012e-03, 2.876089e-01, -2.335219e-01, 6.235620e-02, 8.323681e-02, 2.641745e-01, -2.860002e-01, -9.677748e-02, 2.158663e-01, 6.517756e-02, 1.991838e-01, -9.814525e-02, 1.730170e-01, 2.897276e-01, -5.804821e-02, 3.127468e-02, -8.206328e-02, 1.367536e-01, 9.111842e-03, 3.254448e-01, 6.179603e-02, -2.172228e-01, 6.440079e-02, 2.521933e-01, 1.311167e-01, 3.416006e-02, -1.639557e-01, 1.828443e-01, 3.718416e-03, 2.550828e-01, -1.520945e-01, -6.546378e-03, -3.168853e-01, 1.118225e-01, -7.605200e-02, 1.901283e-01, -1.593709e-02, 2.907586e-02, -2.391434e-01, 3.183103e-02, 2.454578e-01, 3.210540e-01, -2.815106e-01, 1.050017e-01, 2.525447e-01, 2.376143e-01, -1.084135e-01, 3.067156e-01, 3.306109e-01, -1.672828e-01, -1.854879e-01, -1.088943e-01, 2.538097e-01, 6.095584e-02, -9.870847e-02, -1.969574e-01, -2.833847e-01, -7.807517e-02, -2.605601e-01, 2.054415e-01, 2.815760e-01, 5.028033e-02, -6.896254e-02, 9.431513e-02, 3.129761e-01, -1.239603e-01, -3.056056e-01, -3.285869e-01, -2.023224e-01, 9.925818e-02, -9.563828e-02, -1.742150e-01, 3.270308e-02, 1.793048e-01, 1.206216e-01, 1.301152e-01, 2.281603e-01, 2.363580e-01, 2.450252e-01, 2.507547e-01, -2.126547e-02, 2.979289e-01, -5.656346e-02, -3.198351e-01, 3.293918e-01, 1.280518e-01, -3.868453e-02, 6.416659e-02, 1.295788e-01, -6.349143e-02, 1.038684e-01, 3.182249e-01, -2.910051e-01, -2.043663e-01, 2.992347e-01, -2.332023e-01, -4.454760e-02, -2.176659e-01, -2.228189e-02, -1.224951e-02, -2.774401e-01, -2.164693e-01, 5.354488e-02, 9.012794e-02, 4.582528e-02, 5.433571e-02, -2.992823e-01, -8.947091e-02, -1.931929e-01, 2.595538e-01, -1.800648e-01, 9.309387e-02, -2.366805e-01, -6.967553e-02, -5.091516e-02, 2.665895e-01, -1.519387e-01, 1.539358e-02, 1.416676e-01, -3.328951e-01, 2.330831e-01, 1.784548e-01, 5.556901e-03, 2.404396e-01, 3.001275e-01, 2.498440e-01, 2.104385e-01, 1.179569e-01, 2.003660e-01, -8.249025e-02, -7.370134e-02, ]).reshape(16, 1, 3, 3, )
__constant_16xf32 = np.array([2.271649e-01, 7.346737e-02, -3.929543e-02, -1.784222e-01, -2.877739e-01, -1.696576e-01, -2.319605e-01, 9.129874e-02, 2.890620e-01, -3.352678e-02, 9.146659e-02, -1.890008e-02, 2.862074e-01, 5.874125e-02, 2.544049e-01, 8.584078e-02, ]).reshape(16, )
# fmt: on


def forward(
    _arg0=ArrayDecl("_arg0", 1, 1, 11, 11, input=True),
    _arg1=ArrayDecl("_arg1", 1, 16, 9, 9, output=True),
):
    # %c16 = arith.constant 16 : index
    # %c1 = arith.constant 1 : index
    # %c9 = arith.constant 9 : index
    # %c0 = arith.constant 0 : index
    # %c3 = arith.constant 3 : index
    # %0 = memref.get_global @__constant_16x1x3x3xf32 : memref<16x1x3x3xf32>
    _0 = Global("_0", "__constant_16x1x3x3xf32", __constant_16x1x3x3xf32)
    # %1 = memref.get_global @__constant_16xf32 : memref<16xf32>
    _1 = Global("_1", "__constant_16xf32", __constant_16xf32)
    # scf.for %arg2 = %c0 to %c1 step %c1 {
    for _arg2 in range(0, 1, 1):
        # scf.for %arg3 = %c0 to %c16 step %c1 {
        for _arg3 in range(0, 16, 1):
            # scf.for %arg4 = %c0 to %c9 step %c1 {
            for _arg4 in range(0, 9, 1):
                # scf.for %arg5 = %c0 to %c9 step %c1 {
                for _arg5 in range(0, 9, 1):
                    # %2 = memref.load %1[%arg3] : memref<16xf32>
                    _2 = _1[_arg3,]
                    # memref.store %2, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
                    _arg1[_arg2, _arg3, _arg4, _arg5] = _2

    # scf.for %arg2 = %c0 to %c1 step %c1 {
    for _arg2 in range(0, 1, 1):
        # scf.for %arg3 = %c0 to %c16 step %c1 {
        for _arg3 in range(0, 16, 1):
            # scf.for %arg4 = %c0 to %c9 step %c1 {
            for _arg4 in range(0, 9, 1):
                # scf.for %arg5 = %c0 to %c9 step %c1 {
                for _arg5 in range(0, 9, 1):
                    # scf.for %arg6 = %c0 to %c1 step %c1 {
                    for _arg6 in range(0, 1, 1):
                        # scf.for %arg7 = %c0 to %c3 step %c1 {
                        for _arg7 in range(0, 3, 1):
                            # scf.for %arg8 = %c0 to %c3 step %c1 {
                            for _arg8 in range(0, 3, 1):
                                # %2 = arith.addi %arg4, %arg7 : index
                                _2 = _arg4 + _arg7
                                # %3 = arith.addi %arg5, %arg8 : index
                                _3 = _arg5 + _arg8
                                # %4 = memref.load %arg0[%arg2, %arg6, %2, %3] : memref<1x1x11x11xf32>
                                _4 = _arg0[_arg2, _arg6, ]
                                # %5 = memref.load %0[%arg3, %arg6, %arg7, %arg8] : memref<16x1x3x3xf32>
                                _5 = _0[_arg3, _arg6, _arg7, _arg8]
                                # %6 = memref.load %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
                                _6 = _arg1[_arg2, _arg3, _arg4, _arg5]
                                # %7 = arith.mulf %4, %5 : f32
                                _7 = _4 * _5
                                # %8 = arith.addf %6, %7 : f32
                                _8 = _6 + _7
                                # memref.store %8, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
                                _arg1[_arg2, _arg3, _arg4, _arg5] = _8

    # return


Forward(forward)
