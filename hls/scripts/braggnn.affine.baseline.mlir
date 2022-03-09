#map = affine_map<(d0, d1) -> (d0 + d1)>
module attributes {torch.debug_module_name = "BraggNN"} {
  memref.global "private" constant @__constant_64x1x3x3xf32 : memref<64x1x3x3xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_32x64x1x1xf32 : memref<32x64x1x1xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_64x32x1x1xf32 : memref<64x32x1x1xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_32x64x3x3xf32 : memref<32x64x3x3xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_8x32x3x3xf32 : memref<8x32x3x3xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_64x200xf32 : memref<64x200xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_32x64xf32 : memref<32x64xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_32xf32 : memref<32xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_16x32xf32 : memref<16x32xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_16xf32 : memref<16xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_2x8xf32 : memref<2x8xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_2xf32 : memref<2xf32> = dense<1.000000e+00>
  func @forward(%arg0: memref<1x1x11x11xf32>, %arg1: memref<1x2xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-02 : f64
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_64x1x3x3xf32 : memref<64x1x3x3xf32>
    %1 = memref.get_global @__constant_32x64x1x1xf32 : memref<32x64x1x1xf32>
    %2 = memref.get_global @__constant_64x32x1x1xf32 : memref<64x32x1x1xf32>
    %3 = memref.get_global @__constant_32x64x3x3xf32 : memref<32x64x3x3xf32>
    %4 = memref.get_global @__constant_8x32x3x3xf32 : memref<8x32x3x3xf32>
    %5 = memref.get_global @__constant_64x200xf32 : memref<64x200xf32>
    %6 = memref.get_global @__constant_64xf32 : memref<64xf32>
    %7 = memref.get_global @__constant_32x64xf32 : memref<32x64xf32>
    %8 = memref.get_global @__constant_32xf32 : memref<32xf32>
    %9 = memref.get_global @__constant_16x32xf32 : memref<16x32xf32>
    %10 = memref.get_global @__constant_16xf32 : memref<16xf32>
    %11 = memref.get_global @__constant_8x16xf32 : memref<8x16xf32>
    %12 = memref.get_global @__constant_8xf32 : memref<8xf32>
    %13 = memref.get_global @__constant_2x8xf32 : memref<2x8xf32>
    %14 = memref.get_global @__constant_2xf32 : memref<2xf32>
    %15 = memref.alloca() : memref<1x64x9x9xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            %42 = affine.load %6[%arg3] : memref<64xf32>
            affine.store %42, %15[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            affine.for %arg6 = 0 to 1 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %42 = affine.apply #map(%arg4, %arg7)
                  %43 = affine.apply #map(%arg5, %arg8)
                  %44 = affine.load %arg0[%arg2, %arg6, %42, %43] : memref<1x1x11x11xf32>
                  %45 = affine.load %0[%arg3, %arg6, %arg7, %arg8] : memref<64x1x3x3xf32>
                  %46 = affine.load %15[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  affine.store %48, %15[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32>
                }
              }
            }
          }
        }
      }
    }
    %16 = memref.alloca() : memref<1x32x9x9xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            %42 = affine.load %8[%arg3] : memref<32xf32>
            affine.store %42, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            affine.for %arg6 = 0 to 64 {
              affine.for %arg7 = 0 to 1 {
                affine.for %arg8 = 0 to 1 {
                  %42 = affine.apply #map(%arg4, %arg7)
                  %43 = affine.apply #map(%arg5, %arg8)
                  %44 = affine.load %15[%arg2, %arg6, %42, %43] : memref<1x64x9x9xf32>
                  %45 = affine.load %1[%arg3, %arg6, %arg7, %arg8] : memref<32x64x1x1xf32>
                  %46 = affine.load %16[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  affine.store %48, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
                }
              }
            }
          }
        }
      }
    }
    %17 = memref.alloca() : memref<1x32x9x9xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            %42 = affine.load %8[%arg3] : memref<32xf32>
            affine.store %42, %17[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            affine.for %arg6 = 0 to 64 {
              affine.for %arg7 = 0 to 1 {
                affine.for %arg8 = 0 to 1 {
                  %42 = affine.apply #map(%arg4, %arg7)
                  %43 = affine.apply #map(%arg5, %arg8)
                  %44 = affine.load %15[%arg2, %arg6, %42, %43] : memref<1x64x9x9xf32>
                  %45 = affine.load %1[%arg3, %arg6, %arg7, %arg8] : memref<32x64x1x1xf32>
                  %46 = affine.load %17[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  affine.store %48, %17[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
                }
              }
            }
          }
        }
      }
    }
    %18 = memref.alloca() : memref<1x32x9x9xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            %42 = affine.load %8[%arg3] : memref<32xf32>
            affine.store %42, %18[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            affine.for %arg6 = 0 to 64 {
              affine.for %arg7 = 0 to 1 {
                affine.for %arg8 = 0 to 1 {
                  %42 = affine.apply #map(%arg4, %arg7)
                  %43 = affine.apply #map(%arg5, %arg8)
                  %44 = affine.load %15[%arg2, %arg6, %42, %43] : memref<1x64x9x9xf32>
                  %45 = affine.load %1[%arg3, %arg6, %arg7, %arg8] : memref<32x64x1x1xf32>
                  %46 = affine.load %18[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  affine.store %48, %18[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
                }
              }
            }
          }
        }
      }
    }
    %19 = memref.alloca() : memref<1x32x9x9xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            %42 = affine.load %16[%c0, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
            %43 = affine.load %17[%c0, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
            %44 = arith.mulf %42, %43 : f32
            affine.store %44, %19[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
          }
        }
      }
    }
    %20 = memref.alloca() : memref<1x32x9x9xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            %42 = affine.load %19[%c0, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
            %43 = affine.load %18[%c0, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
            %44 = arith.mulf %42, %43 : f32
            affine.store %44, %20[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32>
          }
        }
      }
    }
    %21 = memref.alloca() : memref<1x64x9x9xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            %42 = affine.load %6[%arg3] : memref<64xf32>
            affine.store %42, %21[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            affine.for %arg6 = 0 to 32 {
              affine.for %arg7 = 0 to 1 {
                affine.for %arg8 = 0 to 1 {
                  %42 = affine.apply #map(%arg4, %arg7)
                  %43 = affine.apply #map(%arg5, %arg8)
                  %44 = affine.load %20[%arg2, %arg6, %42, %43] : memref<1x32x9x9xf32>
                  %45 = affine.load %2[%arg3, %arg6, %arg7, %arg8] : memref<64x32x1x1xf32>
                  %46 = affine.load %21[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  affine.store %48, %21[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32>
                }
              }
            }
          }
        }
      }
    }
    %22 = memref.alloca() : memref<1x64x9x9xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            %42 = affine.load %21[%c0, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32>
            %43 = affine.load %15[%c0, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32>
            %44 = arith.addf %42, %43 : f32
            affine.store %44, %22[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32>
          }
        }
      }
    }
    %23 = memref.alloca() : memref<1x64x9x9xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 9 {
          affine.for %arg5 = 0 to 9 {
            %42 = affine.load %22[%c0, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32>
            %43 = arith.cmpf ugt, %42, %cst : f32
            %44 = arith.select %43, %42, %cst : f32
            %45 = arith.select %43, %cst, %42 : f32
            %46 = arith.truncf %cst_0 : f64 to f32
            %47 = arith.mulf %45, %46 : f32
            %48 = arith.addf %44, %47 : f32
            affine.store %48, %23[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32>
          }
        }
      }
    }
    %24 = memref.alloca() : memref<1x32x7x7xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 7 {
          affine.for %arg5 = 0 to 7 {
            %42 = affine.load %8[%arg3] : memref<32xf32>
            affine.store %42, %24[%arg2, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 7 {
          affine.for %arg5 = 0 to 7 {
            affine.for %arg6 = 0 to 64 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %42 = affine.apply #map(%arg4, %arg7)
                  %43 = affine.apply #map(%arg5, %arg8)
                  %44 = affine.load %23[%arg2, %arg6, %42, %43] : memref<1x64x9x9xf32>
                  %45 = affine.load %3[%arg3, %arg6, %arg7, %arg8] : memref<32x64x3x3xf32>
                  %46 = affine.load %24[%arg2, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  affine.store %48, %24[%arg2, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32>
                }
              }
            }
          }
        }
      }
    }
    %25 = memref.alloca() : memref<1x32x7x7xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 7 {
          affine.for %arg5 = 0 to 7 {
            %42 = affine.load %24[%c0, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32>
            %43 = arith.cmpf ugt, %42, %cst : f32
            %44 = arith.select %43, %42, %cst : f32
            %45 = arith.select %43, %cst, %42 : f32
            %46 = arith.truncf %cst_0 : f64 to f32
            %47 = arith.mulf %45, %46 : f32
            %48 = arith.addf %44, %47 : f32
            affine.store %48, %25[%arg2, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32>
          }
        }
      }
    }
    %26 = memref.alloca() : memref<1x8x5x5xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 8 {
        affine.for %arg4 = 0 to 5 {
          affine.for %arg5 = 0 to 5 {
            %42 = affine.load %12[%arg3] : memref<8xf32>
            affine.store %42, %26[%arg2, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 8 {
        affine.for %arg4 = 0 to 5 {
          affine.for %arg5 = 0 to 5 {
            affine.for %arg6 = 0 to 32 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %42 = affine.apply #map(%arg4, %arg7)
                  %43 = affine.apply #map(%arg5, %arg8)
                  %44 = affine.load %25[%arg2, %arg6, %42, %43] : memref<1x32x7x7xf32>
                  %45 = affine.load %4[%arg3, %arg6, %arg7, %arg8] : memref<8x32x3x3xf32>
                  %46 = affine.load %26[%arg2, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  affine.store %48, %26[%arg2, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32>
                }
              }
            }
          }
        }
      }
    }
    %27 = memref.alloca() : memref<1x8x5x5xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 8 {
        affine.for %arg4 = 0 to 5 {
          affine.for %arg5 = 0 to 5 {
            %42 = affine.load %26[%c0, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32>
            %43 = arith.cmpf ugt, %42, %cst : f32
            %44 = arith.select %43, %42, %cst : f32
            %45 = arith.select %43, %cst, %42 : f32
            %46 = arith.truncf %cst_0 : f64 to f32
            %47 = arith.mulf %45, %46 : f32
            %48 = arith.addf %44, %47 : f32
            affine.store %48, %27[%arg2, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32>
          }
        }
      }
    }
    %28 = memref.collapse_shape %27 [[0], [1, 2, 3]] : memref<1x8x5x5xf32> into memref<1x200xf32>
    %29 = memref.alloca() : memref<1x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        %42 = affine.load %6[%arg3] : memref<64xf32>
        affine.store %42, %29[%arg2, %arg3] : memref<1x64xf32>
      }
    }
    %30 = memref.alloca() : memref<200x64xf32>
    affine.for %arg2 = 0 to 200 {
      affine.for %arg3 = 0 to 64 {
        %42 = affine.load %5[%arg3, %arg2] : memref<64x200xf32>
        affine.store %42, %30[%arg2, %arg3] : memref<200x64xf32>
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 200 {
          %42 = affine.load %28[%arg2, %arg4] : memref<1x200xf32>
          %43 = affine.load %30[%arg4, %arg3] : memref<200x64xf32>
          %44 = affine.load %29[%arg2, %arg3] : memref<1x64xf32>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          affine.store %46, %29[%arg2, %arg3] : memref<1x64xf32>
        }
      }
    }
    %31 = memref.alloca() : memref<1x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        %42 = affine.load %29[%c0, %arg3] : memref<1x64xf32>
        %43 = arith.cmpf ugt, %42, %cst : f32
        %44 = arith.select %43, %42, %cst : f32
        %45 = arith.select %43, %cst, %42 : f32
        %46 = arith.truncf %cst_0 : f64 to f32
        %47 = arith.mulf %45, %46 : f32
        %48 = arith.addf %44, %47 : f32
        affine.store %48, %31[%arg2, %arg3] : memref<1x64xf32>
      }
    }
    %32 = memref.alloca() : memref<1x32xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        %42 = affine.load %8[%arg3] : memref<32xf32>
        affine.store %42, %32[%arg2, %arg3] : memref<1x32xf32>
      }
    }
    %33 = memref.alloca() : memref<64x32xf32>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 32 {
        %42 = affine.load %7[%arg3, %arg2] : memref<32x64xf32>
        affine.store %42, %33[%arg2, %arg3] : memref<64x32xf32>
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        affine.for %arg4 = 0 to 64 {
          %42 = affine.load %31[%arg2, %arg4] : memref<1x64xf32>
          %43 = affine.load %33[%arg4, %arg3] : memref<64x32xf32>
          %44 = affine.load %32[%arg2, %arg3] : memref<1x32xf32>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          affine.store %46, %32[%arg2, %arg3] : memref<1x32xf32>
        }
      }
    }
    %34 = memref.alloca() : memref<1x32xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 32 {
        %42 = affine.load %32[%c0, %arg3] : memref<1x32xf32>
        %43 = arith.cmpf ugt, %42, %cst : f32
        %44 = arith.select %43, %42, %cst : f32
        %45 = arith.select %43, %cst, %42 : f32
        %46 = arith.truncf %cst_0 : f64 to f32
        %47 = arith.mulf %45, %46 : f32
        %48 = arith.addf %44, %47 : f32
        affine.store %48, %34[%arg2, %arg3] : memref<1x32xf32>
      }
    }
    %35 = memref.alloca() : memref<1x16xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 16 {
        %42 = affine.load %10[%arg3] : memref<16xf32>
        affine.store %42, %35[%arg2, %arg3] : memref<1x16xf32>
      }
    }
    %36 = memref.alloca() : memref<32x16xf32>
    affine.for %arg2 = 0 to 32 {
      affine.for %arg3 = 0 to 16 {
        %42 = affine.load %9[%arg3, %arg2] : memref<16x32xf32>
        affine.store %42, %36[%arg2, %arg3] : memref<32x16xf32>
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 32 {
          %42 = affine.load %34[%arg2, %arg4] : memref<1x32xf32>
          %43 = affine.load %36[%arg4, %arg3] : memref<32x16xf32>
          %44 = affine.load %35[%arg2, %arg3] : memref<1x16xf32>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          affine.store %46, %35[%arg2, %arg3] : memref<1x16xf32>
        }
      }
    }
    %37 = memref.alloca() : memref<1x16xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 16 {
        %42 = affine.load %35[%c0, %arg3] : memref<1x16xf32>
        %43 = arith.cmpf ugt, %42, %cst : f32
        %44 = arith.select %43, %42, %cst : f32
        %45 = arith.select %43, %cst, %42 : f32
        %46 = arith.truncf %cst_0 : f64 to f32
        %47 = arith.mulf %45, %46 : f32
        %48 = arith.addf %44, %47 : f32
        affine.store %48, %37[%arg2, %arg3] : memref<1x16xf32>
      }
    }
    %38 = memref.alloca() : memref<1x8xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 8 {
        %42 = affine.load %12[%arg3] : memref<8xf32>
        affine.store %42, %38[%arg2, %arg3] : memref<1x8xf32>
      }
    }
    %39 = memref.alloca() : memref<16x8xf32>
    affine.for %arg2 = 0 to 16 {
      affine.for %arg3 = 0 to 8 {
        %42 = affine.load %11[%arg3, %arg2] : memref<8x16xf32>
        affine.store %42, %39[%arg2, %arg3] : memref<16x8xf32>
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 8 {
        affine.for %arg4 = 0 to 16 {
          %42 = affine.load %37[%arg2, %arg4] : memref<1x16xf32>
          %43 = affine.load %39[%arg4, %arg3] : memref<16x8xf32>
          %44 = affine.load %38[%arg2, %arg3] : memref<1x8xf32>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          affine.store %46, %38[%arg2, %arg3] : memref<1x8xf32>
        }
      }
    }
    %40 = memref.alloca() : memref<1x8xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 8 {
        %42 = affine.load %38[%c0, %arg3] : memref<1x8xf32>
        %43 = arith.cmpf ugt, %42, %cst : f32
        %44 = arith.select %43, %42, %cst : f32
        %45 = arith.select %43, %cst, %42 : f32
        %46 = arith.truncf %cst_0 : f64 to f32
        %47 = arith.mulf %45, %46 : f32
        %48 = arith.addf %44, %47 : f32
        affine.store %48, %40[%arg2, %arg3] : memref<1x8xf32>
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        %42 = affine.load %14[%arg3] : memref<2xf32>
        affine.store %42, %arg1[%arg2, %arg3] : memref<1x2xf32>
      }
    }
    %41 = memref.alloca() : memref<8x2xf32>
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 2 {
        %42 = affine.load %13[%arg3, %arg2] : memref<2x8xf32>
        affine.store %42, %41[%arg2, %arg3] : memref<8x2xf32>
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 2 {
        affine.for %arg4 = 0 to 8 {
          %42 = affine.load %40[%arg2, %arg4] : memref<1x8xf32>
          %43 = affine.load %41[%arg4, %arg3] : memref<8x2xf32>
          %44 = affine.load %arg1[%arg2, %arg3] : memref<1x2xf32>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          affine.store %46, %arg1[%arg2, %arg3] : memref<1x2xf32>
        }
      }
    }
    return
  }
}
