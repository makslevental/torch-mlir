module attributes {torch.debug_module_name = "BraggNN"} {
  memref.global "private" constant @__constant_16x1x3x3xf32 : memref<16x1x3x3xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_8x16x1x1xf32 : memref<8x16x1x1xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_16x8x1x1xf32 : memref<16x8x1x1xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_8x16x3x3xf32 : memref<8x16x3x3xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_2x8x3x3xf32 : memref<2x8x3x3xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_16x50xf32 : memref<16x50xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_16xf32 : memref<16xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_4x8xf32 : memref<4x8xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_4xf32 : memref<4xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_2x4xf32 : memref<2x4xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_2x2xf32 : memref<2x2xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_2xf32 : memref<2xf32> = dense<1.000000e+00>
  func @forward(%arg0: memref<1x1x11x11xf32>, %arg1: memref<1x2xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-02 : f64
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c50 = arith.constant 50 : index
    %c4 = arith.constant 4 : index
    %0 = memref.get_global @__constant_16x1x3x3xf32 : memref<16x1x3x3xf32>
    %1 = memref.get_global @__constant_8x16x1x1xf32 : memref<8x16x1x1xf32>
    %2 = memref.get_global @__constant_16x8x1x1xf32 : memref<16x8x1x1xf32>
    %3 = memref.get_global @__constant_8x16x3x3xf32 : memref<8x16x3x3xf32>
    %4 = memref.get_global @__constant_2x8x3x3xf32 : memref<2x8x3x3xf32>
    %5 = memref.get_global @__constant_16x50xf32 : memref<16x50xf32>
    %6 = memref.get_global @__constant_16xf32 : memref<16xf32>
    %7 = memref.get_global @__constant_8x16xf32 : memref<8x16xf32>
    %8 = memref.get_global @__constant_8xf32 : memref<8xf32>
    %9 = memref.get_global @__constant_4x8xf32 : memref<4x8xf32>
    %10 = memref.get_global @__constant_4xf32 : memref<4xf32>
    %11 = memref.get_global @__constant_2x4xf32 : memref<2x4xf32>
    %12 = memref.get_global @__constant_2x2xf32 : memref<2x2xf32>
    %13 = memref.get_global @__constant_2xf32 : memref<2xf32>
    %14 = memref.alloc() : memref<1x16x9x9xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %42 = memref.load %6[%arg3] : memref<16xf32>
            memref.store %42, %14[%arg2, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
          }
        }
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            scf.for %arg6 = %c0 to %c1 step %c1 {
              scf.for %arg7 = %c0 to %c3 step %c1 {
                scf.for %arg8 = %c0 to %c3 step %c1 {
                  %42 = arith.addi %arg4, %arg7 : index
                  %43 = arith.addi %arg5, %arg8 : index
                  %44 = memref.load %arg0[%arg2, %arg6, %42, %43] : memref<1x1x11x11xf32>
                  %45 = memref.load %0[%arg3, %arg6, %arg7, %arg8] : memref<16x1x3x3xf32>
                  %46 = memref.load %14[%arg2, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  memref.store %48, %14[%arg2, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
                }
              }
            }
          }
        }
      }
    }
    %15 = memref.alloc() : memref<1x8x9x9xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %42 = memref.load %8[%arg3] : memref<8xf32>
            memref.store %42, %15[%arg2, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
          }
        }
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            scf.for %arg6 = %c0 to %c16 step %c1 {
              scf.for %arg7 = %c0 to %c1 step %c1 {
                scf.for %arg8 = %c0 to %c1 step %c1 {
                  %42 = arith.addi %arg4, %arg7 : index
                  %43 = arith.addi %arg5, %arg8 : index
                  %44 = memref.load %14[%arg2, %arg6, %42, %43] : memref<1x16x9x9xf32>
                  %45 = memref.load %1[%arg3, %arg6, %arg7, %arg8] : memref<8x16x1x1xf32>
                  %46 = memref.load %15[%arg2, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  memref.store %48, %15[%arg2, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
                }
              }
            }
          }
        }
      }
    }
    %16 = memref.alloc() : memref<1x8x9x9xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %42 = memref.load %8[%arg3] : memref<8xf32>
            memref.store %42, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
          }
        }
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            scf.for %arg6 = %c0 to %c16 step %c1 {
              scf.for %arg7 = %c0 to %c1 step %c1 {
                scf.for %arg8 = %c0 to %c1 step %c1 {
                  %42 = arith.addi %arg4, %arg7 : index
                  %43 = arith.addi %arg5, %arg8 : index
                  %44 = memref.load %14[%arg2, %arg6, %42, %43] : memref<1x16x9x9xf32>
                  %45 = memref.load %1[%arg3, %arg6, %arg7, %arg8] : memref<8x16x1x1xf32>
                  %46 = memref.load %16[%arg2, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  memref.store %48, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
                }
              }
            }
          }
        }
      }
    }
    %17 = memref.alloc() : memref<1x8x9x9xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %42 = memref.load %8[%arg3] : memref<8xf32>
            memref.store %42, %17[%arg2, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
          }
        }
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            scf.for %arg6 = %c0 to %c16 step %c1 {
              scf.for %arg7 = %c0 to %c1 step %c1 {
                scf.for %arg8 = %c0 to %c1 step %c1 {
                  %42 = arith.addi %arg4, %arg7 : index
                  %43 = arith.addi %arg5, %arg8 : index
                  %44 = memref.load %14[%arg2, %arg6, %42, %43] : memref<1x16x9x9xf32>
                  %45 = memref.load %1[%arg3, %arg6, %arg7, %arg8] : memref<8x16x1x1xf32>
                  %46 = memref.load %17[%arg2, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  memref.store %48, %17[%arg2, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
                }
              }
            }
          }
        }
      }
    }
    %18 = memref.alloc() : memref<1x8x9x9xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %42 = memref.load %15[%c0, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
            %43 = memref.load %16[%c0, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
            %44 = arith.mulf %42, %43 : f32
            memref.store %44, %18[%arg2, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
          }
        }
      }
    }
    %19 = memref.alloc() : memref<1x8x9x9xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %42 = memref.load %18[%c0, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
            %43 = memref.load %17[%c0, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
            %44 = arith.mulf %42, %43 : f32
            memref.store %44, %19[%arg2, %arg3, %arg4, %arg5] : memref<1x8x9x9xf32>
          }
        }
      }
    }
    %20 = memref.alloc() : memref<1x16x9x9xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %42 = memref.load %6[%arg3] : memref<16xf32>
            memref.store %42, %20[%arg2, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
          }
        }
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            scf.for %arg6 = %c0 to %c8 step %c1 {
              scf.for %arg7 = %c0 to %c1 step %c1 {
                scf.for %arg8 = %c0 to %c1 step %c1 {
                  %42 = arith.addi %arg4, %arg7 : index
                  %43 = arith.addi %arg5, %arg8 : index
                  %44 = memref.load %19[%arg2, %arg6, %42, %43] : memref<1x8x9x9xf32>
                  %45 = memref.load %2[%arg3, %arg6, %arg7, %arg8] : memref<16x8x1x1xf32>
                  %46 = memref.load %20[%arg2, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  memref.store %48, %20[%arg2, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
                }
              }
            }
          }
        }
      }
    }
    %21 = memref.alloc() : memref<1x16x9x9xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %42 = memref.load %20[%c0, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
            %43 = memref.load %14[%c0, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
            %44 = arith.addf %42, %43 : f32
            memref.store %44, %21[%arg2, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
          }
        }
      }
    }
    %22 = memref.alloc() : memref<1x16x9x9xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %42 = memref.load %21[%c0, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
            %43 = arith.cmpf ugt, %42, %cst : f32
            %44 = arith.select %43, %42, %cst : f32
            %45 = arith.select %43, %cst, %42 : f32
            %46 = arith.truncf %cst_0 : f64 to f32
            %47 = arith.mulf %45, %46 : f32
            %48 = arith.addf %44, %47 : f32
            memref.store %48, %22[%arg2, %arg3, %arg4, %arg5] : memref<1x16x9x9xf32>
          }
        }
      }
    }
    %23 = memref.alloc() : memref<1x8x7x7xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c7 step %c1 {
          scf.for %arg5 = %c0 to %c7 step %c1 {
            %42 = memref.load %8[%arg3] : memref<8xf32>
            memref.store %42, %23[%arg2, %arg3, %arg4, %arg5] : memref<1x8x7x7xf32>
          }
        }
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c7 step %c1 {
          scf.for %arg5 = %c0 to %c7 step %c1 {
            scf.for %arg6 = %c0 to %c16 step %c1 {
              scf.for %arg7 = %c0 to %c3 step %c1 {
                scf.for %arg8 = %c0 to %c3 step %c1 {
                  %42 = arith.addi %arg4, %arg7 : index
                  %43 = arith.addi %arg5, %arg8 : index
                  %44 = memref.load %22[%arg2, %arg6, %42, %43] : memref<1x16x9x9xf32>
                  %45 = memref.load %3[%arg3, %arg6, %arg7, %arg8] : memref<8x16x3x3xf32>
                  %46 = memref.load %23[%arg2, %arg3, %arg4, %arg5] : memref<1x8x7x7xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  memref.store %48, %23[%arg2, %arg3, %arg4, %arg5] : memref<1x8x7x7xf32>
                }
              }
            }
          }
        }
      }
    }
    %24 = memref.alloc() : memref<1x8x7x7xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c7 step %c1 {
          scf.for %arg5 = %c0 to %c7 step %c1 {
            %42 = memref.load %23[%c0, %arg3, %arg4, %arg5] : memref<1x8x7x7xf32>
            %43 = arith.cmpf ugt, %42, %cst : f32
            %44 = arith.select %43, %42, %cst : f32
            %45 = arith.select %43, %cst, %42 : f32
            %46 = arith.truncf %cst_0 : f64 to f32
            %47 = arith.mulf %45, %46 : f32
            %48 = arith.addf %44, %47 : f32
            memref.store %48, %24[%arg2, %arg3, %arg4, %arg5] : memref<1x8x7x7xf32>
          }
        }
      }
    }
    %25 = memref.alloc() : memref<1x2x5x5xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c5 step %c1 {
          scf.for %arg5 = %c0 to %c5 step %c1 {
            %42 = memref.load %13[%arg3] : memref<2xf32>
            memref.store %42, %25[%arg2, %arg3, %arg4, %arg5] : memref<1x2x5x5xf32>
          }
        }
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c5 step %c1 {
          scf.for %arg5 = %c0 to %c5 step %c1 {
            scf.for %arg6 = %c0 to %c8 step %c1 {
              scf.for %arg7 = %c0 to %c3 step %c1 {
                scf.for %arg8 = %c0 to %c3 step %c1 {
                  %42 = arith.addi %arg4, %arg7 : index
                  %43 = arith.addi %arg5, %arg8 : index
                  %44 = memref.load %24[%arg2, %arg6, %42, %43] : memref<1x8x7x7xf32>
                  %45 = memref.load %4[%arg3, %arg6, %arg7, %arg8] : memref<2x8x3x3xf32>
                  %46 = memref.load %25[%arg2, %arg3, %arg4, %arg5] : memref<1x2x5x5xf32>
                  %47 = arith.mulf %44, %45 : f32
                  %48 = arith.addf %46, %47 : f32
                  memref.store %48, %25[%arg2, %arg3, %arg4, %arg5] : memref<1x2x5x5xf32>
                }
              }
            }
          }
        }
      }
    }
    %26 = memref.alloc() : memref<1x2x5x5xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c5 step %c1 {
          scf.for %arg5 = %c0 to %c5 step %c1 {
            %42 = memref.load %25[%c0, %arg3, %arg4, %arg5] : memref<1x2x5x5xf32>
            %43 = arith.cmpf ugt, %42, %cst : f32
            %44 = arith.select %43, %42, %cst : f32
            %45 = arith.select %43, %cst, %42 : f32
            %46 = arith.truncf %cst_0 : f64 to f32
            %47 = arith.mulf %45, %46 : f32
            %48 = arith.addf %44, %47 : f32
            memref.store %48, %26[%arg2, %arg3, %arg4, %arg5] : memref<1x2x5x5xf32>
          }
        }
      }
    }
    %27 = memref.collapse_shape %26 [[0], [1, 2, 3]] : memref<1x2x5x5xf32> into memref<1x50xf32>
    %28 = memref.alloc() : memref<1x16xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %42 = memref.load %6[%arg3] : memref<16xf32>
        memref.store %42, %28[%arg2, %arg3] : memref<1x16xf32>
      }
    }
    %29 = memref.alloc() : memref<50x16xf32>
    scf.for %arg2 = %c0 to %c50 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %42 = memref.load %5[%arg3, %arg2] : memref<16x50xf32>
        memref.store %42, %29[%arg2, %arg3] : memref<50x16xf32>
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        scf.for %arg4 = %c0 to %c50 step %c1 {
          %42 = memref.load %27[%arg2, %arg4] : memref<1x50xf32>
          %43 = memref.load %29[%arg4, %arg3] : memref<50x16xf32>
          %44 = memref.load %28[%arg2, %arg3] : memref<1x16xf32>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          memref.store %46, %28[%arg2, %arg3] : memref<1x16xf32>
        }
      }
    }
    %30 = memref.alloc() : memref<1x16xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %42 = memref.load %28[%c0, %arg3] : memref<1x16xf32>
        %43 = arith.cmpf ugt, %42, %cst : f32
        %44 = arith.select %43, %42, %cst : f32
        %45 = arith.select %43, %cst, %42 : f32
        %46 = arith.truncf %cst_0 : f64 to f32
        %47 = arith.mulf %45, %46 : f32
        %48 = arith.addf %44, %47 : f32
        memref.store %48, %30[%arg2, %arg3] : memref<1x16xf32>
      }
    }
    %31 = memref.alloc() : memref<1x8xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %42 = memref.load %8[%arg3] : memref<8xf32>
        memref.store %42, %31[%arg2, %arg3] : memref<1x8xf32>
      }
    }
    %32 = memref.alloc() : memref<16x8xf32>
    scf.for %arg2 = %c0 to %c16 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %42 = memref.load %7[%arg3, %arg2] : memref<8x16xf32>
        memref.store %42, %32[%arg2, %arg3] : memref<16x8xf32>
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c16 step %c1 {
          %42 = memref.load %30[%arg2, %arg4] : memref<1x16xf32>
          %43 = memref.load %32[%arg4, %arg3] : memref<16x8xf32>
          %44 = memref.load %31[%arg2, %arg3] : memref<1x8xf32>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          memref.store %46, %31[%arg2, %arg3] : memref<1x8xf32>
        }
      }
    }
    %33 = memref.alloc() : memref<1x8xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %42 = memref.load %31[%c0, %arg3] : memref<1x8xf32>
        %43 = arith.cmpf ugt, %42, %cst : f32
        %44 = arith.select %43, %42, %cst : f32
        %45 = arith.select %43, %cst, %42 : f32
        %46 = arith.truncf %cst_0 : f64 to f32
        %47 = arith.mulf %45, %46 : f32
        %48 = arith.addf %44, %47 : f32
        memref.store %48, %33[%arg2, %arg3] : memref<1x8xf32>
      }
    }
    %34 = memref.alloc() : memref<1x4xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %42 = memref.load %10[%arg3] : memref<4xf32>
        memref.store %42, %34[%arg2, %arg3] : memref<1x4xf32>
      }
    }
    %35 = memref.alloc() : memref<8x4xf32>
    scf.for %arg2 = %c0 to %c8 step %c1 {
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %42 = memref.load %9[%arg3, %arg2] : memref<4x8xf32>
        memref.store %42, %35[%arg2, %arg3] : memref<8x4xf32>
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c4 step %c1 {
        scf.for %arg4 = %c0 to %c8 step %c1 {
          %42 = memref.load %33[%arg2, %arg4] : memref<1x8xf32>
          %43 = memref.load %35[%arg4, %arg3] : memref<8x4xf32>
          %44 = memref.load %34[%arg2, %arg3] : memref<1x4xf32>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          memref.store %46, %34[%arg2, %arg3] : memref<1x4xf32>
        }
      }
    }
    %36 = memref.alloc() : memref<1x4xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %42 = memref.load %34[%c0, %arg3] : memref<1x4xf32>
        %43 = arith.cmpf ugt, %42, %cst : f32
        %44 = arith.select %43, %42, %cst : f32
        %45 = arith.select %43, %cst, %42 : f32
        %46 = arith.truncf %cst_0 : f64 to f32
        %47 = arith.mulf %45, %46 : f32
        %48 = arith.addf %44, %47 : f32
        memref.store %48, %36[%arg2, %arg3] : memref<1x4xf32>
      }
    }
    %37 = memref.alloc() : memref<1x2xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %42 = memref.load %13[%arg3] : memref<2xf32>
        memref.store %42, %37[%arg2, %arg3] : memref<1x2xf32>
      }
    }
    %38 = memref.alloc() : memref<4x2xf32>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %42 = memref.load %11[%arg3, %arg2] : memref<2x4xf32>
        memref.store %42, %38[%arg2, %arg3] : memref<4x2xf32>
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c4 step %c1 {
          %42 = memref.load %36[%arg2, %arg4] : memref<1x4xf32>
          %43 = memref.load %38[%arg4, %arg3] : memref<4x2xf32>
          %44 = memref.load %37[%arg2, %arg3] : memref<1x2xf32>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          memref.store %46, %37[%arg2, %arg3] : memref<1x2xf32>
        }
      }
    }
    %39 = memref.alloc() : memref<1x2xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %42 = memref.load %37[%c0, %arg3] : memref<1x2xf32>
        %43 = arith.cmpf ugt, %42, %cst : f32
        %44 = arith.select %43, %42, %cst : f32
        %45 = arith.select %43, %cst, %42 : f32
        %46 = arith.truncf %cst_0 : f64 to f32
        %47 = arith.mulf %45, %46 : f32
        %48 = arith.addf %44, %47 : f32
        memref.store %48, %39[%arg2, %arg3] : memref<1x2xf32>
      }
    }
    %40 = memref.alloc() : memref<1x2xf32>
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %42 = memref.load %13[%arg3] : memref<2xf32>
        memref.store %42, %arg1[%arg2, %arg3] : memref<1x2xf32>
      }
    }
    %41 = memref.alloc() : memref<2x2xf32>
    scf.for %arg2 = %c0 to %c2 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %42 = memref.load %12[%arg3, %arg2] : memref<2x2xf32>
        memref.store %42, %41[%arg2, %arg3] : memref<2x2xf32>
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c2 step %c1 {
          %42 = memref.load %39[%arg2, %arg4] : memref<1x2xf32>
          %43 = memref.load %41[%arg4, %arg3] : memref<2x2xf32>
          %44 = memref.load %arg1[%arg2, %arg3] : memref<1x2xf32>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          memref.store %46, %arg1[%arg2, %arg3] : memref<1x2xf32>
        }
      }
    }
    return
  }
}
