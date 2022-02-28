#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 3468 + d1 * 1156 + d2 * 34 + d3 + 35)>
#map1 = affine_map<(d0, d1) -> (d0 + d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * 73984 + d1 * 1156 + d2 * 34 + d3 + 35)>
#map3 = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0 * 41472 + d1 * 324 + d2 * 18 + d3 + 19)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0 * 25600 + d1 * 100 + d2 * 10 + d3 + 11)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0 * 18432 + d1 * 36 + d2 * 6 + d3 + 7)>
module attributes {torch.debug_module_name = "ResNet"} {
  memref.global "private" constant @__constant_64x3x3x3xi8 : memref<64x3x3x3xi8> = dense<1>
  memref.global "private" constant @__constant_64x64x3x3xi8 : memref<64x64x3x3xi8> = dense<1>
  memref.global "private" constant @__constant_128x64x3x3xi8 : memref<128x64x3x3xi8> = dense<1>
  memref.global "private" constant @__constant_128x64x1x1xi8 : memref<128x64x1x1xi8> = dense<1>
  memref.global "private" constant @__constant_128x128x3x3xi8 : memref<128x128x3x3xi8> = dense<1>
  memref.global "private" constant @__constant_256x128x3x3xi8 : memref<256x128x3x3xi8> = dense<1>
  memref.global "private" constant @__constant_256x128x1x1xi8 : memref<256x128x1x1xi8> = dense<1>
  memref.global "private" constant @__constant_256x256x3x3xi8 : memref<256x256x3x3xi8> = dense<1>
  memref.global "private" constant @__constant_512x256x3x3xi8 : memref<512x256x3x3xi8> = dense<1>
  memref.global "private" constant @__constant_512x256x1x1xi8 : memref<512x256x1x1xi8> = dense<1>
  memref.global "private" constant @__constant_512x512x3x3xi8 : memref<512x512x3x3xi8> = dense<1>
  memref.global "private" constant @__constant_10x512xi8 : memref<10x512xi8> = dense<1>
  memref.global "private" constant @__constant_10xi8 : memref<10xi8> = dense<1>
  func @forward(%arg0: memref<1x3x32x32xi8>, %arg1: memref<1x10xi8>) {
    %c16_i8 = arith.constant 16 : i8
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_64x3x3x3xi8 : memref<64x3x3x3xi8>
    %1 = memref.get_global @__constant_64x64x3x3xi8 : memref<64x64x3x3xi8>
    %2 = memref.get_global @__constant_128x64x3x3xi8 : memref<128x64x3x3xi8>
    %3 = memref.get_global @__constant_128x64x1x1xi8 : memref<128x64x1x1xi8>
    %4 = memref.get_global @__constant_128x128x3x3xi8 : memref<128x128x3x3xi8>
    %5 = memref.get_global @__constant_256x128x3x3xi8 : memref<256x128x3x3xi8>
    %6 = memref.get_global @__constant_256x128x1x1xi8 : memref<256x128x1x1xi8>
    %7 = memref.get_global @__constant_256x256x3x3xi8 : memref<256x256x3x3xi8>
    %8 = memref.get_global @__constant_512x256x3x3xi8 : memref<512x256x3x3xi8>
    %9 = memref.get_global @__constant_512x256x1x1xi8 : memref<512x256x1x1xi8>
    %10 = memref.get_global @__constant_512x512x3x3xi8 : memref<512x512x3x3xi8>
    %11 = memref.get_global @__constant_10x512xi8 : memref<10x512xi8>
    %12 = memref.get_global @__constant_10xi8 : memref<10xi8>
    %13 = memref.alloca() : memref<1x3x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            affine.store %c0_i8, %13[%arg2, %arg3, %arg4, %arg5] : memref<1x3x34x34xi8>
          }
        }
      }
    }
    %14 = memref.alloca() : memref<1x3x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            %97 = affine.load %13[%arg2, %arg3, %arg4, %arg5] : memref<1x3x34x34xi8>
            affine.store %97, %14[%arg2, %arg3, %arg4, %arg5] : memref<1x3x34x34xi8>
          }
        }
      }
    }
    %15 = memref.subview %14[0, 0, 1, 1] [1, 3, 32, 32] [1, 1, 1, 1] : memref<1x3x34x34xi8> to memref<1x3x32x32xi8, #map0>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 3 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<1x3x32x32xi8>
            affine.store %97, %15[%arg2, %arg3, %arg4, %arg5] : memref<1x3x32x32xi8, #map0>
          }
        }
      }
    }
    %16 = memref.alloca() : memref<1x64x32x32xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            affine.store %c0_i8, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            affine.for %arg6 = 0 to 3 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %14[%arg2, %arg6, %97, %98] : memref<1x3x34x34xi8>
                  %100 = affine.load %0[%arg3, %arg6, %arg7, %arg8] : memref<64x3x3x3xi8>
                  %101 = affine.load %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
                }
              }
            }
          }
        }
      }
    }
    %17 = memref.alloca() : memref<1x64x32x32xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %16[%c0, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %17[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    %18 = memref.alloca() : memref<1x64x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            affine.store %c0_i8, %18[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
          }
        }
      }
    }
    %19 = memref.alloca() : memref<1x64x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            %97 = affine.load %18[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
            affine.store %97, %19[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
          }
        }
      }
    }
    %20 = memref.subview %19[0, 0, 1, 1] [1, 64, 32, 32] [1, 1, 1, 1] : memref<1x64x34x34xi8> to memref<1x64x32x32xi8, #map2>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %17[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            affine.store %97, %20[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8, #map2>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            affine.store %c0_i8, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            affine.for %arg6 = 0 to 64 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %19[%arg2, %arg6, %97, %98] : memref<1x64x34x34xi8>
                  %100 = affine.load %1[%arg3, %arg6, %arg7, %arg8] : memref<64x64x3x3xi8>
                  %101 = affine.load %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
                }
              }
            }
          }
        }
      }
    }
    %21 = memref.alloca() : memref<1x64x32x32xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %16[%c0, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %21[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    %22 = memref.alloca() : memref<1x64x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            affine.store %c0_i8, %22[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
          }
        }
      }
    }
    %23 = memref.alloca() : memref<1x64x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            %97 = affine.load %22[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
            affine.store %97, %23[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
          }
        }
      }
    }
    %24 = memref.subview %23[0, 0, 1, 1] [1, 64, 32, 32] [1, 1, 1, 1] : memref<1x64x34x34xi8> to memref<1x64x32x32xi8, #map2>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %21[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            affine.store %97, %24[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8, #map2>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            affine.store %c0_i8, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            affine.for %arg6 = 0 to 64 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %23[%arg2, %arg6, %97, %98] : memref<1x64x34x34xi8>
                  %100 = affine.load %1[%arg3, %arg6, %arg7, %arg8] : memref<64x64x3x3xi8>
                  %101 = affine.load %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
                }
              }
            }
          }
        }
      }
    }
    %25 = memref.alloca() : memref<1x64x32x32xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %16[%c0, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            %98 = affine.load %17[%c0, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            %99 = arith.addi %97, %98 : i8
            affine.store %99, %25[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    %26 = memref.alloca() : memref<1x64x32x32xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %25[%c0, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %26[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    %27 = memref.alloca() : memref<1x64x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            affine.store %c0_i8, %27[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
          }
        }
      }
    }
    %28 = memref.alloca() : memref<1x64x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            %97 = affine.load %27[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
            affine.store %97, %28[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
          }
        }
      }
    }
    %29 = memref.subview %28[0, 0, 1, 1] [1, 64, 32, 32] [1, 1, 1, 1] : memref<1x64x34x34xi8> to memref<1x64x32x32xi8, #map2>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %26[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            affine.store %97, %29[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8, #map2>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            affine.store %c0_i8, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            affine.for %arg6 = 0 to 64 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %28[%arg2, %arg6, %97, %98] : memref<1x64x34x34xi8>
                  %100 = affine.load %1[%arg3, %arg6, %arg7, %arg8] : memref<64x64x3x3xi8>
                  %101 = affine.load %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
                }
              }
            }
          }
        }
      }
    }
    %30 = memref.alloca() : memref<1x64x32x32xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %16[%c0, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %30[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    %31 = memref.alloca() : memref<1x64x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            affine.store %c0_i8, %31[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
          }
        }
      }
    }
    %32 = memref.alloca() : memref<1x64x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            %97 = affine.load %31[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
            affine.store %97, %32[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
          }
        }
      }
    }
    %33 = memref.subview %32[0, 0, 1, 1] [1, 64, 32, 32] [1, 1, 1, 1] : memref<1x64x34x34xi8> to memref<1x64x32x32xi8, #map2>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %30[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            affine.store %97, %33[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8, #map2>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            affine.store %c0_i8, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            affine.for %arg6 = 0 to 64 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %32[%arg2, %arg6, %97, %98] : memref<1x64x34x34xi8>
                  %100 = affine.load %1[%arg3, %arg6, %arg7, %arg8] : memref<64x64x3x3xi8>
                  %101 = affine.load %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
                }
              }
            }
          }
        }
      }
    }
    %34 = memref.alloca() : memref<1x64x32x32xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %16[%c0, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            %98 = affine.load %26[%c0, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            %99 = arith.addi %97, %98 : i8
            affine.store %99, %34[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    %35 = memref.alloca() : memref<1x64x32x32xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %34[%c0, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %35[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
          }
        }
      }
    }
    %36 = memref.alloca() : memref<1x64x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            affine.store %c0_i8, %36[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
          }
        }
      }
    }
    %37 = memref.alloca() : memref<1x64x34x34xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 34 {
          affine.for %arg5 = 0 to 34 {
            %97 = affine.load %36[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
            affine.store %97, %37[%arg2, %arg3, %arg4, %arg5] : memref<1x64x34x34xi8>
          }
        }
      }
    }
    %38 = memref.subview %37[0, 0, 1, 1] [1, 64, 32, 32] [1, 1, 1, 1] : memref<1x64x34x34xi8> to memref<1x64x32x32xi8, #map2>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 32 {
          affine.for %arg5 = 0 to 32 {
            %97 = affine.load %35[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8>
            affine.store %97, %38[%arg2, %arg3, %arg4, %arg5] : memref<1x64x32x32xi8, #map2>
          }
        }
      }
    }
    %39 = memref.alloca() : memref<1x128x16x16xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            affine.store %c0_i8, %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 64 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map3(%arg4, %arg7)
                  %98 = affine.apply #map3(%arg5, %arg8)
                  %99 = affine.load %37[%arg2, %arg6, %97, %98] : memref<1x64x34x34xi8>
                  %100 = affine.load %2[%arg3, %arg6, %arg7, %arg8] : memref<128x64x3x3xi8>
                  %101 = affine.load %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
                }
              }
            }
          }
        }
      }
    }
    %40 = memref.alloca() : memref<1x128x16x16xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %97 = affine.load %39[%c0, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %40[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
          }
        }
      }
    }
    %41 = memref.alloca() : memref<1x128x18x18xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 18 {
          affine.for %arg5 = 0 to 18 {
            affine.store %c0_i8, %41[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
          }
        }
      }
    }
    %42 = memref.alloca() : memref<1x128x18x18xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 18 {
          affine.for %arg5 = 0 to 18 {
            %97 = affine.load %41[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
            affine.store %97, %42[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
          }
        }
      }
    }
    %43 = memref.subview %42[0, 0, 1, 1] [1, 128, 16, 16] [1, 1, 1, 1] : memref<1x128x18x18xi8> to memref<1x128x16x16xi8, #map4>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %97 = affine.load %40[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            affine.store %97, %43[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8, #map4>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            affine.store %c0_i8, %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 128 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %42[%arg2, %arg6, %97, %98] : memref<1x128x18x18xi8>
                  %100 = affine.load %4[%arg3, %arg6, %arg7, %arg8] : memref<128x128x3x3xi8>
                  %101 = affine.load %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
                }
              }
            }
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            affine.store %c0_i8, %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 64 {
              affine.for %arg7 = 0 to 1 {
                affine.for %arg8 = 0 to 1 {
                  %97 = affine.apply #map3(%arg4, %arg7)
                  %98 = affine.apply #map3(%arg5, %arg8)
                  %99 = affine.load %35[%arg2, %arg6, %97, %98] : memref<1x64x32x32xi8>
                  %100 = affine.load %3[%arg3, %arg6, %arg7, %arg8] : memref<128x64x1x1xi8>
                  %101 = affine.load %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
                }
              }
            }
          }
        }
      }
    }
    %44 = memref.alloca() : memref<1x128x16x16xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %97 = affine.load %39[%c0, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            %98 = affine.load %39[%c0, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            %99 = arith.addi %97, %98 : i8
            affine.store %99, %44[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
          }
        }
      }
    }
    %45 = memref.alloca() : memref<1x128x16x16xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %97 = affine.load %44[%c0, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %45[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
          }
        }
      }
    }
    %46 = memref.alloca() : memref<1x128x18x18xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 18 {
          affine.for %arg5 = 0 to 18 {
            affine.store %c0_i8, %46[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
          }
        }
      }
    }
    %47 = memref.alloca() : memref<1x128x18x18xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 18 {
          affine.for %arg5 = 0 to 18 {
            %97 = affine.load %46[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
            affine.store %97, %47[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
          }
        }
      }
    }
    %48 = memref.subview %47[0, 0, 1, 1] [1, 128, 16, 16] [1, 1, 1, 1] : memref<1x128x18x18xi8> to memref<1x128x16x16xi8, #map4>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %97 = affine.load %45[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            affine.store %97, %48[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8, #map4>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            affine.store %c0_i8, %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 128 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %47[%arg2, %arg6, %97, %98] : memref<1x128x18x18xi8>
                  %100 = affine.load %4[%arg3, %arg6, %arg7, %arg8] : memref<128x128x3x3xi8>
                  %101 = affine.load %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
                }
              }
            }
          }
        }
      }
    }
    %49 = memref.alloca() : memref<1x128x16x16xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %97 = affine.load %39[%c0, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %49[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
          }
        }
      }
    }
    %50 = memref.alloca() : memref<1x128x18x18xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 18 {
          affine.for %arg5 = 0 to 18 {
            affine.store %c0_i8, %50[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
          }
        }
      }
    }
    %51 = memref.alloca() : memref<1x128x18x18xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 18 {
          affine.for %arg5 = 0 to 18 {
            %97 = affine.load %50[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
            affine.store %97, %51[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
          }
        }
      }
    }
    %52 = memref.subview %51[0, 0, 1, 1] [1, 128, 16, 16] [1, 1, 1, 1] : memref<1x128x18x18xi8> to memref<1x128x16x16xi8, #map4>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %97 = affine.load %49[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            affine.store %97, %52[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8, #map4>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            affine.store %c0_i8, %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 128 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %51[%arg2, %arg6, %97, %98] : memref<1x128x18x18xi8>
                  %100 = affine.load %4[%arg3, %arg6, %arg7, %arg8] : memref<128x128x3x3xi8>
                  %101 = affine.load %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %39[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
                }
              }
            }
          }
        }
      }
    }
    %53 = memref.alloca() : memref<1x128x16x16xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %97 = affine.load %39[%c0, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            %98 = affine.load %45[%c0, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            %99 = arith.addi %97, %98 : i8
            affine.store %99, %53[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
          }
        }
      }
    }
    %54 = memref.alloca() : memref<1x128x16x16xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %97 = affine.load %53[%c0, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %54[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
          }
        }
      }
    }
    %55 = memref.alloca() : memref<1x128x18x18xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 18 {
          affine.for %arg5 = 0 to 18 {
            affine.store %c0_i8, %55[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
          }
        }
      }
    }
    %56 = memref.alloca() : memref<1x128x18x18xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 18 {
          affine.for %arg5 = 0 to 18 {
            %97 = affine.load %55[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
            affine.store %97, %56[%arg2, %arg3, %arg4, %arg5] : memref<1x128x18x18xi8>
          }
        }
      }
    }
    %57 = memref.subview %56[0, 0, 1, 1] [1, 128, 16, 16] [1, 1, 1, 1] : memref<1x128x18x18xi8> to memref<1x128x16x16xi8, #map4>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %97 = affine.load %54[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8>
            affine.store %97, %57[%arg2, %arg3, %arg4, %arg5] : memref<1x128x16x16xi8, #map4>
          }
        }
      }
    }
    %58 = memref.alloca() : memref<1x256x8x8xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            affine.store %c0_i8, %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 128 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map3(%arg4, %arg7)
                  %98 = affine.apply #map3(%arg5, %arg8)
                  %99 = affine.load %56[%arg2, %arg6, %97, %98] : memref<1x128x18x18xi8>
                  %100 = affine.load %5[%arg3, %arg6, %arg7, %arg8] : memref<256x128x3x3xi8>
                  %101 = affine.load %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
                }
              }
            }
          }
        }
      }
    }
    %59 = memref.alloca() : memref<1x256x8x8xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %97 = affine.load %58[%c0, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %59[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
          }
        }
      }
    }
    %60 = memref.alloca() : memref<1x256x10x10xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 10 {
          affine.for %arg5 = 0 to 10 {
            affine.store %c0_i8, %60[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
          }
        }
      }
    }
    %61 = memref.alloca() : memref<1x256x10x10xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 10 {
          affine.for %arg5 = 0 to 10 {
            %97 = affine.load %60[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
            affine.store %97, %61[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
          }
        }
      }
    }
    %62 = memref.subview %61[0, 0, 1, 1] [1, 256, 8, 8] [1, 1, 1, 1] : memref<1x256x10x10xi8> to memref<1x256x8x8xi8, #map5>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %97 = affine.load %59[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            affine.store %97, %62[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8, #map5>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            affine.store %c0_i8, %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 256 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %61[%arg2, %arg6, %97, %98] : memref<1x256x10x10xi8>
                  %100 = affine.load %7[%arg3, %arg6, %arg7, %arg8] : memref<256x256x3x3xi8>
                  %101 = affine.load %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
                }
              }
            }
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            affine.store %c0_i8, %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 128 {
              affine.for %arg7 = 0 to 1 {
                affine.for %arg8 = 0 to 1 {
                  %97 = affine.apply #map3(%arg4, %arg7)
                  %98 = affine.apply #map3(%arg5, %arg8)
                  %99 = affine.load %54[%arg2, %arg6, %97, %98] : memref<1x128x16x16xi8>
                  %100 = affine.load %6[%arg3, %arg6, %arg7, %arg8] : memref<256x128x1x1xi8>
                  %101 = affine.load %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
                }
              }
            }
          }
        }
      }
    }
    %63 = memref.alloca() : memref<1x256x8x8xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %97 = affine.load %58[%c0, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            %98 = affine.load %58[%c0, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            %99 = arith.addi %97, %98 : i8
            affine.store %99, %63[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
          }
        }
      }
    }
    %64 = memref.alloca() : memref<1x256x8x8xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %97 = affine.load %63[%c0, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %64[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
          }
        }
      }
    }
    %65 = memref.alloca() : memref<1x256x10x10xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 10 {
          affine.for %arg5 = 0 to 10 {
            affine.store %c0_i8, %65[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
          }
        }
      }
    }
    %66 = memref.alloca() : memref<1x256x10x10xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 10 {
          affine.for %arg5 = 0 to 10 {
            %97 = affine.load %65[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
            affine.store %97, %66[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
          }
        }
      }
    }
    %67 = memref.subview %66[0, 0, 1, 1] [1, 256, 8, 8] [1, 1, 1, 1] : memref<1x256x10x10xi8> to memref<1x256x8x8xi8, #map5>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %97 = affine.load %64[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            affine.store %97, %67[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8, #map5>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            affine.store %c0_i8, %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 256 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %66[%arg2, %arg6, %97, %98] : memref<1x256x10x10xi8>
                  %100 = affine.load %7[%arg3, %arg6, %arg7, %arg8] : memref<256x256x3x3xi8>
                  %101 = affine.load %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
                }
              }
            }
          }
        }
      }
    }
    %68 = memref.alloca() : memref<1x256x8x8xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %97 = affine.load %58[%c0, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %68[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
          }
        }
      }
    }
    %69 = memref.alloca() : memref<1x256x10x10xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 10 {
          affine.for %arg5 = 0 to 10 {
            affine.store %c0_i8, %69[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
          }
        }
      }
    }
    %70 = memref.alloca() : memref<1x256x10x10xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 10 {
          affine.for %arg5 = 0 to 10 {
            %97 = affine.load %69[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
            affine.store %97, %70[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
          }
        }
      }
    }
    %71 = memref.subview %70[0, 0, 1, 1] [1, 256, 8, 8] [1, 1, 1, 1] : memref<1x256x10x10xi8> to memref<1x256x8x8xi8, #map5>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %97 = affine.load %68[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            affine.store %97, %71[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8, #map5>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            affine.store %c0_i8, %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 256 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %70[%arg2, %arg6, %97, %98] : memref<1x256x10x10xi8>
                  %100 = affine.load %7[%arg3, %arg6, %arg7, %arg8] : memref<256x256x3x3xi8>
                  %101 = affine.load %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %58[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
                }
              }
            }
          }
        }
      }
    }
    %72 = memref.alloca() : memref<1x256x8x8xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %97 = affine.load %58[%c0, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            %98 = affine.load %64[%c0, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            %99 = arith.addi %97, %98 : i8
            affine.store %99, %72[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
          }
        }
      }
    }
    %73 = memref.alloca() : memref<1x256x8x8xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %97 = affine.load %72[%c0, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %73[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
          }
        }
      }
    }
    %74 = memref.alloca() : memref<1x256x10x10xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 10 {
          affine.for %arg5 = 0 to 10 {
            affine.store %c0_i8, %74[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
          }
        }
      }
    }
    %75 = memref.alloca() : memref<1x256x10x10xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 10 {
          affine.for %arg5 = 0 to 10 {
            %97 = affine.load %74[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
            affine.store %97, %75[%arg2, %arg3, %arg4, %arg5] : memref<1x256x10x10xi8>
          }
        }
      }
    }
    %76 = memref.subview %75[0, 0, 1, 1] [1, 256, 8, 8] [1, 1, 1, 1] : memref<1x256x10x10xi8> to memref<1x256x8x8xi8, #map5>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 256 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %97 = affine.load %73[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8>
            affine.store %97, %76[%arg2, %arg3, %arg4, %arg5] : memref<1x256x8x8xi8, #map5>
          }
        }
      }
    }
    %77 = memref.alloca() : memref<1x512x4x4xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            affine.store %c0_i8, %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 256 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map3(%arg4, %arg7)
                  %98 = affine.apply #map3(%arg5, %arg8)
                  %99 = affine.load %75[%arg2, %arg6, %97, %98] : memref<1x256x10x10xi8>
                  %100 = affine.load %8[%arg3, %arg6, %arg7, %arg8] : memref<512x256x3x3xi8>
                  %101 = affine.load %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
                }
              }
            }
          }
        }
      }
    }
    %78 = memref.alloca() : memref<1x512x4x4xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            %97 = affine.load %77[%c0, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %78[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
          }
        }
      }
    }
    %79 = memref.alloca() : memref<1x512x6x6xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 6 {
          affine.for %arg5 = 0 to 6 {
            affine.store %c0_i8, %79[%arg2, %arg3, %arg4, %arg5] : memref<1x512x6x6xi8>
          }
        }
      }
    }
    %80 = memref.alloca() : memref<1x512x6x6xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 6 {
          affine.for %arg5 = 0 to 6 {
            %97 = affine.load %79[%arg2, %arg3, %arg4, %arg5] : memref<1x512x6x6xi8>
            affine.store %97, %80[%arg2, %arg3, %arg4, %arg5] : memref<1x512x6x6xi8>
          }
        }
      }
    }
    %81 = memref.subview %80[0, 0, 1, 1] [1, 512, 4, 4] [1, 1, 1, 1] : memref<1x512x6x6xi8> to memref<1x512x4x4xi8, #map6>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            %97 = affine.load %78[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            affine.store %97, %81[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8, #map6>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            affine.store %c0_i8, %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 512 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %80[%arg2, %arg6, %97, %98] : memref<1x512x6x6xi8>
                  %100 = affine.load %10[%arg3, %arg6, %arg7, %arg8] : memref<512x512x3x3xi8>
                  %101 = affine.load %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
                }
              }
            }
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            affine.store %c0_i8, %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 256 {
              affine.for %arg7 = 0 to 1 {
                affine.for %arg8 = 0 to 1 {
                  %97 = affine.apply #map3(%arg4, %arg7)
                  %98 = affine.apply #map3(%arg5, %arg8)
                  %99 = affine.load %73[%arg2, %arg6, %97, %98] : memref<1x256x8x8xi8>
                  %100 = affine.load %9[%arg3, %arg6, %arg7, %arg8] : memref<512x256x1x1xi8>
                  %101 = affine.load %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
                }
              }
            }
          }
        }
      }
    }
    %82 = memref.alloca() : memref<1x512x4x4xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            %97 = affine.load %77[%c0, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            %98 = affine.load %77[%c0, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            %99 = arith.addi %97, %98 : i8
            affine.store %99, %82[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
          }
        }
      }
    }
    %83 = memref.alloca() : memref<1x512x4x4xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            %97 = affine.load %82[%c0, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %83[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
          }
        }
      }
    }
    %84 = memref.alloca() : memref<1x512x6x6xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 6 {
          affine.for %arg5 = 0 to 6 {
            affine.store %c0_i8, %84[%arg2, %arg3, %arg4, %arg5] : memref<1x512x6x6xi8>
          }
        }
      }
    }
    %85 = memref.alloca() : memref<1x512x6x6xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 6 {
          affine.for %arg5 = 0 to 6 {
            %97 = affine.load %84[%arg2, %arg3, %arg4, %arg5] : memref<1x512x6x6xi8>
            affine.store %97, %85[%arg2, %arg3, %arg4, %arg5] : memref<1x512x6x6xi8>
          }
        }
      }
    }
    %86 = memref.subview %85[0, 0, 1, 1] [1, 512, 4, 4] [1, 1, 1, 1] : memref<1x512x6x6xi8> to memref<1x512x4x4xi8, #map6>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            %97 = affine.load %83[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            affine.store %97, %86[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8, #map6>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            affine.store %c0_i8, %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 512 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %85[%arg2, %arg6, %97, %98] : memref<1x512x6x6xi8>
                  %100 = affine.load %10[%arg3, %arg6, %arg7, %arg8] : memref<512x512x3x3xi8>
                  %101 = affine.load %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
                }
              }
            }
          }
        }
      }
    }
    %87 = memref.alloca() : memref<1x512x4x4xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            %97 = affine.load %77[%c0, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %87[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
          }
        }
      }
    }
    %88 = memref.alloca() : memref<1x512x6x6xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 6 {
          affine.for %arg5 = 0 to 6 {
            affine.store %c0_i8, %88[%arg2, %arg3, %arg4, %arg5] : memref<1x512x6x6xi8>
          }
        }
      }
    }
    %89 = memref.alloca() : memref<1x512x6x6xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 6 {
          affine.for %arg5 = 0 to 6 {
            %97 = affine.load %88[%arg2, %arg3, %arg4, %arg5] : memref<1x512x6x6xi8>
            affine.store %97, %89[%arg2, %arg3, %arg4, %arg5] : memref<1x512x6x6xi8>
          }
        }
      }
    }
    %90 = memref.subview %89[0, 0, 1, 1] [1, 512, 4, 4] [1, 1, 1, 1] : memref<1x512x6x6xi8> to memref<1x512x4x4xi8, #map6>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            %97 = affine.load %87[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            affine.store %97, %90[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8, #map6>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            affine.store %c0_i8, %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
          }
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 512 {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  %97 = affine.apply #map1(%arg4, %arg7)
                  %98 = affine.apply #map1(%arg5, %arg8)
                  %99 = affine.load %89[%arg2, %arg6, %97, %98] : memref<1x512x6x6xi8>
                  %100 = affine.load %10[%arg3, %arg6, %arg7, %arg8] : memref<512x512x3x3xi8>
                  %101 = affine.load %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
                  %102 = arith.muli %99, %100 : i8
                  %103 = arith.addi %101, %102 : i8
                  affine.store %103, %77[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
                }
              }
            }
          }
        }
      }
    }
    %91 = memref.alloca() : memref<1x512x4x4xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            %97 = affine.load %77[%c0, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            %98 = affine.load %83[%c0, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            %99 = arith.addi %97, %98 : i8
            affine.store %99, %91[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
          }
        }
      }
    }
    %92 = memref.alloca() : memref<1x512x4x4xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            %97 = affine.load %91[%c0, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            %98 = arith.cmpi ugt, %97, %c0_i8 : i8
            %99 = arith.select %98, %97, %c0_i8 : i8
            affine.store %99, %92[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
          }
        }
      }
    }
    %93 = memref.alloca() : memref<1x512xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.store %c0_i8, %93[%arg2, %arg3] : memref<1x512xi8>
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 4 {
          affine.for %arg5 = 0 to 4 {
            %97 = affine.load %92[%arg2, %arg3, %arg4, %arg5] : memref<1x512x4x4xi8>
            %98 = affine.load %93[%arg2, %arg3] : memref<1x512xi8>
            %99 = arith.addi %98, %97 : i8
            affine.store %99, %93[%arg2, %arg3] : memref<1x512xi8>
          }
        }
      }
    }
    %94 = memref.alloca() : memref<1x512x1x1xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 512 {
        affine.for %arg4 = 0 to 1 {
          affine.for %arg5 = 0 to 1 {
            %97 = affine.load %93[%arg2, %arg3] : memref<1x512xi8>
            %98 = arith.divui %97, %c16_i8 : i8
            affine.store %98, %94[%arg2, %arg3, %arg4, %arg5] : memref<1x512x1x1xi8>
          }
        }
      }
    }
    %95 = memref.collapse_shape %94 [[0], [1, 2, 3]] : memref<1x512x1x1xi8> into memref<1x512xi8>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 10 {
        %97 = affine.load %12[%arg3] : memref<10xi8>
        affine.store %97, %arg1[%arg2, %arg3] : memref<1x10xi8>
      }
    }
    %96 = memref.alloca() : memref<512x10xi8>
    affine.for %arg2 = 0 to 512 {
      affine.for %arg3 = 0 to 10 {
        %97 = affine.load %11[%arg3, %arg2] : memref<10x512xi8>
        affine.store %97, %96[%arg2, %arg3] : memref<512x10xi8>
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 10 {
        affine.for %arg4 = 0 to 512 {
          %97 = affine.load %95[%arg2, %arg4] : memref<1x512xi8>
          %98 = affine.load %96[%arg4, %arg3] : memref<512x10xi8>
          %99 = affine.load %arg1[%arg2, %arg3] : memref<1x10xi8>
          %100 = arith.muli %97, %98 : i8
          %101 = arith.addi %99, %100 : i8
          affine.store %101, %arg1[%arg2, %arg3] : memref<1x10xi8>
        }
      }
    }
    return
  }
}
