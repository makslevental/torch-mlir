; ModuleID = '/home/mlevental/dev_projects/torch-mlir/hls/vitis_stuff/matmul.llvm.pre.ll'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

define void @forward(float* %arg_2, float* %arg_3, i64 %arg_4, i64 %arg_5, i64 %arg_6, i64 %arg_7, i64 %arg_8, float* %arg_9, float* %arg_10, i64 %arg_11, i64 %arg_12, i64 %arg_13, i64 %arg_14, i64 %arg_15, float* %arg_16, float* %arg_17, i64 %arg_18, i64 %arg_19, i64 %arg_20, i64 %arg_21, i64 %arg_22) #0 {
bb_0:
  br label %bb_1

bb_1:                                             ; preds = %bb_8, %bb_0
  %val_1 = phi i64 [ %val_26, %bb_8 ], [ 0, %bb_0 ]
  %val_2 = icmp slt i64 %val_1, 4
  br i1 %val_2, label %bb_2, label %bb_9

bb_2:                                             ; preds = %bb_1
  br label %bb_3

bb_3:                                             ; preds = %bb_7, %bb_2
  %val_3 = phi i64 [ %val_25, %bb_7 ], [ 0, %bb_2 ]
  %val_4 = icmp slt i64 %val_3, 10
  br i1 %val_4, label %bb_4, label %bb_8

bb_4:                                             ; preds = %bb_3
  br label %bb_5

bb_5:                                             ; preds = %bb_6, %bb_4
  %val_5 = phi i64 [ %val_24, %bb_6 ], [ 0, %bb_4 ]
  %val_6 = icmp slt i64 %val_5, 5
  br i1 %val_6, label %bb_6, label %bb_7

bb_6:                                             ; preds = %bb_5
  %val_7 = mul i64 %val_1, 5
  %val_8 = add i64 %val_7, %val_5
  %val_9 = getelementptr float, float* %arg_3, i64 %val_8
  %val_10 = load float, float* %val_9, align 4
  %val_11 = mul i64 %val_5, 10
  %val_12 = add i64 %val_11, %val_3
  %val_13 = getelementptr float, float* %arg_10, i64 %val_12
  %val_14 = load float, float* %val_13, align 4
  %val_15 = mul i64 %val_1, 10
  %val_16 = add i64 %val_15, %val_3
  %val_17 = getelementptr float, float* %arg_17, i64 %val_16
  %val_18 = load float, float* %val_17, align 4
  %val_19 = fmul float %val_10, %val_14
  %val_20 = fadd float %val_18, %val_19
  %val_21 = mul i64 %val_1, 10
  %val_22 = add i64 %val_21, %val_3
  %val_23 = getelementptr float, float* %arg_17, i64 %val_22
  store float %val_20, float* %val_23, align 4
  %val_24 = add i64 %val_5, 1
  br label %bb_5

bb_7:                                             ; preds = %bb_5
  %val_25 = add i64 %val_3, 1
  br label %bb_3

bb_8:                                             ; preds = %bb_3
  %val_26 = add i64 %val_1, 1
  br label %bb_1

bb_9:                                             ; preds = %bb_1
  ret void
}

attributes #0 = { "fpga.top.func"="forward" }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
