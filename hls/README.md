# IRs

## Torch
```mlir
module attributes {llvm.data_layout = "", torch.debug_module_name = "MatmulDotOut"}  {
  func @forward(%arg0: !torch.vtensor<[4,5],f32>, %arg1: !torch.vtensor<[5,10],f32>, %arg2: !torch.vtensor<[4,10],f32>) {
    %0 = torch.aten.mm.out %arg0, %arg1, %arg2 : !torch.vtensor<[4,5],f32>, !torch.vtensor<[5,10],f32>, !torch.vtensor<[4,10],f32> -> !torch.vtensor<[4,10],f32>
    llvm.return
  }
}
```

## LinAlg

```mlir
module attributes {torch.debug_module_name = "MatmulDotOut"}  {
  func @forward(%arg0: tensor<4x5xf32>, %arg1: tensor<5x10xf32>, %arg2: tensor<4x10xf32>) -> tensor<4x10xf32> {
    %0 = linalg.matmul {variant = "out"} ins(%arg0, %arg1 : tensor<4x5xf32>, tensor<5x10xf32>) outs(%arg2 : tensor<4x10xf32>) -> tensor<4x10xf32>
    return %0 : tensor<4x10xf32>
  }
}
```

note `{variant = "out"}`.

## Bufferized

```mlir
module attributes {torch.debug_module_name = "MatmulDotOut"}  {
  func @forward(%arg0: memref<4x5xf32>, %arg1: memref<5x10xf32>, %arg2: memref<4x10xf32>) {
    linalg.matmul {variant = "out"} ins(%arg0, %arg1 : memref<4x5xf32>, memref<5x10xf32>) outs(%arg2 : memref<4x10xf32>)
    return
  }
}
```

note no return.

## LLVM

```mlir
module attributes {llvm.data_layout = "", torch.debug_module_name = "MatmulDotOut"}  {
  llvm.func @forward(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<f32>, %arg15: !llvm.ptr<f32>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg14, %16[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.insertvalue %arg15, %17[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %arg16, %18[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.insertvalue %arg17, %19[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %arg19, %20[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.insertvalue %arg18, %21[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %23 = llvm.insertvalue %arg20, %22[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.mlir.constant(4 : index) : i64
    %25 = llvm.mlir.constant(5 : index) : i64
    %26 = llvm.mlir.constant(10 : index) : i64
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%27 : i64)
  ^bb1(%29: i64):  // 2 preds: ^bb0, ^bb8
    %30 = llvm.icmp "slt" %29, %24 : i64
    llvm.cond_br %30, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%27 : i64)
  ^bb3(%31: i64):  // 2 preds: ^bb2, ^bb7
    %32 = llvm.icmp "slt" %31, %26 : i64
    llvm.cond_br %32, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%27 : i64)
  ^bb5(%33: i64):  // 2 preds: ^bb4, ^bb6
    %34 = llvm.icmp "slt" %33, %25 : i64
    llvm.cond_br %34, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %35 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.mlir.constant(5 : index) : i64
    %37 = llvm.mul %29, %36  : i64
    %38 = llvm.add %37, %33  : i64
    %39 = llvm.getelementptr %35[%38] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %40 = llvm.load %39 : !llvm.ptr<f32>
    %41 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %42 = llvm.mlir.constant(10 : index) : i64
    %43 = llvm.mul %33, %42  : i64
    %44 = llvm.add %43, %31  : i64
    %45 = llvm.getelementptr %41[%44] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %46 = llvm.load %45 : !llvm.ptr<f32>
    %47 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.mlir.constant(10 : index) : i64
    %49 = llvm.mul %29, %48  : i64
    %50 = llvm.add %49, %31  : i64
    %51 = llvm.getelementptr %47[%50] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %52 = llvm.load %51 : !llvm.ptr<f32>
    %53 = llvm.fmul %40, %46  : f32
    %54 = llvm.fadd %52, %53  : f32
    %55 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %56 = llvm.mlir.constant(10 : index) : i64
    %57 = llvm.mul %29, %56  : i64
    %58 = llvm.add %57, %31  : i64
    %59 = llvm.getelementptr %55[%58] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %54, %59 : !llvm.ptr<f32>
    %60 = llvm.add %33, %28  : i64
    llvm.br ^bb5(%60 : i64)
  ^bb7:  // pred: ^bb5
    %61 = llvm.add %31, %28  : i64
    llvm.br ^bb3(%61 : i64)
  ^bb8:  // pred: ^bb3
    %62 = llvm.add %29, %28  : i64
    llvm.br ^bb1(%62 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return
  }
}
```

## LLVMIR

```mlir
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

```

## Verilog

```systemverilog
`timescale 1 ns / 1 ps 

module forward (
        ap_local_block,
        ap_local_deadlock,
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        arg_2,
        arg_3,
        arg_4,
        arg_5,
        arg_6,
        arg_7,
        arg_8,
        arg_9,
        arg_10,
        arg_11,
        arg_12,
        arg_13,
        arg_14,
        arg_15,
        arg_16,
        arg_17_i,
        arg_17_o,
        arg_17_o_ap_vld,
        arg_18,
        arg_19,
        arg_20,
        arg_21,
        arg_22
);

parameter    ap_ST_fsm_state1 = 10'd1;
parameter    ap_ST_fsm_state2 = 10'd2;
parameter    ap_ST_fsm_state3 = 10'd4;
parameter    ap_ST_fsm_state4 = 10'd8;
parameter    ap_ST_fsm_state5 = 10'd16;
parameter    ap_ST_fsm_state6 = 10'd32;
parameter    ap_ST_fsm_state7 = 10'd64;
parameter    ap_ST_fsm_state8 = 10'd128;
parameter    ap_ST_fsm_state9 = 10'd256;
parameter    ap_ST_fsm_state10 = 10'd512;

output   ap_local_block;
output   ap_local_deadlock;
input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [31:0] arg_2;
input  [31:0] arg_3;
input  [63:0] arg_4;
input  [63:0] arg_5;
input  [63:0] arg_6;
input  [63:0] arg_7;
input  [63:0] arg_8;
input  [31:0] arg_9;
input  [31:0] arg_10;
input  [63:0] arg_11;
input  [63:0] arg_12;
input  [63:0] arg_13;
input  [63:0] arg_14;
input  [63:0] arg_15;
input  [31:0] arg_16;
input  [31:0] arg_17_i;
output  [31:0] arg_17_o;
output   arg_17_o_ap_vld;
input  [63:0] arg_18;
input  [63:0] arg_19;
input  [63:0] arg_20;
input  [63:0] arg_21;
input  [63:0] arg_22;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg[31:0] arg_17_o;

(* fsm_encoding = "none" *) reg   [9:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire   [31:0] grp_fu_79_p2;
reg   [31:0] val_s_reg_103;
wire    ap_CS_fsm_state8;
wire    grp_forward_Pipeline_1_fu_72_ap_start;
wire    grp_forward_Pipeline_1_fu_72_ap_done;
wire    grp_forward_Pipeline_1_fu_72_ap_idle;
wire    grp_forward_Pipeline_1_fu_72_ap_ready;
wire   [31:0] grp_forward_Pipeline_1_fu_72_arg_17_o;
wire    grp_forward_Pipeline_1_fu_72_arg_17_o_ap_vld;
reg    grp_forward_Pipeline_1_fu_72_ap_start_reg;
wire    ap_CS_fsm_state9;
wire    ap_CS_fsm_state10;
wire   [31:0] grp_fu_79_p0;
wire   [31:0] grp_fu_79_p1;
reg   [9:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
wire    ap_ST_fsm_state6_blk;
wire    ap_ST_fsm_state7_blk;
wire    ap_ST_fsm_state8_blk;
wire    ap_ST_fsm_state9_blk;
reg    ap_ST_fsm_state10_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 10'd1;
#0 grp_forward_Pipeline_1_fu_72_ap_start_reg = 1'b0;
end

forward_forward_Pipeline_1 grp_forward_Pipeline_1_fu_72(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_forward_Pipeline_1_fu_72_ap_start),
    .ap_done(grp_forward_Pipeline_1_fu_72_ap_done),
    .ap_idle(grp_forward_Pipeline_1_fu_72_ap_idle),
    .ap_ready(grp_forward_Pipeline_1_fu_72_ap_ready),
    .arg_17_i(arg_17_i),
    .arg_17_o(grp_forward_Pipeline_1_fu_72_arg_17_o),
    .arg_17_o_ap_vld(grp_forward_Pipeline_1_fu_72_arg_17_o_ap_vld),
    .val_s(val_s_reg_103)
);

forward_fmul_32ns_32ns_32_8_max_dsp_1 #(
    .ID( 1 ),
    .NUM_STAGE( 8 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .dout_WIDTH( 32 ))
fmul_32ns_32ns_32_8_max_dsp_1_U5(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(grp_fu_79_p0),
    .din1(grp_fu_79_p1),
    .ce(1'b1),
    .dout(grp_fu_79_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_forward_Pipeline_1_fu_72_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state9)) begin
            grp_forward_Pipeline_1_fu_72_ap_start_reg <= 1'b1;
        end else if ((grp_forward_Pipeline_1_fu_72_ap_ready == 1'b1)) begin
            grp_forward_Pipeline_1_fu_72_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state8)) begin
        val_s_reg_103 <= grp_fu_79_p2;
    end
end

always @ (*) begin
    if ((grp_forward_Pipeline_1_fu_72_ap_done == 1'b0)) begin
        ap_ST_fsm_state10_blk = 1'b1;
    end else begin
        ap_ST_fsm_state10_blk = 1'b0;
    end
end

always @ (*) begin
    if ((ap_start == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

assign ap_ST_fsm_state3_blk = 1'b0;

assign ap_ST_fsm_state4_blk = 1'b0;

assign ap_ST_fsm_state5_blk = 1'b0;

assign ap_ST_fsm_state6_blk = 1'b0;

assign ap_ST_fsm_state7_blk = 1'b0;

assign ap_ST_fsm_state8_blk = 1'b0;

assign ap_ST_fsm_state9_blk = 1'b0;

always @ (*) begin
    if (((grp_forward_Pipeline_1_fu_72_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state10))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((grp_forward_Pipeline_1_fu_72_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state10))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((grp_forward_Pipeline_1_fu_72_arg_17_o_ap_vld == 1'b1) & (1'b1 == ap_CS_fsm_state10))) begin
        arg_17_o = grp_forward_Pipeline_1_fu_72_arg_17_o;
    end else begin
        arg_17_o = arg_17_i;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            if (((grp_forward_Pipeline_1_fu_72_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state10))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state10 = ap_CS_fsm[32'd9];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

assign ap_local_block = 1'b0;

assign ap_local_deadlock = 1'b0;

assign arg_17_o_ap_vld = grp_forward_Pipeline_1_fu_72_arg_17_o_ap_vld;

assign grp_forward_Pipeline_1_fu_72_ap_start = grp_forward_Pipeline_1_fu_72_ap_start_reg;

assign grp_fu_79_p0 = arg_3;

assign grp_fu_79_p1 = arg_10;

endmodule //forward

```

# Test it out

Set `VITIS_DIR="/home/mlevental/dev_projects/Xilinx/Vitis_HLS/2021.2"` in `torchmlir.sh` and run `e2e.sh`