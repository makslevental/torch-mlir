// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2021.2 (64-bit)
// Version: 2021.2
// Copyright (C) Copyright 1986-2021 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module forward_forward_Pipeline_14 (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        p_reload1122,
        arg_3,
        arg_3_ap_vld,
        p_out,
        p_out_ap_vld,
        grp_fu_758_p_din0,
        grp_fu_758_p_din1,
        grp_fu_758_p_dout0,
        grp_fu_758_p_ce
);

parameter    ap_ST_fsm_pp0_stage0 = 31'd1;
parameter    ap_ST_fsm_pp0_stage1 = 31'd2;
parameter    ap_ST_fsm_pp0_stage2 = 31'd4;
parameter    ap_ST_fsm_pp0_stage3 = 31'd8;
parameter    ap_ST_fsm_pp0_stage4 = 31'd16;
parameter    ap_ST_fsm_pp0_stage5 = 31'd32;
parameter    ap_ST_fsm_pp0_stage6 = 31'd64;
parameter    ap_ST_fsm_pp0_stage7 = 31'd128;
parameter    ap_ST_fsm_pp0_stage8 = 31'd256;
parameter    ap_ST_fsm_pp0_stage9 = 31'd512;
parameter    ap_ST_fsm_pp0_stage10 = 31'd1024;
parameter    ap_ST_fsm_pp0_stage11 = 31'd2048;
parameter    ap_ST_fsm_pp0_stage12 = 31'd4096;
parameter    ap_ST_fsm_pp0_stage13 = 31'd8192;
parameter    ap_ST_fsm_pp0_stage14 = 31'd16384;
parameter    ap_ST_fsm_pp0_stage15 = 31'd32768;
parameter    ap_ST_fsm_pp0_stage16 = 31'd65536;
parameter    ap_ST_fsm_pp0_stage17 = 31'd131072;
parameter    ap_ST_fsm_pp0_stage18 = 31'd262144;
parameter    ap_ST_fsm_pp0_stage19 = 31'd524288;
parameter    ap_ST_fsm_pp0_stage20 = 31'd1048576;
parameter    ap_ST_fsm_pp0_stage21 = 31'd2097152;
parameter    ap_ST_fsm_pp0_stage22 = 31'd4194304;
parameter    ap_ST_fsm_pp0_stage23 = 31'd8388608;
parameter    ap_ST_fsm_pp0_stage24 = 31'd16777216;
parameter    ap_ST_fsm_pp0_stage25 = 31'd33554432;
parameter    ap_ST_fsm_pp0_stage26 = 31'd67108864;
parameter    ap_ST_fsm_pp0_stage27 = 31'd134217728;
parameter    ap_ST_fsm_pp0_stage28 = 31'd268435456;
parameter    ap_ST_fsm_pp0_stage29 = 31'd536870912;
parameter    ap_ST_fsm_pp0_stage30 = 31'd1073741824;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [7:0] p_reload1122;
output  [7:0] arg_3;
output   arg_3_ap_vld;
output  [7:0] p_out;
output   p_out_ap_vld;
output  [31:0] grp_fu_758_p_din0;
output  [31:0] grp_fu_758_p_din1;
input  [31:0] grp_fu_758_p_dout0;
output   grp_fu_758_p_ce;

reg ap_idle;
reg arg_3_ap_vld;
reg p_out_ap_vld;

(* fsm_encoding = "none" *) reg   [30:0] ap_CS_fsm;
wire    ap_CS_fsm_pp0_stage0;
reg    ap_enable_reg_pp0_iter0;
reg    ap_enable_reg_pp0_iter1;
reg    ap_idle_pp0;
wire    ap_block_state1_pp0_stage0_iter0;
wire    ap_block_state32_pp0_stage0_iter1;
wire    ap_block_pp0_stage0_subdone;
wire   [0:0] exitcond_flatten382_fu_78_p2;
reg    ap_condition_exit_pp0_iter0_stage0;
wire    ap_loop_exit_ready;
reg    ap_ready_int;
wire    ap_CS_fsm_pp0_stage30;
wire    ap_block_state31_pp0_stage30_iter0;
wire    ap_block_pp0_stage30_subdone;
reg   [0:0] exitcond_flatten382_reg_138;
wire    ap_block_pp0_stage0_11001;
wire   [11:0] indvar_flatten_next381_fu_84_p2;
reg   [11:0] indvar_flatten_next381_reg_142;
wire   [31:0] empty_95_fu_97_p1;
wire   [7:0] empty_97_fu_110_p1;
reg   [7:0] empty_97_reg_152;
wire    ap_block_pp0_stage30_11001;
reg    ap_enable_reg_pp0_iter0_reg;
reg   [7:0] empty_fu_32;
reg   [7:0] ap_sig_allocacmp_p_load2;
wire    ap_block_pp0_stage0;
wire    ap_loop_init;
reg   [11:0] indvar_flatten380_fu_36;
wire    ap_CS_fsm_pp0_stage1;
wire    ap_block_state2_pp0_stage1_iter0;
wire    ap_block_pp0_stage1_11001;
reg   [11:0] ap_sig_allocacmp_indvar_flatten380_load;
wire    ap_block_pp0_stage30_01001;
wire    ap_block_pp0_stage0_01001;
wire   [31:0] p_cast31_fu_93_p1;
wire    ap_block_pp0_stage30;
wire   [31:0] empty_96_fu_106_p1;
reg    ap_done_reg;
wire    ap_continue_int;
reg    ap_done_int;
reg   [30:0] ap_NS_fsm;
reg    ap_idle_pp0_1to1;
wire    ap_block_pp0_stage1_subdone;
wire    ap_block_state3_pp0_stage2_iter0;
wire    ap_block_pp0_stage2_subdone;
wire    ap_block_state4_pp0_stage3_iter0;
wire    ap_block_pp0_stage3_subdone;
wire    ap_block_state5_pp0_stage4_iter0;
wire    ap_block_pp0_stage4_subdone;
wire    ap_block_state6_pp0_stage5_iter0;
wire    ap_block_pp0_stage5_subdone;
wire    ap_block_state7_pp0_stage6_iter0;
wire    ap_block_pp0_stage6_subdone;
wire    ap_block_state8_pp0_stage7_iter0;
wire    ap_block_pp0_stage7_subdone;
wire    ap_block_state9_pp0_stage8_iter0;
wire    ap_block_pp0_stage8_subdone;
wire    ap_block_state10_pp0_stage9_iter0;
wire    ap_block_pp0_stage9_subdone;
wire    ap_block_state11_pp0_stage10_iter0;
wire    ap_block_pp0_stage10_subdone;
wire    ap_block_state12_pp0_stage11_iter0;
wire    ap_block_pp0_stage11_subdone;
wire    ap_block_state13_pp0_stage12_iter0;
wire    ap_block_pp0_stage12_subdone;
wire    ap_block_state14_pp0_stage13_iter0;
wire    ap_block_pp0_stage13_subdone;
wire    ap_block_state15_pp0_stage14_iter0;
wire    ap_block_pp0_stage14_subdone;
wire    ap_block_state16_pp0_stage15_iter0;
wire    ap_block_pp0_stage15_subdone;
wire    ap_block_state17_pp0_stage16_iter0;
wire    ap_block_pp0_stage16_subdone;
wire    ap_block_state18_pp0_stage17_iter0;
wire    ap_block_pp0_stage17_subdone;
wire    ap_block_state19_pp0_stage18_iter0;
wire    ap_block_pp0_stage18_subdone;
wire    ap_block_state20_pp0_stage19_iter0;
wire    ap_block_pp0_stage19_subdone;
wire    ap_block_state21_pp0_stage20_iter0;
wire    ap_block_pp0_stage20_subdone;
wire    ap_block_state22_pp0_stage21_iter0;
wire    ap_block_pp0_stage21_subdone;
wire    ap_block_state23_pp0_stage22_iter0;
wire    ap_block_pp0_stage22_subdone;
wire    ap_block_state24_pp0_stage23_iter0;
wire    ap_block_pp0_stage23_subdone;
wire    ap_block_state25_pp0_stage24_iter0;
wire    ap_block_pp0_stage24_subdone;
wire    ap_block_state26_pp0_stage25_iter0;
wire    ap_block_pp0_stage25_subdone;
wire    ap_block_state27_pp0_stage26_iter0;
wire    ap_block_pp0_stage26_subdone;
wire    ap_block_state28_pp0_stage27_iter0;
wire    ap_block_pp0_stage27_subdone;
wire    ap_block_state29_pp0_stage28_iter0;
wire    ap_block_pp0_stage28_subdone;
wire    ap_block_state30_pp0_stage29_iter0;
wire    ap_block_pp0_stage29_subdone;
wire    ap_enable_pp0;
wire    ap_start_int;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 31'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_enable_reg_pp0_iter0_reg = 1'b0;
#0 ap_done_reg = 1'b0;
end

forward_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(ap_start),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_start_int(ap_start_int),
    .ap_loop_init(ap_loop_init),
    .ap_ready_int(ap_ready_int),
    .ap_loop_exit_ready(ap_condition_exit_pp0_iter0_stage0),
    .ap_loop_exit_done(ap_done_int),
    .ap_continue_int(ap_continue_int),
    .ap_done_int(ap_done_int)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_pp0_stage0;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_done_reg <= 1'b0;
    end else begin
        if ((ap_continue_int == 1'b1)) begin
            ap_done_reg <= 1'b0;
        end else if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_done_reg <= 1'b1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter0_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_condition_exit_pp0_iter0_stage0)) begin
            ap_enable_reg_pp0_iter0_reg <= 1'b0;
        end else if ((1'b1 == ap_CS_fsm_pp0_stage0)) begin
            ap_enable_reg_pp0_iter0_reg <= ap_start_int;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter1 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
            ap_enable_reg_pp0_iter1 <= 1'b0;
        end else if (((1'b0 == ap_block_pp0_stage30_subdone) & (1'b1 == ap_CS_fsm_pp0_stage30))) begin
            ap_enable_reg_pp0_iter1 <= ap_enable_reg_pp0_iter0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            empty_fu_32 <= p_reload1122;
        end else if ((ap_enable_reg_pp0_iter1 == 1'b1)) begin
            empty_fu_32 <= empty_97_reg_152;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0) & (ap_loop_init == 1'b1))) begin
        indvar_flatten380_fu_36 <= 12'd0;
    end else if (((exitcond_flatten382_reg_138 == 1'd0) & (1'b0 == ap_block_pp0_stage1_11001) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage1))) begin
        indvar_flatten380_fu_36 <= indvar_flatten_next381_reg_142;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond_flatten382_reg_138 == 1'd0) & (1'b0 == ap_block_pp0_stage30_11001) & (1'b1 == ap_CS_fsm_pp0_stage30))) begin
        empty_97_reg_152 <= empty_97_fu_110_p1;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        exitcond_flatten382_reg_138 <= exitcond_flatten382_fu_78_p2;
        indvar_flatten_next381_reg_142 <= indvar_flatten_next381_fu_84_p2;
    end
end

always @ (*) begin
    if (((exitcond_flatten382_fu_78_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b1;
    end else begin
        ap_condition_exit_pp0_iter0_stage0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_loop_exit_ready == 1'b1) & (1'b0 == ap_block_pp0_stage0_subdone) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_done_int = 1'b1;
    end else begin
        ap_done_int = ap_done_reg;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_pp0_stage0)) begin
        ap_enable_reg_pp0_iter0 = ap_start_int;
    end else begin
        ap_enable_reg_pp0_iter0 = ap_enable_reg_pp0_iter0_reg;
    end
end

always @ (*) begin
    if (((ap_idle_pp0 == 1'b1) & (ap_start_int == 1'b0) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if ((ap_enable_reg_pp0_iter1 == 1'b0)) begin
        ap_idle_pp0_1to1 = 1'b1;
    end else begin
        ap_idle_pp0_1to1 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage30_subdone) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage30))) begin
        ap_ready_int = 1'b1;
    end else begin
        ap_ready_int = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (1'b1 == ap_CS_fsm_pp0_stage0) & (ap_loop_init == 1'b1))) begin
        ap_sig_allocacmp_indvar_flatten380_load = 12'd0;
    end else begin
        ap_sig_allocacmp_indvar_flatten380_load = indvar_flatten380_fu_36;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage0) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        if ((ap_loop_init == 1'b1)) begin
            ap_sig_allocacmp_p_load2 = p_reload1122;
        end else if ((ap_enable_reg_pp0_iter1 == 1'b1)) begin
            ap_sig_allocacmp_p_load2 = empty_97_reg_152;
        end else begin
            ap_sig_allocacmp_p_load2 = empty_fu_32;
        end
    end else begin
        ap_sig_allocacmp_p_load2 = empty_fu_32;
    end
end

always @ (*) begin
    if (((exitcond_flatten382_reg_138 == 1'd0) & (1'b0 == ap_block_pp0_stage30_11001) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage30))) begin
        arg_3_ap_vld = 1'b1;
    end else begin
        arg_3_ap_vld = 1'b0;
    end
end

always @ (*) begin
    if (((exitcond_flatten382_fu_78_p2 == 1'd1) & (1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        p_out_ap_vld = 1'b1;
    end else begin
        p_out_ap_vld = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_pp0_stage0 : begin
            if ((1'b1 == ap_condition_exit_pp0_iter0_stage0)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else if ((~((ap_start_int == 1'b0) & (ap_idle_pp0_1to1 == 1'b1)) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end
        end
        ap_ST_fsm_pp0_stage1 : begin
            if ((1'b0 == ap_block_pp0_stage1_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage1;
            end
        end
        ap_ST_fsm_pp0_stage2 : begin
            if ((1'b0 == ap_block_pp0_stage2_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage2;
            end
        end
        ap_ST_fsm_pp0_stage3 : begin
            if ((1'b0 == ap_block_pp0_stage3_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage3;
            end
        end
        ap_ST_fsm_pp0_stage4 : begin
            if ((1'b0 == ap_block_pp0_stage4_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage4;
            end
        end
        ap_ST_fsm_pp0_stage5 : begin
            if ((1'b0 == ap_block_pp0_stage5_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage6;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage5;
            end
        end
        ap_ST_fsm_pp0_stage6 : begin
            if ((1'b0 == ap_block_pp0_stage6_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage7;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage6;
            end
        end
        ap_ST_fsm_pp0_stage7 : begin
            if ((1'b0 == ap_block_pp0_stage7_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage8;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage7;
            end
        end
        ap_ST_fsm_pp0_stage8 : begin
            if ((1'b0 == ap_block_pp0_stage8_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage9;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage8;
            end
        end
        ap_ST_fsm_pp0_stage9 : begin
            if ((1'b0 == ap_block_pp0_stage9_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage10;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage9;
            end
        end
        ap_ST_fsm_pp0_stage10 : begin
            if ((1'b0 == ap_block_pp0_stage10_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage11;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage10;
            end
        end
        ap_ST_fsm_pp0_stage11 : begin
            if ((1'b0 == ap_block_pp0_stage11_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage12;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage11;
            end
        end
        ap_ST_fsm_pp0_stage12 : begin
            if ((1'b0 == ap_block_pp0_stage12_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage13;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage12;
            end
        end
        ap_ST_fsm_pp0_stage13 : begin
            if ((1'b0 == ap_block_pp0_stage13_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage14;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage13;
            end
        end
        ap_ST_fsm_pp0_stage14 : begin
            if ((1'b0 == ap_block_pp0_stage14_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage15;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage14;
            end
        end
        ap_ST_fsm_pp0_stage15 : begin
            if ((1'b0 == ap_block_pp0_stage15_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage16;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage15;
            end
        end
        ap_ST_fsm_pp0_stage16 : begin
            if ((1'b0 == ap_block_pp0_stage16_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage17;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage16;
            end
        end
        ap_ST_fsm_pp0_stage17 : begin
            if ((1'b0 == ap_block_pp0_stage17_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage18;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage17;
            end
        end
        ap_ST_fsm_pp0_stage18 : begin
            if ((1'b0 == ap_block_pp0_stage18_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage19;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage18;
            end
        end
        ap_ST_fsm_pp0_stage19 : begin
            if ((1'b0 == ap_block_pp0_stage19_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage20;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage19;
            end
        end
        ap_ST_fsm_pp0_stage20 : begin
            if ((1'b0 == ap_block_pp0_stage20_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage21;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage20;
            end
        end
        ap_ST_fsm_pp0_stage21 : begin
            if ((1'b0 == ap_block_pp0_stage21_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage22;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage21;
            end
        end
        ap_ST_fsm_pp0_stage22 : begin
            if ((1'b0 == ap_block_pp0_stage22_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage23;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage22;
            end
        end
        ap_ST_fsm_pp0_stage23 : begin
            if ((1'b0 == ap_block_pp0_stage23_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage24;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage23;
            end
        end
        ap_ST_fsm_pp0_stage24 : begin
            if ((1'b0 == ap_block_pp0_stage24_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage25;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage24;
            end
        end
        ap_ST_fsm_pp0_stage25 : begin
            if ((1'b0 == ap_block_pp0_stage25_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage26;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage25;
            end
        end
        ap_ST_fsm_pp0_stage26 : begin
            if ((1'b0 == ap_block_pp0_stage26_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage27;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage26;
            end
        end
        ap_ST_fsm_pp0_stage27 : begin
            if ((1'b0 == ap_block_pp0_stage27_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage28;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage27;
            end
        end
        ap_ST_fsm_pp0_stage28 : begin
            if ((1'b0 == ap_block_pp0_stage28_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage29;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage28;
            end
        end
        ap_ST_fsm_pp0_stage29 : begin
            if ((1'b0 == ap_block_pp0_stage29_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage30;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage29;
            end
        end
        ap_ST_fsm_pp0_stage30 : begin
            if ((1'b0 == ap_block_pp0_stage30_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage30;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_pp0_stage1 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_pp0_stage30 = ap_CS_fsm[32'd30];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage0_01001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage0_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage0_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage10_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage11_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage12_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage13_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage14_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage15_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage16_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage17_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage18_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage19_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage1_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage1_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage20_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage21_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage22_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage23_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage24_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage25_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage26_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage27_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage28_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage29_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage2_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage30 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage30_01001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage30_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage30_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage3_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage4_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage5_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage6_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage7_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage8_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage9_subdone = ~(1'b1 == 1'b1);

assign ap_block_state10_pp0_stage9_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state11_pp0_stage10_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state12_pp0_stage11_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state13_pp0_stage12_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state14_pp0_stage13_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state15_pp0_stage14_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state16_pp0_stage15_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state17_pp0_stage16_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state18_pp0_stage17_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state19_pp0_stage18_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state1_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state20_pp0_stage19_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state21_pp0_stage20_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state22_pp0_stage21_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state23_pp0_stage22_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state24_pp0_stage23_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state25_pp0_stage24_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state26_pp0_stage25_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state27_pp0_stage26_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state28_pp0_stage27_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state29_pp0_stage28_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state2_pp0_stage1_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state30_pp0_stage29_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state31_pp0_stage30_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state32_pp0_stage0_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state3_pp0_stage2_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state4_pp0_stage3_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state5_pp0_stage4_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state6_pp0_stage5_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state7_pp0_stage6_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state8_pp0_stage7_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state9_pp0_stage8_iter0 = ~(1'b1 == 1'b1);

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_loop_exit_ready = ap_condition_exit_pp0_iter0_stage0;

assign arg_3 = empty_96_fu_106_p1[7:0];

assign empty_95_fu_97_p1 = p_cast31_fu_93_p1;

assign empty_96_fu_106_p1 = grp_fu_758_p_dout0;

assign empty_97_fu_110_p1 = empty_96_fu_106_p1[7:0];

assign exitcond_flatten382_fu_78_p2 = ((ap_sig_allocacmp_indvar_flatten380_load == 12'd2592) ? 1'b1 : 1'b0);

assign grp_fu_758_p_ce = 1'b1;

assign grp_fu_758_p_din0 = 32'd0;

assign grp_fu_758_p_din1 = empty_95_fu_97_p1;

assign indvar_flatten_next381_fu_84_p2 = (ap_sig_allocacmp_indvar_flatten380_load + 12'd1);

assign p_cast31_fu_93_p1 = ap_sig_allocacmp_p_load2;

assign p_out = empty_fu_32;

endmodule //forward_forward_Pipeline_14
