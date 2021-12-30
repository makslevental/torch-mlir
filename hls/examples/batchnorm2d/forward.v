// ==============================================================
// RTL generated by Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2021.2 (64-bit)
// Version: 2021.2
// Copyright (C) Copyright 1986-2021 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

(* CORE_GENERATION_INFO="forward_forward,hls_ip_2021_2,{HLS_INPUT_TYPE=c,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xc7a12ti-csg325-1L,HLS_INPUT_CLOCK=3.333000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=0.000000,HLS_SYN_LAT=0,HLS_SYN_TPT=none,HLS_SYN_MEM=0,HLS_SYN_DSP=0,HLS_SYN_FF=0,HLS_SYN_LUT=0,HLS_VERSION=2021_2}" *)

module forward (
        ap_local_block,
        ap_local_deadlock,
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
        arg_17,
        arg_18,
        arg_19
);


output   ap_local_block;
output   ap_local_deadlock;
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
input  [63:0] arg_9;
input  [63:0] arg_10;
input  [31:0] arg_11;
input  [31:0] arg_12;
input  [63:0] arg_13;
input  [63:0] arg_14;
input  [63:0] arg_15;
input  [63:0] arg_16;
input  [63:0] arg_17;
input  [63:0] arg_18;
input  [63:0] arg_19;

assign ap_done = ap_start;

assign ap_idle = 1'b1;

assign ap_local_block = 1'b0;

assign ap_local_deadlock = 1'b0;

assign ap_ready = ap_start;

endmodule //forward
